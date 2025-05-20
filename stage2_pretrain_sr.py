import os
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomCrop, ToPILImage
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter

from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from data.sr_dataset import SRDataset
from utils.logger import setup_logger
from utils.metrics import calculate_psnr, calculate_ssim
from data.sr_dataset import SRDataset
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Pretrain SR Networks (SR_Fast and SR_Quality)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage2_sr_pretrain.yaml)")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Model save interval in steps")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

# def get_transforms(config):
#     patch_size = config['dataset'].get('patch_size', None)
#     transforms_list = []
#     if patch_size:
#         print(f"Warning: Patch cropping not fully implemented. Applying ToTensor directly.")
#         print("Consider implementing HR patch cropping and corresponding LR generation.")
#         transforms_list.append(ToTensor())
#     else:
#         transforms_list.append(ToTensor())
#     return Compose(transforms_list)

def run_sr_evaluation(model, model_name, val_dataloaders, device, logger, writer, global_step):
    """在所有验证集上运行 SR 模型评估并记录结果。"""
    model.eval()
    all_metrics = {}
    primary_val_set = list(val_dataloaders.keys())[0] if val_dataloaders else None
    avg_primary_psnr = 0.0
    avg_primary_ssim = 0.0

    with torch.no_grad():
        for val_set, dataloader in val_dataloaders.items():
            total_psnr, total_ssim, count = 0, 0, 0
            for lr_val, hr_val in tqdm(dataloader, desc=f"Evaluating {model_name} on {val_set}", leave=False):
                if lr_val is None or hr_val is None: continue
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                sr_val = model(lr_val)
                total_psnr += calculate_psnr(sr_val, hr_val)
                total_ssim += calculate_ssim(sr_val, hr_val)
                count += 1

            avg_psnr = total_psnr / count if count > 0 else 0
            avg_ssim = total_ssim / count if count > 0 else 0
            logger.info(f"Validation Step {global_step} - {val_set}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
            all_metrics[val_set] = {'psnr': avg_psnr, 'ssim': avg_ssim}
            if writer:
                writer.add_scalar(f"{model_name}/Validation/{val_set}_PSNR", avg_psnr, global_step)
                writer.add_scalar(f"{model_name}/Validation/{val_set}_SSIM", avg_ssim, global_step)

            if val_set == primary_val_set:
                avg_primary_psnr = avg_psnr
                avg_primary_ssim = avg_ssim

    model.train()
    return {'psnr': avg_primary_psnr, 'ssim': avg_primary_ssim}

def train_sr_model(model, model_name, model_config, train_config, dataset_config, logger, args):
    """训练单个 SR 模型 (Fast 或 Quality)"""
    device = torch.device(train_config['device'])
    model.to(device)

    # 从YAML的 dataset 部分获取 patch_size 和 scale_factor
    # dataset_config 参数就是从YAML加载的 config['dataset']
    lr_patch_size = dataset_config.get('patch_size', None)
    scale_factor = dataset_config.get('scale_factor', 4) # 确保配置文件中有 scale_factor
    
    # 定义应用于Tensor的额外转换 (例如归一化，如果需要的话)
    # SRDataset 内部已经处理了 PIL Image -> Tensor 的转换
    # 所以这里的 transform 是用于 Tensor 格式的数据
    additional_tensor_transforms = None
    # 例如，如果您需要归一化:
    # additional_tensor_transforms = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_lr_dir = os.path.join(dataset_config['base_dir'], "LR")
    train_hr_dir = os.path.join(dataset_config['base_dir'], "HR")
    try:
        train_dataset = SRDataset(
            lr_dir=train_lr_dir,
            hr_dir=train_hr_dir,
            patch_size=lr_patch_size,         # 传递LR patch_size
            scale_factor=scale_factor,        # 传递scale_factor
            transform=additional_tensor_transforms, # 传递Tensor的额外转换
            augment=True                      # 训练时启用内置增强
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=True,
            # 添加 collate_fn 以处理 SRDataset 在图像加载失败时可能返回 (None, None) 的情况
            collate_fn=lambda batch: torch.utils.data.dataloader.default_collate([item for item in batch if item[0] is not None and item[1] is not None])
        )
        logger.info(f"训练数据集已从 {dataset_config['base_dir']} 加载，使用 patch_size: {lr_patch_size}")
    except FileNotFoundError as e:
        logger.error(f"训练数据未找到 ({model_name}): {e}")
        return
    except Exception as e: # 更通用的异常捕获
        logger.error(f"加载训练数据时发生错误 ({model_name}): {e}")
        return

    val_dataloaders = {}
    if 'evaluation' in train_config and 'val_dataset' in train_config['evaluation']:
        val_eval_config = train_config['evaluation']['val_dataset'] # evaluation下的val_dataset部分
        val_base_dir = val_eval_config.get('base_dir')
        val_sets = val_eval_config.get('sets', []) # sets应该是一个列表

        if val_base_dir and val_sets:
            for val_set_name in val_sets:
                val_lr_dir = os.path.join(val_base_dir, val_set_name, "LR")
                val_hr_dir = os.path.join(val_base_dir, val_set_name, "HR")
                if os.path.exists(val_lr_dir) and os.path.exists(val_hr_dir):
                    try:
                        val_dataset = SRDataset(
                            lr_dir=val_lr_dir,
                            hr_dir=val_hr_dir,
                            patch_size=None,  # 验证时通常不使用随机裁剪的patch
                            scale_factor=scale_factor,
                            transform=additional_tensor_transforms, # 应用同样的额外Tensor转换
                            augment=False # 验证时不进行数据增强
                        )
                        val_dataloaders[val_set_name] = DataLoader(
                            val_dataset,
                            batch_size=train_config.get('val_batch_size', 1), # 验证时batch_size通常为1
                            shuffle=False,
                            num_workers=train_config.get('num_workers', 1),
                            pin_memory=True,
                            collate_fn=lambda batch: torch.utils.data.dataloader.default_collate([item for item in batch if item[0] is not None and item[1] is not None])
                        )
                        logger.info(f"已加载验证数据集: {val_set_name} 从 {val_base_dir}")
                    except FileNotFoundError:
                        logger.warning(f"验证数据文件未找到: {val_set_name} 在 {val_base_dir}")
                    except Exception as e:
                        logger.error(f"加载验证数据 {val_set_name} 时发生错误: {e}")
                else:
                    logger.warning(f"验证数据目录未找到: {val_lr_dir} 或 {val_hr_dir}")
        else:
            logger.info("配置文件中未完整指定验证数据集 (evaluation.val_dataset.base_dir 或 sets)。跳过验证集加载。")

    optimizer_name = train_config['optimizer']['name'].lower()
    optimizer_args = train_config['optimizer']['args']
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_args)
    elif optimizer_name == 'adamw':
         optimizer = optim.AdamW(model.parameters(), **optimizer_args)
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}")
        return

    scheduler_name = train_config['scheduler']['name'].lower()
    scheduler_args = train_config['scheduler']['args']
    if scheduler_name == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
    elif scheduler_name == 'cosineannealinglr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
    else:
        logger.warning(f"Unsupported scheduler: {scheduler_name}. Using no scheduler.")
        scheduler = None

    loss_type = train_config['loss'].upper()
    if loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        logger.error(f"Unsupported loss type: {loss_type}")
        return
    criterion.to(device)

    writer = SummaryWriter(log_dir=os.path.join(train_config['log_dir'], f"{model_name}_tensorboard"))

    global_step = 0
    logger.info(f"--- Starting Training for {model_name} ---")
    logger.info(f"Epochs: {train_config['epochs']}, Batch Size: {train_config['batch_size']}")
    logger.info(f"Optimizer: {optimizer_name}, LR: {optimizer_args['lr']}")
    logger.info(f"Scheduler: {scheduler_name}")
    logger.info(f"Loss: {loss_type}")

    best_psnr = 0.0
    output_path = model_config['output_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for epoch in range(train_config['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"{model_name} Epoch {epoch+1}/{train_config['epochs']}")

        for lr_imgs, hr_imgs in progress_bar:
            if lr_imgs is None or hr_imgs is None: continue

            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            
            if torch.isnan(loss):
                logger.error("Loss is NaN. Stopping training.")
                return

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            global_step += 1

            writer.add_scalar(f"{model_name}/Train/Loss", loss.item(), global_step)
            writer.add_scalar(f"{model_name}/Train/LearningRate", optimizer.param_groups[0]['lr'], global_step)

            if val_dataloaders and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                logger.info(f"Step {global_step}: Running evaluation...")
                avg_metrics = run_sr_evaluation(model, model_name, val_dataloaders, device, logger, writer, global_step)

                current_psnr = avg_metrics.get('psnr', 0.0)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    save_path = output_path.replace('.pth', '_best.pth')
                    torch.save({'step': global_step, 'model_state_dict': model.state_dict(), 'psnr': best_psnr}, save_path)
                    logger.info(f"Saved best {model_name} model (Step {global_step}, PSNR: {best_psnr:.2f}) to {save_path}")

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_path = output_path.replace('.pth', f'_step{global_step}.pth')
                torch.save({'step': global_step, 'model_state_dict': model.state_dict()}, save_path)
                logger.info(f"Saved {model_name} checkpoint to {save_path}")

        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logger.info(f"{model_name} Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        if scheduler:
            scheduler.step()

    torch.save({'epoch': train_config['epochs'], 'step': global_step, 'model_state_dict': model.state_dict()}, output_path)
    logger.info(f"Saved final {model_name} model to {output_path}")
    logger.info(f"--- Finished Training for {model_name} ---")

    writer.close()

def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    logger = setup_logger(config['log_dir'], "stage2_train.log")
    logger.info("Starting Stage 2: SR Network Pretraining")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Step-based Eval Interval: {args.eval_interval}, Save Interval: {args.save_interval}")

    # --- 配置校验 ---
    logger.info("--- Validating Configuration ---")
    dataset_config = config.get('dataset', {})
    train_base_dir = dataset_config.get('base_dir')
    scale_factor = dataset_config.get('scale_factor')

    if not train_base_dir or not os.path.exists(train_base_dir):
        logger.error(f"Training dataset base directory not found: {train_base_dir}. Exiting.")
        exit(1)
    if not isinstance(scale_factor, int) or scale_factor <= 0:
         logger.error(f"Invalid scale_factor in dataset config: {scale_factor}. Must be a positive integer. Exiting.")
         exit(1)

    sr_fast_config = config.get('models', {}).get('sr_fast', {})
    sr_quality_config = config.get('models', {}).get('sr_quality', {})

    if sr_fast_config.get('scale_factor') != scale_factor:
        logger.error(f"Scale factor mismatch: dataset ({scale_factor}) vs sr_fast ({sr_fast_config.get('scale_factor')}). Exiting.")
        exit(1)
    if sr_quality_config.get('scale_factor') != scale_factor:
        logger.error(f"Scale factor mismatch: dataset ({scale_factor}) vs sr_quality ({sr_quality_config.get('scale_factor')}). Exiting.")
        exit(1)

    logger.info("--- Configuration Validated ---")
    # --- End Configuration Validation ---

    logger.info("Initializing SR_Fast model...")
    sr_fast_config = config['models']['sr_fast']
    sr_fast_model = SRFast(**sr_fast_config)
    # train_sr_model(sr_fast_model, "SR_Fast", sr_fast_config, config['train'], config['dataset'], logger, args)
    train_sr_model(sr_fast_model, "SR_Fast", config['models']['sr_fast'], config['train'], config['dataset'], logger, args)
    train_sr_model(sr_quality_model, "SR_Quality", config['models']['sr_quality'], config['train'], config['dataset'], logger, args)

    logger.info("Initializing SR_Quality model...")
    sr_quality_config = config['models']['sr_quality']
    sr_quality_model = SRQuality(**sr_quality_config)
    train_sr_model(sr_quality_model, "SR_Quality", sr_quality_config, config['train'], config['dataset'], logger, args)

    logger.info("Stage 2 finished.")

if __name__ == "__main__":
    main()
