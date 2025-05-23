# stage2_pretrain_sr.py
import os
import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose # Compose 可能不需要了，因为 SRDataset 内部处理
from tqdm import tqdm
import math # 用于学习率调度器等
from torch.utils.tensorboard import SummaryWriter
import logging # 导入 logging

# 从 models 导入
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
# 从 data 导入
from data.sr_dataset import SRDataset
# 从 utils 导入
from utils.logger import setup_logger, set_logger_level
from utils.metrics import calculate_psnr, calculate_ssim
from utils.common_utils import get_device
from utils.config_utils import validate_config
from utils.model_utils import load_full_checkpoint, load_model_weights

# torchvision.transforms 已经在上面导入
# from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Pretrain SR Networks (SR_Fast and SR_Quality)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (stage2_sr_pretrain.yaml)")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval in steps. Default: 100. Set 0 to disable.")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Model save interval in steps. Default: 500. Set 0 to disable.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--resume_path_fast", type=str, default=None,
                        help="Path to an SR_Fast checkpoint to resume training from (optional).")
    parser.add_argument("--resume_path_quality", type=str, default=None,
                        help="Path to an SR_Quality checkpoint to resume training from (optional).")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return parser.parse_args()

def get_sr_dataloaders(dataset_config: Dict,
                       train_config_main: Dict, # 主train配置，包含batch_size, num_workers等
                       model_name_for_log: str,
                       logger_instance: logging.Logger
                       ) -> Tuple[Optional[DataLoader], Dict[str, DataLoader]]:
    """
    辅助函数，用于创建训练和验证 SR 数据加载器。
    """
    lr_patch_size = dataset_config.get('patch_size') # LR patch_size for training
    scale_factor = dataset_config.get('scale_factor', 4)
    additional_tensor_transforms = None # SRDataset 内部处理 ToTensor

    # --- 训练数据加载器 ---
    train_lr_dir = os.path.join(dataset_config['base_dir'], "LR")
    train_hr_dir = os.path.join(dataset_config['base_dir'], "HR")
    train_dataloader = None
    try:
        train_dataset = SRDataset(
            lr_dir=train_lr_dir,
            hr_dir=train_hr_dir,
            patch_size=lr_patch_size,
            scale_factor=scale_factor,
            transform=additional_tensor_transforms,
            augment=True # 训练时启用增强
        )
        if len(train_dataset) == 0:
            logger_instance.warning(f"为 {model_name_for_log} 创建的训练数据集为空 (路径: {train_lr_dir}, {train_hr_dir})。")
            # 根据策略，可以返回 None 或引发错误
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_config_main['batch_size'],
                shuffle=True,
                num_workers=train_config_main.get('num_workers', 0),
                pin_memory=True,
                # 处理 SRDataset 在图像加载失败时可能返回 (None, None) 的情况
                collate_fn=lambda batch: torch.utils.data.dataloader.default_collate(
                    [item for item in batch if item is not None and item[0] is not None and item[1] is not None]
                )
            )
            logger_instance.info(f"为 {model_name_for_log} 加载的训练数据集来自 {dataset_config['base_dir']}，"
                                 f"共 {len(train_dataset)} 张图像，使用 LR patch_size: {lr_patch_size}。")
    except FileNotFoundError as e:
        logger_instance.error(f"为 {model_name_for_log} 准备训练数据时文件未找到: {e}")
        return None, {} # 返回 None 表示训练数据加载失败
    except Exception as e:
        logger_instance.error(f"为 {model_name_for_log} 加载训练数据时发生错误: {e}", exc_info=True)
        return None, {}

    # --- 验证数据加载器 ---
    val_dataloaders_dict: Dict[str, DataLoader] = {}
    # train_config_main 中可能有 'evaluation' 部分，或者直接在顶层 config 中
    # 这里假设 eval_config 在 train_config_main['evaluation']
    eval_config = train_config_main.get('evaluation', {})
    val_dataset_config = eval_config.get('val_dataset', {})
    val_base_dir = val_dataset_config.get('base_dir')
    val_sets_names = val_dataset_config.get('sets', []) # sets 是一个列表

    if val_base_dir and val_sets_names:
        for val_set_name in val_sets_names:
            val_lr_path = os.path.join(val_base_dir, val_set_name, "LR")
            val_hr_path = os.path.join(val_base_dir, val_set_name, "HR")
            if os.path.exists(val_lr_path) and os.path.exists(val_hr_path):
                try:
                    val_dataset_instance = SRDataset(
                        lr_dir=val_lr_path,
                        hr_dir=val_hr_path,
                        patch_size=None,  # 验证时通常不使用随机裁剪
                        scale_factor=scale_factor,
                        transform=additional_tensor_transforms,
                        augment=False # 验证时不进行数据增强
                    )
                    if len(val_dataset_instance) == 0:
                        logger_instance.warning(f"为 {model_name_for_log} 创建的验证数据集 '{val_set_name}' 为空。")
                        continue

                    val_dataloaders_dict[val_set_name] = DataLoader(
                        val_dataset_instance,
                        batch_size=train_config_main.get('val_batch_size', 1),
                        shuffle=False,
                        num_workers=train_config_main.get('num_workers', 0),
                        pin_memory=True,
                        collate_fn=lambda batch: torch.utils.data.dataloader.default_collate(
                            [item for item in batch if item is not None and item[0] is not None and item[1] is not None]
                        )
                    )
                    logger_instance.info(f"为 {model_name_for_log} 加载的验证数据集: {val_set_name} (来自 {val_base_dir})，"
                                         f"共 {len(val_dataset_instance)} 张图像。")
                except FileNotFoundError:
                    logger_instance.warning(f"为 {model_name_for_log} 准备验证数据 '{val_set_name}' 时文件未找到。")
                except Exception as e:
                    logger_instance.error(f"为 {model_name_for_log} 加载验证数据 {val_set_name} 时发生错误: {e}", exc_info=True)
            else:
                logger_instance.warning(f"为 {model_name_for_log} 准备的验证数据目录未找到: {val_lr_path} 或 {val_hr_path}")
    else:
        logger_instance.info(f"配置文件中未完整指定 {model_name_for_log} 的验证数据集。将跳过验证集加载。")

    return train_dataloader, val_dataloaders_dict


def run_sr_evaluation(model: nn.Module,
                      model_name: str,
                      val_dataloaders: Dict[str, DataLoader],
                      device: torch.device,
                      logger_instance: logging.Logger,
                      writer: SummaryWriter, # TensorBoard writer
                      current_global_step: int):
    """在所有验证集上运行 SR 模型评估并记录结果。"""
    model.eval() # 设置为评估模式
    all_eval_metrics: Dict[str, Dict[str, float]] = {} # 存储所有验证集的指标
    primary_val_set_name = list(val_dataloaders.keys())[0] if val_dataloaders else None
    # 初始化用于返回的主验证集指标
    avg_primary_psnr = 0.0
    avg_primary_ssim = 0.0

    if not val_dataloaders:
        logger_instance.info(f"{model_name} (步骤 {current_global_step}): 无可用验证数据加载器，跳过评估。")
        model.train() # 恢复训练模式
        return {'psnr': 0.0, 'ssim': 0.0} # 返回默认值

    with torch.no_grad():
        for val_set_key, dataloader_val in val_dataloaders.items():
            total_psnr_set, total_ssim_set, count_set = 0.0, 0.0, 0
            progress_val = tqdm(dataloader_val, desc=f"评估 {model_name} on {val_set_key} (步骤 {current_global_step})", leave=False)
            for lr_val_batch, hr_val_batch in progress_val:
                if lr_val_batch is None or hr_val_batch is None or lr_val_batch.numel() == 0:
                    continue # 跳过无效批次 (例如由collate_fn过滤后为空)
                lr_val_batch, hr_val_batch = lr_val_batch.to(device), hr_val_batch.to(device)
                sr_val_batch = model(lr_val_batch)

                # 逐张图像计算 PSNR/SSIM
                for i in range(sr_val_batch.size(0)):
                    sr_img_single = sr_val_batch[i:i+1] # 保持批次维度
                    hr_img_single = hr_val_batch[i:i+1]
                    try:
                        psnr_val = calculate_psnr(sr_img_single, hr_img_single)
                        ssim_val = calculate_ssim(sr_img_single, hr_img_single)
                        total_psnr_set += psnr_val
                        total_ssim_set += ssim_val
                        count_set += 1
                    except Exception as e_metric:
                        logger_instance.warning(f"在 {val_set_key} 上计算指标时出错: {e_metric}")

            avg_psnr_set = total_psnr_set / count_set if count_set > 0 else 0.0
            avg_ssim_set = total_ssim_set / count_set if count_set > 0 else 0.0

            logger_instance.info(f"{model_name} - 验证步骤 {current_global_step} - {val_set_key}: "
                                 f"PSNR={avg_psnr_set:.2f}, SSIM={avg_ssim_set:.4f} (基于 {count_set} 张图像)")
            all_eval_metrics[val_set_key] = {'psnr': avg_psnr_set, 'ssim': avg_ssim_set}

            if writer:
                writer.add_scalar(f"{model_name}/Validation/{val_set_key}_PSNR_Step", avg_psnr_set, current_global_step)
                writer.add_scalar(f"{model_name}/Validation/{val_set_key}_SSIM_Step", avg_ssim_set, current_global_step)

            if val_set_key == primary_val_set_name:
                avg_primary_psnr = avg_psnr_set
                avg_primary_ssim = avg_ssim_set

    model.train() # 恢复训练模式
    return {'psnr': avg_primary_psnr, 'ssim': avg_primary_ssim}


def train_sr_model(model: nn.Module,
                   model_name: str, # "SR_Fast" or "SR_Quality"
                   model_specific_config: Dict, # config['models']['sr_fast'] or config['models']['sr_quality']
                   main_train_config: Dict,     # config['train']
                   main_dataset_config: Dict,   # config['dataset']
                   logger_instance: logging.Logger,
                   cmd_args: argparse.Namespace, # 命令行参数
                   device: torch.device):        # 从 main 传入的设备
    """训练单个 SR 模型 (Fast 或 Quality)"""
    model.to(device)

    # 初始化训练状态变量
    start_epoch = 0
    global_step_counter = 0 # 使用更明确的名称
    best_metric_val = 0.0 # 通用最佳指标值 (例如 PSNR)
    metric_to_monitor = main_train_config.get('evaluation',{}).get('primary_metric', 'psnr').lower()


    # --- 断点续训逻辑 ---
    resume_path_specific = cmd_args.resume_path_fast if model_name == "SR_Fast" else cmd_args.resume_path_quality
    if resume_path_specific:
        logger_instance.info(f"为 {model_name} 尝试从检查点恢复训练: {resume_path_specific}")
        checkpoint = load_full_checkpoint(resume_path_specific, device, logger_instance)
        if checkpoint:
            # 尝试加载模型权重，即使模型结构可能已在外部初始化
            # 如果模型已由 main 初始化，则 load_model_weights 会加载权重
            # 如果模型需要从检查点中的配置重新构建，则需要更复杂的逻辑 (目前未实现)
            if 'model_state_dict' in checkpoint:
                load_model_weights(model, resume_path_specific, device, model_name, logger_instance, strict=False)

            # 后续加载 optimizer 和 scheduler 状态
            # global_step_counter 和 best_metric_val 会在下面从 checkpoint 中获取
        else:
            logger_instance.warning(f"为 {model_name} 加载检查点 {resume_path_specific} 失败。将从头开始训练。")

    # --- 数据加载 ---
    train_loader, val_loaders_map = get_sr_dataloaders(main_dataset_config, main_train_config, model_name, logger_instance)
    if train_loader is None:
        logger_instance.error(f"无法为 {model_name} 创建训练数据加载器。训练中止。")
        return

    # --- 优化器和调度器 ---
    optimizer_cfg = main_train_config.get('optimizer', {})
    optimizer_name_str = optimizer_cfg.get('name', 'Adam').lower()
    optimizer_args_dict = optimizer_cfg.get('args', {'lr': 1e-4}) # 提供默认lr

    if optimizer_name_str == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_args_dict)
    elif optimizer_name_str == 'adamw':
         optimizer = optim.AdamW(model.parameters(), **optimizer_args_dict)
    else:
        logger_instance.error(f"{model_name}: 不支持的优化器: {optimizer_name_str}")
        return
    logger_instance.info(f"{model_name}: 使用优化器 {optimizer_name_str}，参数: {optimizer_args_dict}")

    scheduler = None
    scheduler_cfg = main_train_config.get('scheduler', {})
    scheduler_name_str = scheduler_cfg.get('name', '').lower()
    scheduler_args_dict = scheduler_cfg.get('args', {})

    if scheduler_name_str == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_args_dict)
    elif scheduler_name_str == 'cosineannealinglr':
        # T_max 通常是总步数或总轮数
        if 'T_max' not in scheduler_args_dict:
            scheduler_args_dict['T_max'] = main_train_config['epochs'] * len(train_loader) if train_loader else main_train_config['epochs']
            logger_instance.info(f"{model_name}: CosineAnnealingLR 的 T_max 未在配置中指定，已计算为 {scheduler_args_dict['T_max']} (总步数)。")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args_dict)
    elif scheduler_name_str:
        logger_instance.warning(f"{model_name}: 不支持的调度器: {scheduler_name_str}。将不使用调度器。")

    if scheduler:
        logger_instance.info(f"{model_name}: 使用调度器 {scheduler_name_str}，参数: {scheduler_args_dict}")


    # --- 如果从检查点恢复，加载优化器和调度器状态 ---
    if resume_path_specific and checkpoint: # checkpoint 来自之前的加载尝试
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger_instance.info(f"{model_name}: 已从检查点加载优化器状态。")
            except Exception as e_opt:
                logger_instance.warning(f"{model_name}: 从检查点加载优化器状态失败: {e_opt}。优化器将重新开始。")
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger_instance.info(f"{model_name}: 已从检查点加载调度器状态。")
            except Exception as e_sch:
                logger_instance.warning(f"{model_name}: 从检查点加载调度器状态失败: {e_sch}。调度器将重新开始。")

        start_epoch = checkpoint.get('epoch', 0) + 1 # 从下一轮开始
        global_step_counter = checkpoint.get('step', 0) # 或 'global_step'
        # 从检查点恢复最佳指标值，确保键名一致
        best_metric_val = checkpoint.get(metric_to_monitor, 0.0)
        logger_instance.info(f"{model_name}: 从检查点恢复，将从 Epoch {start_epoch +1}, Global Step {global_step_counter} 开始。记录的最佳 {metric_to_monitor}: {best_metric_val:.4f}")


    # --- 损失函数 ---
    loss_type_str = main_train_config.get('loss', 'L1').upper()
    if loss_type_str == 'L1':
        criterion = nn.L1Loss()
    elif loss_type_str == 'MSE':
        criterion = nn.MSELoss()
    else:
        logger_instance.error(f"{model_name}: 不支持的损失类型: {loss_type_str}")
        return
    criterion.to(device)
    logger_instance.info(f"{model_name}: 使用损失函数: {loss_type_str}")

    # --- TensorBoard Writer ---
    # 使用 config['train']['log_dir'] 作为基础日志目录
    tb_log_dir = os.path.join(main_train_config.get('log_dir', './temp_logs/stage2_sr'), f"{model_name}_tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger_instance.info(f"{model_name}: TensorBoard 日志将保存到: {tb_log_dir}")

    # --- 训练循环 ---
    logger_instance.info(f"--- 开始为 {model_name} 进行训练 (从 Epoch {start_epoch + 1}) ---")
    logger_instance.info(f"总轮数: {main_train_config['epochs']}, 批次大小: {main_train_config['batch_size']}")
    logger_instance.info(f"初始学习率: {optimizer.param_groups[0]['lr']:.6f}")


    output_model_path = model_specific_config['output_path'] # 模型最终保存路径
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    num_epochs_to_run = main_train_config['epochs']

    for epoch_idx in range(start_epoch, num_epochs_to_run):
        model.train() # 设置模型为训练模式
        epoch_accumulated_loss = 0.0
        progress_bar_train = tqdm(train_loader, desc=f"{model_name} Epoch {epoch_idx+1}/{num_epochs_to_run}", total=len(train_loader))

        for lr_imgs_batch, hr_imgs_batch in progress_bar_train:
            if lr_imgs_batch is None or hr_imgs_batch is None or lr_imgs_batch.numel() == 0:
                # logger_instance.debug(f"{model_name} (步骤 {global_step_counter}): 跳过空批次。")
                continue # collate_fn 应该处理了这个问题

            lr_imgs_batch, hr_imgs_batch = lr_imgs_batch.to(device), hr_imgs_batch.to(device)

            optimizer.zero_grad()
            sr_imgs_batch = model(lr_imgs_batch)
            loss = criterion(sr_imgs_batch, hr_imgs_batch)

            if torch.isnan(loss):
                logger_instance.error(f"{model_name} (步骤 {global_step_counter}): 损失为 NaN。停止训练。")
                writer.close()
                return

            loss.backward()
            optimizer.step()

            epoch_accumulated_loss += loss.item()
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            # --- TensorBoard 日志 (每步) ---
            if writer:
                writer.add_scalar(f"{model_name}/Train/Loss_Step", loss.item(), global_step_counter)
                writer.add_scalar(f"{model_name}/Train/LearningRate_Step", optimizer.param_groups[0]['lr'], global_step_counter)

            global_step_counter += 1

            # --- 基于步骤的评估和保存 ---
            if val_loaders_map and cmd_args.eval_interval > 0 and global_step_counter % cmd_args.eval_interval == 0:
                logger_instance.info(f"{model_name} (步骤 {global_step_counter}): 运行评估...")
                current_metrics = run_sr_evaluation(model, model_name, val_loaders_map, device, logger_instance, writer, global_step_counter)
                current_metric_val_to_check = current_metrics.get(metric_to_monitor, 0.0)

                if current_metric_val_to_check > best_metric_val:
                    best_metric_val = current_metric_val_to_check
                    best_model_save_path = output_model_path.replace('.pth', '_best.pth')
                    checkpoint_data_to_save = {
                        'epoch': epoch_idx,
                        'step': global_step_counter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        metric_to_monitor: best_metric_val, # 保存被监控的指标
                        'config_model': model_specific_config, # 保存模型特定配置
                        'config_train': main_train_config,   # 保存主要训练配置
                        'config_dataset': main_dataset_config # 保存数据集配置
                    }
                    if scheduler:
                        checkpoint_data_to_save['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(checkpoint_data_to_save, best_model_save_path)
                    logger_instance.info(
                        f"{model_name} (步骤 {global_step_counter}): 保存最佳模型 ({metric_to_monitor}: {best_metric_val:.4f}) 到 {best_model_save_path}"
                    )

            if cmd_args.save_interval > 0 and global_step_counter % cmd_args.save_interval == 0:
                step_model_save_path = output_model_path.replace('.pth', f'_step{global_step_counter}.pth')
                checkpoint_data_to_save = {
                    'epoch': epoch_idx,
                    'step': global_step_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    metric_to_monitor: current_metrics.get(metric_to_monitor, 0.0) if 'current_metrics' in locals() else 0.0,
                    'config_model': model_specific_config,
                    'config_train': main_train_config,
                    'config_dataset': main_dataset_config
                }
                if scheduler:
                    checkpoint_data_to_save['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint_data_to_save, step_model_save_path)
                logger_instance.info(f"{model_name} (步骤 {global_step_counter}): 保存检查点到 {step_model_save_path}")

        # --- Epoch 结束 ---
        avg_epoch_loss_val = epoch_accumulated_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0.0
        logger_instance.info(f"{model_name} - Epoch {epoch_idx+1}/{num_epochs_to_run} 完成。平均损失: {avg_epoch_loss_val:.4f}")
        if writer:
            writer.add_scalar(f"{model_name}/Train/Loss_Epoch", avg_epoch_loss_val, epoch_idx + 1)

        if scheduler and not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR): # CosineAnnealingLR 每步更新
            scheduler.step()
            logger_instance.info(f"{model_name}: 调度器已步进 (Epoch结束)。新学习率: {optimizer.param_groups[0]['lr']:.6f}")


    # --- 训练完成 ---
    final_checkpoint_data = {
        'epoch': num_epochs_to_run -1, # 保存完成的 epoch 数
        'step': global_step_counter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        metric_to_monitor: best_metric_val, # 保存最后记录的最佳指标
        'config_model': model_specific_config,
        'config_train': main_train_config,
        'config_dataset': main_dataset_config
    }
    if scheduler:
        final_checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(final_checkpoint_data, output_model_path)
    logger_instance.info(f"为 {model_name} 保存最终模型到 {output_model_path}")
    logger_instance.info(f"--- {model_name} 训练完成 ---")

    if writer:
        writer.close()


def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件在 {args.config} 未找到") # 在 logger 初始化前，用 print
        exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件时出错: {e}")
        exit(1)

    # 日志和检查点目录 (从配置文件中读取，提供默认值)
    # config['train'] 可能不存在，所以用 .get
    train_conf = config.get('train', {})
    log_dir_base = train_conf.get('log_dir', './temp_logs/stage2_sr')
    # 确保基础日志目录存在，setup_logger 会处理子目录
    os.makedirs(log_dir_base, exist_ok=True)
    logger = setup_logger(log_dir_base, "stage2_pretrain_sr_main.log") # 主日志文件

    # 根据命令行参数设置日志级别
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logger.critical(f'无效的日志级别: {args.log_level}')
        raise ValueError(f'无效的日志级别: {args.log_level}')
    set_logger_level(logger, numeric_log_level)
    logger.info(f"日志级别已设置为: {args.log_level}")


    logger.info("--- 开始阶段 2: SR 网络预训练 ---")
    logger.info(f"已从以下路径加载配置: {args.config}")
    logger.info(f"命令行参数: {args}")

    # --- 设备选择 ---
    device = get_device(args.use_gpu, logger)
    logger.info(f"将使用设备: {device} 进行训练。")
    # 注意：旧代码中 config['train']['device'] = device 的操作现在由 get_device 内部处理日志，
    # 并且 device 对象直接传递给训练函数。

    # --- 配置校验 ---
    # 将 args 添加到 config 中，以便 validate_config 访问
    config['args'] = vars(args)
    if not validate_config(config, "stage2_sr", logger):
        logger.error("配置校验失败。正在退出。")
        return
    # 从已校验的 config 中安全地获取参数
    dataset_cfg = config['dataset']
    models_cfg = config['models'] # stage2 的模型配置在 'models' 下
    train_cfg = config['train']


    # --- 训练 SR_Fast ---
    logger.info("--- 初始化 SR_Fast 模型并开始训练 ---")
    sr_fast_model_cfg = models_cfg.get('sr_fast', {})
    sr_fast_init_args = {
        'in_channels': sr_fast_model_cfg.get('in_channels', 3),
        'd': sr_fast_model_cfg.get('d', 56),
        's': sr_fast_model_cfg.get('s', 12),
        'm': sr_fast_model_cfg.get('m', 4),
        'scale_factor': sr_fast_model_cfg.get('scale_factor', dataset_cfg.get('scale_factor', 4))
    }
    sr_fast_instance = SRFast(**sr_fast_init_args)
    train_sr_model(sr_fast_instance, "SR_Fast",
                   sr_fast_model_cfg, train_cfg, dataset_cfg,
                   logger, args, device)

    # --- 训练 SR_Quality ---
    logger.info("--- 初始化 SR_Quality 模型并开始训练 ---")
    sr_quality_model_cfg = models_cfg.get('sr_quality', {})
    sr_quality_init_args = {
        'in_channels': sr_quality_model_cfg.get('in_channels', 3),
        'num_channels': sr_quality_model_cfg.get('num_channels', 64),
        'num_blocks': sr_quality_model_cfg.get('num_blocks', 16),
        'scale_factor': sr_quality_model_cfg.get('scale_factor', dataset_cfg.get('scale_factor', 4))
    }
    sr_quality_instance = SRQuality(**sr_quality_init_args)
    train_sr_model(sr_quality_instance, "SR_Quality",
                   sr_quality_model_cfg, train_cfg, dataset_cfg,
                   logger, args, device)

    logger.info("--- 阶段 2 SR 网络预训练完成 ---")

if __name__ == "__main__":
    main()