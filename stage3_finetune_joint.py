import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import setup_logger
from utils.losses import calculate_joint_loss
from utils.optimizer_utils import get_optimizer_with_differential_lr
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from torch.utils.tensorboard import SummaryWriter
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from utils.evaluation_utils import run_coco_evaluation
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Finetuning (ConditionalSR + YOLO)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage3_joint_finetune.yaml)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model save interval in steps")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def train_joint(config, logger, args):
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available, falling back to CPU")
    logger.info(f"Using device: {device}")

    # --- 配置校验 ---
    logger.info("--- Validating Configuration ---")
    dataset_config = config.get('dataset', {})
    image_dir = dataset_config.get('image_dir')
    annotation_file = dataset_config.get('annotation_file')
    scale_factor = dataset_config.get('scale_factor')

    if not image_dir or not os.path.exists(image_dir):
        logger.error(f"Training image directory not found: {image_dir}. Exiting.")
        return
    if not annotation_file or not os.path.exists(annotation_file):
        logger.error(f"Training annotation file not found: {annotation_file}. Exiting.")
        return
    if not isinstance(scale_factor, int) or scale_factor <= 0:
         logger.error(f"Invalid scale_factor in dataset config: {scale_factor}. Must be a positive integer. Exiting.")
         return

    model_config = config.get('model', {})
    sr_fast_config = model_config.get('sr_fast', {})
    sr_quality_config = model_config.get('sr_quality', {})
    masker_config = model_config.get('masker', {})
    weights_config = model_config.get('weights', {})

    if sr_fast_config.get('scale_factor') != scale_factor:
        logger.error(f"Scale factor mismatch: dataset ({scale_factor}) vs sr_fast ({sr_fast_config.get('scale_factor')}). Exiting.")
        return
    if sr_quality_config.get('scale_factor') != scale_factor:
        logger.error(f"Scale factor mismatch: dataset ({scale_factor}) vs sr_quality ({sr_quality_config.get('scale_factor')}). Exiting.")
        return

    masker_patch_size = masker_config.get('output_patch_size')
    if not isinstance(masker_patch_size, int) or masker_patch_size <= 0:
         logger.error(f"Invalid output_patch_size in masker config: {masker_patch_size}. Must be a positive integer. Exiting.")
         return
    # Basic compatibility check: scale_factor should be divisible by patch_size
    if scale_factor % masker_patch_size != 0:
         logger.warning(f"Scale factor ({scale_factor}) is not divisible by masker output_patch_size ({masker_patch_size}). This might lead to issues with mask alignment.")

    masker_threshold = masker_config.get('threshold')
    if not isinstance(masker_threshold, (int, float)) or not (0 <= masker_threshold <= 1):
         logger.warning(f"Masker threshold ({masker_threshold}) is outside the expected range [0, 1].")

    target_sparsity_ratio = config.get('train', {}).get('target_sparsity_ratio')
    if target_sparsity_ratio is not None and (not isinstance(target_sparsity_ratio, (int, float)) or not (0 <= target_sparsity_ratio <= 1)):
         logger.warning(f"Target sparsity ratio ({target_sparsity_ratio}) is outside the expected range [0, 1].")


    # Check required weights paths
    required_weights = ['detector', 'sr_fast', 'sr_quality']
    for weight_key in required_weights:
        weight_path = weights_config.get(weight_key)
        if not weight_path or not os.path.exists(weight_path):
            logger.error(f"Required weight file not found for '{weight_key}': {weight_path}. Exiting.")
            return

    eval_config = config.get('evaluation', {})
    if eval_config.get('val_image_dir') or eval_config.get('val_annotation_file'):
        if not eval_config.get('val_image_dir') or not os.path.exists(eval_config['val_image_dir']):
             logger.warning(f"Validation image directory not found: {eval_config.get('val_image_dir')}. Evaluation might be skipped or fail.")
        if not eval_config.get('val_annotation_file') or not os.path.exists(eval_config['val_annotation_file']):
             logger.warning(f"Validation annotation file not found: {eval_config.get('val_annotation_file')}. Evaluation might be skipped or fail.")


    logger.info("--- Configuration Validated ---")
    # --- End Configuration Validation ---

    dataset_config = config['dataset']
    try:
        train_dataset = DetectionDataset(
            image_dir=os.path.join(dataset_config['image_dir'], "LR"),
            annotation_file=dataset_config['annotation_file'],
            transform=transforms.ToTensor(),
            return_image_id=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['train']['num_workers'],
            pin_memory=True,
            collate_fn=getattr(train_dataset, 'collate_fn', None)
        )
        logger.info(f"Train dataloader initialized with {len(train_dataset)} images.")
    except FileNotFoundError as e:
        logger.error(f"Training data/annotation file not found: {e}. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error initializing training dataloader: {e}. Exiting.")
        return

    eval_config = config.get('evaluation', {})
    val_dataloader = None
    if eval_config.get('val_image_dir') and eval_config.get('val_annotation_file'):
        try:
            val_dataset = DetectionDataset(
                image_dir=os.path.join(eval_config['val_image_dir'], "LR"),
                annotation_file=eval_config['val_annotation_file'],
                transform=transforms.ToTensor(),
                return_image_id=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['train'].get('val_batch_size', 1),
                shuffle=False,
                num_workers=config['train']['num_workers'],
                pin_memory=True,
                collate_fn=getattr(val_dataset, 'collate_fn', None)
            )
            logger.info(f"Validation dataloader initialized with {len(val_dataset)} images.")
        except FileNotFoundError as e:
            logger.error(f"Validation data/annotation file not found: {e}. Evaluation will be skipped.")
        except Exception as e:
            logger.error(f"Error initializing validation dataloader: {e}. Evaluation will be skipped.")
    else:
        logger.warning("Validation image directory or annotation file not specified in config. Evaluation will be skipped.")

    sr_fast_config = config['model']['sr_fast']
    sr_quality_config = config['model']['sr_quality']
    masker_config_full = config['model']['masker'] 
    
    valid_masker_init_keys = [
    'in_channels', 
    'base_channels', 
    'num_blocks', 
    'output_channels', 
    'output_patch_size'
]
    masker_init_args = {}
    for key in valid_masker_init_keys:
        if key in masker_config_full: # 确保配置中存在该键
            masker_init_args[key] = masker_config_full[key]
        else:
            logger.warning(f"Masker init argument '{key}' not found in config, using default or model's default.")

    sr_fast_model = SRFast(**sr_fast_config)
    sr_quality_model = SRQuality(**sr_quality_config)
    masker_model = Masker(**masker_init_args)

    conditional_sr = ConditionalSR(
        sr_fast=sr_fast_model,
        sr_quality=sr_quality_model,
        masker=masker_model,
        detector_weights=config['model']['weights']['detector'],
        sr_fast_weights=config['model']['weights']['sr_fast'],
        sr_quality_weights=config['model']['weights']['sr_quality'],
        masker_weights=config['model']['weights'].get('masker', None),
        device=device,
        config=config
    ).to(device)

    optimizer = get_optimizer_with_differential_lr(conditional_sr, config)
    scheduler = None
    if 'scheduler' in config['train'] and config['train']['scheduler'].get('name', '').lower() == 'cosineannealinglr':
        scheduler_args = config['train']['scheduler'].get('args', {})
        total_steps = config['train']['epochs'] * len(train_dataloader)
        if 'T_max' in scheduler_args and 'eta_min' in scheduler_args:
            if scheduler_args['T_max'] > config['train']['epochs']:
                scheduler_T_max = scheduler_args['T_max']
            else:
                scheduler_T_max = scheduler_args['T_max'] * len(train_dataloader)
                logger.warning(f"Assuming scheduler T_max ({scheduler_args['T_max']}) is in epochs. Converting to steps: {scheduler_T_max}")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_T_max,
                eta_min=scheduler_args['eta_min']
            )
            logger.info(f"Using CosineAnnealingLR scheduler with T_max={scheduler_T_max} steps.")
        else:
            logger.warning("Scheduler config missing T_max or eta_min. Scheduler disabled.")
    else:
        logger.warning("Scheduler not configured or not CosineAnnealingLR. Scheduler disabled.")

    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], "tensorboard"))

    logger.info("Starting joint finetuning...")
    best_map = 0.0
    global_step = 0

    gumbel_config = config['train'].get('gumbel', {})
    initial_tau = gumbel_config.get('initial_tau', 1.0)
    final_tau = gumbel_config.get('final_tau', 0.1)
    anneal_steps = gumbel_config.get('anneal_steps', config['train']['epochs'] * len(train_dataloader))
    anneal_schedule = gumbel_config.get('anneal_schedule', 'linear').lower()
    use_annealing = anneal_steps > 0 and initial_tau != final_tau
    logger.info(f"Gumbel Annealing: Use={use_annealing}, Initial Tau={initial_tau}, Final Tau={final_tau}, Anneal Steps={anneal_steps}, Schedule={anneal_schedule}")

    for epoch in range(config['train']['epochs']):
        conditional_sr.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")

        for lr_images, targets in progress_bar:
            lr_images = lr_images.to(device)
            targets_on_device = []
            if targets:
                for t in targets:
                    target_device = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            target_device[k] = v.to(device)
                        else:
                            target_device[k] = v
                    targets_on_device.append(target_device)
            else:
                targets_on_device = None

            current_tau = initial_tau
            if use_annealing:
                if global_step < anneal_steps:
                    if anneal_schedule == 'linear':
                        anneal_progress = global_step / anneal_steps
                        current_tau = initial_tau - (initial_tau - final_tau) * anneal_progress
                    elif anneal_schedule == 'cosine':
                        current_tau = final_tau + 0.5 * (initial_tau - final_tau) * (1 + math.cos(math.pi * global_step / anneal_steps))
                    else:
                        logger.warning(f"Unknown anneal_schedule: {anneal_schedule}. Using initial_tau.")
                else:
                    current_tau = final_tau
                current_tau = max(final_tau, current_tau)

            optimizer.zero_grad()
            outputs = conditional_sr(lr_images, targets=targets_on_device, temperature=current_tau)

            sr_images = outputs["sr_image"]
            mask_coarse = outputs["mask_coarse"]
            precomputed_detection_loss = outputs["detection_loss"]

            total_loss, loss_dict = calculate_joint_loss(
                sr_images=sr_images,
                mask_coarse=mask_coarse,
                targets=targets_on_device,
                detector=conditional_sr.detector,
                config=config,
                logger=logger,
                precomputed_detection_loss=precomputed_detection_loss
            )

            if torch.isnan(total_loss):
                logger.error(f"NaN loss detected at step {global_step}. Stopping training.")
                writer.close()
                return

            total_loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                "total_loss": total_loss.item(),
                "det": loss_dict.get("loss_detection", 0.0),
                "spar": loss_dict.get("loss_sparsity", 0.0),
                "smooth": loss_dict.get("loss_smooth", 0.0),
                "tau": current_tau
            })

            log_interval_steps = config['train'].get('log_interval_steps', 10)
            if global_step % log_interval_steps == 0:
                writer.add_scalar("Train/TotalLoss", total_loss.item(), global_step)
                writer.add_scalar("Train/GumbelTau", current_tau, global_step)
                for loss_name, loss_val in loss_dict.items():
                    if isinstance(loss_val, (int, float)):
                        writer.add_scalar(f"Train/Loss_{loss_name}", loss_val, global_step)
                    elif torch.is_tensor(loss_val) and loss_val.numel() == 1:
                        writer.add_scalar(f"Train/Loss_{loss_name}", loss_val.item(), global_step)

                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Train/LR_Group_{i}", param_group['lr'], global_step)

                if mask_coarse is not None:
                    actual_sparsity = torch.mean(mask_coarse).item()
                    writer.add_scalar("Train/Mask_ActualSparsity", actual_sparsity, global_step)

                    if mask_coarse.min() >= 0 and mask_coarse.max() <= 1:
                        p = mask_coarse
                        epsilon = 1e-10
                        entropy = - (p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon)).mean().item()
                        writer.add_scalar("Train/Mask_Entropy", entropy, global_step)

            if val_dataloader and args.eval_interval > 0 and global_step > 0 and global_step % args.eval_interval == 0:
                logger.info(f"Step {global_step}: Running evaluation...")
                map_results, avg_sparsity = run_coco_evaluation(
                    model=conditional_sr,
                    dataloader=val_dataloader,
                    device=device,
                    annotation_file=config['evaluation']['val_annotation_file'],
                    output_dir=config['log_dir'],
                    step_or_epoch=global_step,
                    logger=logger,
                    use_hard_mask=True
                )

                writer.add_scalar("Validation/mAP50", map_results.get('map_50', 0.0), global_step)
                writer.add_scalar("Validation/mAP", map_results.get('map', 0.0), global_step)
                writer.add_scalar("Validation/Sparsity", avg_sparsity, global_step)

                current_map50 = map_results.get('map_50', 0.0)
                if current_map50 > best_map:
                    best_map = current_map50
                    save_path = os.path.join(config['checkpoint_dir'], "joint_best.pth")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': conditional_sr.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map50': best_map,
                        'config': config
                    }, save_path)
                    logger.info(f"Saved best model (mAP50: {best_map:.4f}) to {save_path}")

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                save_path = os.path.join(config['checkpoint_dir'], f"joint_step{global_step}.pth")
                torch.save({
                    'step': global_step,
                    'model_state_dict': conditional_sr.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    writer.close()
    logger.info("Joint finetuning completed.")
    final_save_path = os.path.join(config['checkpoint_dir'], "joint_final.pth")
    torch.save({
        'step': global_step,
        'model_state_dict': conditional_sr.state_dict(),
        'config': config
    }, final_save_path)
    logger.info(f"Saved final model to {final_save_path}")

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

    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    logger = setup_logger(config['log_dir'], "stage3_finetune.log")
    logger.info("Starting Stage 3: Joint Finetuning")
    logger.info(f"Loaded configuration from: {args.config}")

    train_joint(config, logger, args)

if __name__ == "__main__":
    main()
