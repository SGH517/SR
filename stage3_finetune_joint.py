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
import json  # For saving eval results
from pycocotools.coco import COCO  # For eval
from pycocotools.cocoeval import COCOeval  # For eval
from torchvision import transforms  # For eval dataset transform
from utils.evaluation_utils import run_coco_evaluation  # Import the utility function

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Finetuning (ConditionalSR + YOLO)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage3_joint_finetune.yaml)")
    # Make eval/save intervals step-based as requested in previous assessment
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in steps")  # Changed default
    parser.add_argument("--save_interval", type=int, default=1000, help="Model save interval in steps")  # Changed default
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def train_joint(config, logger, args):
    # 设置device参数，优先使用命令行参数
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available, falling back to CPU")

    # --- 数据集 ---
    dataset_config = config['dataset']
    train_dataset = DetectionDataset(
        image_dir=os.path.join(dataset_config['image_dir'], "LR"),  # Point to LR subdir
        annotation_file=dataset_config['annotation_file'],
        transform=transforms.ToTensor(),  # Use ToTensor for training
        return_image_id=True  # Ensure dataset returns image_id
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        collate_fn=getattr(train_dataset, 'collate_fn', None)  # Use dataset's collate_fn
    )

    # --- 验证数据集 ---
    eval_config = config.get('evaluation', {})
    val_dataloader = None
    if eval_config.get('val_image_dir') and eval_config.get('val_annotation_file'):
        try:
            val_dataset = DetectionDataset(
                image_dir=os.path.join(eval_config['val_image_dir'], "LR"),  # Point to LR subdir
                annotation_file=eval_config['val_annotation_file'],
                transform=transforms.ToTensor(),  # Use ToTensor for validation
                return_image_id=True  # Ensure dataset returns image_id
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['train'].get('val_batch_size', 1),  # Use smaller batch size for eval
                shuffle=False,
                num_workers=config['train']['num_workers'],
                pin_memory=True,
                collate_fn=getattr(val_dataset, 'collate_fn', None)  # Use dataset's collate_fn
            )
            logger.info("Validation dataloader initialized.")
        except FileNotFoundError as e:
            logger.error(f"Validation data/annotation file not found: {e}. Evaluation will be skipped.")
        except Exception as e:
            logger.error(f"Error initializing validation dataloader: {e}. Evaluation will be skipped.")
    else:
        logger.warning("Validation image directory or annotation file not specified in config. Evaluation will be skipped.")


    # --- 模型初始化 ---
    # Pass the whole config to ConditionalSR
    conditional_sr = ConditionalSR(
        sr_fast=SRFast(**config['model']['sr_fast']),
        sr_quality=SRQuality(**config['model']['sr_quality']),
        masker=Masker(**config['model']['masker']),
        detector_weights=config['model']['weights']['detector'],
        sr_fast_weights=config['model']['weights']['sr_fast'],
        sr_quality_weights=config['model']['weights']['sr_quality'],
        masker_weights=config['model']['weights']['masker'],
        device=device,
        config=config  # Pass the full config here
    ).to(device)

    # --- 优化器和调度器 ---
    optimizer = get_optimizer_with_differential_lr(conditional_sr, config)
    # Ensure scheduler config exists and has required keys
    scheduler = None
    if 'scheduler' in config['train'] and config['train']['scheduler'].get('name', '').lower() == 'cosineannealinglr':
        scheduler_args = config['train']['scheduler'].get('args', {})
        if 'T_max' in scheduler_args and 'eta_min' in scheduler_args:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_args['T_max'] * len(train_dataloader),  # T_max in steps
                eta_min=scheduler_args['eta_min']
            )
            logger.info(f"Using CosineAnnealingLR scheduler with T_max={scheduler_args['T_max'] * len(train_dataloader)} steps.")
        else:
            logger.warning("Scheduler config missing T_max or eta_min. Scheduler disabled.")
    else:
        logger.warning("Scheduler not configured or not CosineAnnealingLR. Scheduler disabled.")


    # --- TensorBoard 初始化 ---
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], "tensorboard"))

    logger.info("Starting joint finetuning...")
    best_map = 0.0
    global_step = 0

    # --- Gumbel Temperature Annealing Setup ---
    gumbel_config = config['train'].get('gumbel', {})
    initial_tau = gumbel_config.get('initial_tau', 1.0)
    final_tau = gumbel_config.get('final_tau', 0.1)
    anneal_steps = gumbel_config.get('anneal_epochs', config['train']['epochs'] // 2) * len(train_dataloader)  # Convert epochs to steps
    use_annealing = anneal_steps > 0 and initial_tau != final_tau
    logger.info(f"Gumbel Annealing: Use={use_annealing}, Initial Tau={initial_tau}, Final Tau={final_tau}, Anneal Steps={anneal_steps}")


    for epoch in range(config['train']['epochs']):
        conditional_sr.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")

        for lr_images, targets in progress_bar:
            lr_images = lr_images.to(device)
            # Move targets to device (assuming targets is a list of dicts with tensors)
            targets_on_device = []
            if targets:
                for t in targets:
                    target_device = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            target_device[k] = v.to(device)
                        else:
                            target_device[k] = v  # Keep non-tensors as they are (like image_id)
                    targets_on_device.append(target_device)
            else:
                targets_on_device = None  # Handle case where targets might be None


            # --- Calculate current Gumbel temperature ---
            current_tau = initial_tau
            if use_annealing:
                # Linear annealing example
                anneal_progress = min(1.0, global_step / anneal_steps)
                current_tau = initial_tau - (initial_tau - final_tau) * anneal_progress
                # Ensure tau doesn't go below final_tau
                current_tau = max(final_tau, current_tau)


            optimizer.zero_grad()
            # Pass targets and current temperature to the model
            outputs = conditional_sr(lr_images, targets=targets_on_device, temperature=current_tau)

            # Get necessary outputs for loss calculation
            sr_images = outputs["sr_image"]
            mask_coarse = outputs["mask_coarse"]
            # Get precomputed detection loss from model output
            precomputed_detection_loss = outputs["detection_loss"]

            # Calculate joint loss, passing logger and precomputed loss
            total_loss, loss_dict = calculate_joint_loss(
                sr_images=sr_images,
                mask_coarse=mask_coarse,
                targets=targets_on_device,  # Pass targets again if needed by loss func
                detector=conditional_sr.detector,  # Pass detector if needed
                config=config,
                logger=logger, # Pass the logger instance
                precomputed_detection_loss=precomputed_detection_loss # Pass precomputed loss
            )

            if torch.isnan(total_loss):
                logger.error(f"NaN loss detected at step {global_step}. Stopping training.")
                writer.close()
                return  # Stop training

            total_loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(conditional_sr.parameters(), max_norm=1.0)
            optimizer.step()

            # Update learning rate scheduler (step-based)
            if scheduler:
                scheduler.step()


            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item(), tau=current_tau)  # Show temp in progress bar

            # --- Logging ---
            if global_step % 10 == 0:  # Log every 10 steps
                writer.add_scalar("Train/TotalLoss", total_loss.item(), global_step)
                writer.add_scalar("Train/GumbelTau", current_tau, global_step)
                # Log individual losses from loss_dict
                for loss_name, loss_val in loss_dict.items():
                    if loss_name != "total_loss":  # Avoid logging total loss twice
                        writer.add_scalar(f"Train/Loss_{loss_name}", loss_val, global_step)
                # Log learning rates for different param groups
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Train/LR_Group_{i}", param_group['lr'], global_step)


            # --- Evaluation ---
            if val_dataloader and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                logger.info(f"Step {global_step}: Running evaluation...")
                map_results, avg_sparsity = run_coco_evaluation(
                    model=conditional_sr,
                    dataloader=val_dataloader,
                    device=device,
                    annotation_file=config['evaluation']['val_annotation_file'],
                    output_dir=config['log_dir'],  # Or a dedicated eval output dir
                    step_or_epoch=global_step,
                    logger=logger,
                    use_hard_mask=True  # Or False depending on desired eval mode
                )

                # Log validation metrics
                writer.add_scalar("Validation/mAP50", map_results.get('map_50', 0.0), global_step)
                writer.add_scalar("Validation/mAP", map_results.get('map', 0.0), global_step)
                writer.add_scalar("Validation/Sparsity", avg_sparsity, global_step)

                # Save best model based on mAP50
                current_map50 = map_results.get('map_50', 0.0)
                if current_map50 > best_map:
                    best_map = current_map50
                    save_path = os.path.join(config['checkpoint_dir'], "joint_best.pth")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': conditional_sr.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map50': best_map,
                    }, save_path)
                    logger.info(f"Saved best model (mAP50: {best_map:.4f}) to {save_path}")


            # --- Save Checkpoint ---
            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_path = os.path.join(config['checkpoint_dir'], f"joint_step{global_step}.pth")
                torch.save({
                    'step': global_step,
                    'model_state_dict': conditional_sr.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Optionally save scheduler state: 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")

            global_step += 1  # Increment global step counter

        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        # Epoch-based scheduler step (if not using step-based scheduler)
        # if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        #     scheduler.step()


    writer.close()
    logger.info("Joint finetuning completed.")
    # Save final model
    final_save_path = os.path.join(config['checkpoint_dir'], "joint_final.pth")
    torch.save({'step': global_step, 'model_state_dict': conditional_sr.state_dict()}, final_save_path)
    logger.info(f"Saved final model to {final_save_path}")


def main():
    args = parse_args()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # Create log and checkpoint directories if they don't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    logger = setup_logger(config['log_dir'], "stage3_finetune.log")
    logger.info("Starting Stage 3: Joint Finetuning")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Using device: {config['train']['device']}")

    train_joint(config, logger, args)

if __name__ == "__main__":
    main()
