# stage3_finetune_joint.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import setup_logger, set_logger_level # Import set_logger_level
from utils.losses import calculate_joint_loss
from utils.optimizer_utils import get_optimizer_with_differential_lr
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from torch.utils.tensorboard import SummaryWriter
import json
from torchvision import transforms
from utils.evaluation_utils import run_coco_evaluation
import math
import logging # Import logging for setting level

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Finetuning (ConditionalSR + YOLO)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage3_joint_finetune.yaml)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in steps. Default: 500. Set to 0 to disable step-based eval.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model save interval in steps. Default: 1000. Set to 0 to disable step-based saving.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return parser.parse_args()

def train_joint(config, logger, args):
    # --- 设备选择逻辑 ---
    if args.use_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")
        if num_gpus > 1:
            # Simple strategy: use cuda:0 if multiple GPUs, or let PyTorch handle it if not set.
            # For more specific GPU selection, consider environment variables like CUDA_VISIBLE_DEVICES
            # or more sophisticated device selection logic.
            # For now, we will default to cuda:0 if multiple are present,
            # or let the user manage via CUDA_VISIBLE_DEVICES.
            # If you want to target a specific GPU like 'cuda:1' by default:
            # target_gpu_id = 1
            # if target_gpu_id < num_gpus:
            #     device = torch.device(f'cuda:{target_gpu_id}')
            #     torch.cuda.set_device(device) # Set default CUDA device
            #     logger.info(f"Targeting GPU {target_gpu_id}: {torch.cuda.get_device_name(target_gpu_id)}")
            # else:
            #     device = torch.device('cuda:0')
            #     logger.warning(f"Target GPU {target_gpu_id} not available. Using GPU 0: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda') # Let PyTorch manage via CUDA_VISIBLE_DEVICES or default to cuda:0
            logger.info(f"Using CUDA. Default device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

        elif num_gpus == 1:
            device = torch.device('cuda:0')
            logger.info(f"Using only available CUDA device: GPU 0 ({torch.cuda.get_device_name(0)})")
        else: # Should not happen if torch.cuda.is_available() is true and num_gpus is 0
            logger.warning("CUDA available but no GPUs detected. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        if args.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
        else:
            logger.info("Using CPU for training.")
        device = torch.device('cpu')
    logger.info(f"--- Training on device: {device} ---")

    # --- 配置校验 ---
    logger.info("--- Validating Configuration ---") #
    dataset_config_main = config.get('dataset', {}) #
    image_dir = dataset_config_main.get('image_dir') #
    annotation_file = dataset_config_main.get('annotation_file') #
    scale_factor_cfg = dataset_config_main.get('scale_factor') #

    if not image_dir or not os.path.exists(image_dir): #
        logger.error(f"Training image directory not found: {image_dir}. Exiting.") #
        return
    if not annotation_file or not os.path.exists(annotation_file): #
        logger.error(f"Training annotation file not found: {annotation_file}. Exiting.") #
        return
    if not isinstance(scale_factor_cfg, int) or scale_factor_cfg <= 0: #
         logger.error(f"Invalid scale_factor in dataset config: {scale_factor_cfg}. Must be a positive integer. Exiting.") #
         return

    model_config_main = config.get('model', {}) #
    sr_fast_cfg_yaml = model_config_main.get('sr_fast', {}) #
    sr_quality_cfg_yaml = model_config_main.get('sr_quality', {}) #
    masker_cfg_yaml = model_config_main.get('masker', {}) #
    weights_cfg_yaml = model_config_main.get('weights', {}) #

    if sr_fast_cfg_yaml.get('scale_factor') != scale_factor_cfg: #
        logger.error(f"Scale factor mismatch: dataset ({scale_factor_cfg}) vs sr_fast ({sr_fast_cfg_yaml.get('scale_factor')}). Exiting.") #
        return
    if sr_quality_cfg_yaml.get('scale_factor') != scale_factor_cfg: #
        logger.error(f"Scale factor mismatch: dataset ({scale_factor_cfg}) vs sr_quality ({sr_quality_cfg_yaml.get('scale_factor')}). Exiting.") #
        return

    masker_patch_size_cfg = masker_cfg_yaml.get('output_patch_size') #
    if not isinstance(masker_patch_size_cfg, int) or masker_patch_size_cfg <= 0: #
         logger.error(f"Invalid output_patch_size in masker config: {masker_patch_size_cfg}. Must be a positive integer. Exiting.") #
         return

    if scale_factor_cfg % masker_patch_size_cfg != 0: #
         logger.warning(f"Scale factor ({scale_factor_cfg}) is not divisible by masker output_patch_size ({masker_patch_size_cfg}). This might lead to issues with mask alignment.") #

    masker_threshold_cfg = masker_cfg_yaml.get('threshold') #
    if not isinstance(masker_threshold_cfg, (int, float)) or not (0 <= masker_threshold_cfg <= 1): #
         logger.warning(f"Masker threshold ({masker_threshold_cfg}) is outside the expected range [0, 1].") #

    target_sparsity_ratio_cfg = config.get('train', {}).get('target_sparsity_ratio') #
    if target_sparsity_ratio_cfg is not None and \
       (not isinstance(target_sparsity_ratio_cfg, (int, float)) or not (0 <= target_sparsity_ratio_cfg <= 1)): #
         logger.warning(f"Target sparsity ratio ({target_sparsity_ratio_cfg}) is outside the expected range [0, 1].") #

    required_weights_keys = ['detector', 'sr_fast', 'sr_quality'] #
    for weight_key_check in required_weights_keys: #
        weight_path_check = weights_cfg_yaml.get(weight_key_check) #
        if not weight_path_check or not os.path.exists(weight_path_check): #
            logger.error(f"Required weight file not found for '{weight_key_check}': {weight_path_check}. Exiting.") #
            return

    # YOLO specific params from config for loss calculation
    num_classes_cfg = model_config_main.get('num_classes') #
    yolo_struct_params = model_config_main.get('yolo_params', {}) #
    reg_max_cfg = yolo_struct_params.get('reg_max') #
    strides_cfg = yolo_struct_params.get('strides') #
    yolo_hyp_cfg = config.get('train', {}).get('yolo_hyp', {}) #

    if num_classes_cfg is None: logger.error("model.num_classes not defined in config. Exiting."); return #
    if reg_max_cfg is None: logger.error("model.yolo_params.reg_max not defined in config. Exiting."); return #
    if strides_cfg is None: logger.error("model.yolo_params.strides not defined in config. Exiting."); return #
    if not yolo_hyp_cfg: logger.warning("train.yolo_hyp not defined in config. Default loss gains will be used if not found.") #


    eval_config_main = config.get('evaluation', {}) #
    if eval_config_main.get('val_image_dir') or eval_config_main.get('val_annotation_file'): #
        if not eval_config_main.get('val_image_dir') or not os.path.exists(eval_config_main['val_image_dir']): #
             logger.warning(f"Validation image directory not found: {eval_config_main.get('val_image_dir')}. Evaluation might be skipped or fail.") #
        if not eval_config_main.get('val_annotation_file') or not os.path.exists(eval_config_main['val_annotation_file']): #
             logger.warning(f"Validation annotation file not found: {eval_config_main.get('val_annotation_file')}. Evaluation might be skipped or fail.") #
    logger.info("--- Configuration Validated ---") #

    # --- 数据加载 ---
    train_dataset_config = config['dataset'] #
    train_batch_size = config['train']['batch_size'] #
    num_workers_cfg = config['train'].get('num_workers', 0) #

    try:
        # Assuming LR images are directly in image_dir (not in a subfolder 'LR')
        # as per prepare_detection_data.py which saves LR images to output_dir/LR
        # and updates annotation file_name to "LR/image_name.jpg".
        # DetectionDataset will then join self.image_dir (e.g. "dataset/date_prepared/LR")
        # with os.path.basename(coco_file_name) (e.g. "image_name.jpg")
        # So image_dir for DetectionDataset should be the actual directory containing the images.
        # If your prepare_detection_data.py saves LR images into output_dir/LR,
        # then dataset.image_dir in YAML should point to output_dir (not output_dir/LR)
        # and DetectionDataset will handle the "LR" part from annotation's file_name.
        # OR, if image_dir in YAML is "dataset/date_prepared/LR", then annotation file_name
        # should just be "image_name.jpg".
        # Based on current DetectionDataset, it expects image_dir to be the root of where coco_file_name is relative to.
        # If coco_file_name = "LR/xyz.jpg", image_dir = "output_dir", then path = "output_dir/LR/xyz.jpg"
        # If coco_file_name = "xyz.jpg", image_dir = "output_dir/LR", then path = "output_dir/LR/xyz.jpg"
        # The prepare_detection_data.py saves annotations with file_name "LR/img_name.jpg"
        # and saves LR images to "output_dir/LR/img_name.jpg".
        # So, DetectionDataset's image_dir should be "output_dir" from prepare_detection_data.
        # The YAML has image_dir: "dataset/date_prepared". So DetectionDataset image_dir will be this.
        # And it will look for "dataset/date_prepared/LR/img_name.jpg". This seems correct.

        train_dataset = DetectionDataset(
            image_dir=train_dataset_config['image_dir'], # This is the root. Annotation file_names are relative (e.g. "LR/img.jpg")
            annotation_file=train_dataset_config['annotation_file'], #
            transform=transforms.ToTensor(),
            return_image_id=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size, #
            shuffle=True,
            num_workers=num_workers_cfg, #
            pin_memory=True if device.type == "cuda" else False, #
            collate_fn=DetectionDataset.collate_fn # Use static method
        )
        logger.info(f"Train dataloader initialized with {len(train_dataset)} images. Batch size: {train_batch_size}.") #
    except FileNotFoundError as e:
        logger.error(f"Training data/annotation file not found: {e}. Exiting.") #
        return
    except Exception as e:
        logger.error(f"Error initializing training dataloader: {e}. Exiting.") #
        return

    val_dataloader = None
    if eval_config_main.get('val_image_dir') and eval_config_main.get('val_annotation_file'): #
        try:
            val_dataset = DetectionDataset(
                image_dir=eval_config_main['val_image_dir'], # Similar logic for val image_dir
                annotation_file=eval_config_main['val_annotation_file'], #
                transform=transforms.ToTensor(),
                return_image_id=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['train'].get('val_batch_size', 1),
                shuffle=False,
                num_workers=num_workers_cfg, #
                pin_memory=True if device.type == "cuda" else False, #
                collate_fn=DetectionDataset.collate_fn
            )
            logger.info(f"Validation dataloader initialized with {len(val_dataset)} images.") #
        except FileNotFoundError as e:
            logger.error(f"Validation data/annotation file not found: {e}. Evaluation will be skipped.") #
            val_dataloader = None
        except Exception as e:
            logger.error(f"Error initializing validation dataloader: {e}. Evaluation will be skipped.") #
            val_dataloader = None
    else:
        logger.warning("Validation image directory or annotation file not specified in config. Step-based evaluation will be skipped.") #


    # --- 模型初始化 ---
    sr_fast_model = SRFast(**sr_fast_cfg_yaml).to(device) #
    sr_quality_model = SRQuality(**sr_quality_cfg_yaml).to(device) #
    
    masker_init_args_yaml = {k: masker_cfg_yaml[k] for k in ['in_channels', 'base_channels', 'num_blocks', 'output_channels', 'output_patch_size'] if k in masker_cfg_yaml} #
    masker_model = Masker(**masker_init_args_yaml).to(device) #

    conditional_sr = ConditionalSR(
        sr_fast=sr_fast_model,
        sr_quality=sr_quality_model,
        masker=masker_model,
        detector_weights=weights_cfg_yaml['detector'], #
        sr_fast_weights=weights_cfg_yaml['sr_fast'], #
        sr_quality_weights=weights_cfg_yaml['sr_quality'], #
        masker_weights=weights_cfg_yaml.get('masker', None), #
        device=str(device),
        config=config # Pass the full config to ConditionalSR
    ).to(device)
    logger.info("ConditionalSR model initialized and moved to device.")

    # --- 为YOLO损失计算准备组件 ---
    yolo_model_components_for_loss_dict = None
    if conditional_sr.detector and hasattr(conditional_sr.detector, 'yolo_model_module') and conditional_sr.detector.yolo_model_module:
        yolo_nn_module = conditional_sr.detector.yolo_model_module
        # 这些属性 (stride, nc, reg_max, no) 应该由 ultralytics.nn.tasks.DetectionModel 设置
        # 或者 DetectorWrapper 在加载模型时尝试提取并存储它们
        # For now, we take them from the main config, which is more robust if detector structure varies
        
        yolo_model_components_for_loss_dict = {
            'stride': torch.tensor(strides_cfg, device=device), # From main config
            'nc': num_classes_cfg,                             # From main config
            'reg_max': reg_max_cfg,                            # From main config
            'no': num_classes_cfg + reg_max_cfg * 4,           # Calculated
            'hyp': yolo_hyp_cfg                                # From main config
        }
        logger.info(f"YOLO model components for loss prepared: {yolo_model_components_for_loss_dict}")
    else:
        logger.error("YOLO detector's nn.Module (yolo_model_module) not found or detector not initialized. Cannot compute YOLO loss.")
        # Decide if training can proceed without detection loss or should exit
        if config['train']['loss_weights'].get('detection', 0.0) > 0:
            logger.error("Detection loss weight > 0, but YOLO components are missing. Exiting.")
            return

    # --- 优化器和调度器 ---
    optimizer = get_optimizer_with_differential_lr(conditional_sr, config) #
    
    scheduler = None
    scheduler_config_yaml = config['train'].get('scheduler', {}) #
    if scheduler_config_yaml.get('name', '').lower() == 'cosineannealinglr': #
        scheduler_args_yaml = scheduler_config_yaml.get('args', {}) #
        # T_max is total number of steps if scheduler.step() is called per step
        # T_max is total number of epochs if scheduler.step() is called per epoch
        # The current plan calls scheduler.step() per step.
        total_steps_for_scheduler = config['train']['epochs'] * len(train_dataloader) #
        
        # Allow overriding T_max from config if it's explicitly set for steps
        t_max_from_config = scheduler_args_yaml.get('T_max')
        if t_max_from_config and t_max_from_config > config['train']['epochs']: # Heuristic: if T_max in cfg > epochs, assume it's steps
            final_t_max = t_max_from_config
            logger.info(f"Using T_max for CosineAnnealingLR from config (assumed to be in steps): {final_t_max}")
        else:
            final_t_max = total_steps_for_scheduler
            if t_max_from_config:
                 logger.warning(f"Scheduler T_max in config ({t_max_from_config}) seems small (<= epochs). Using calculated total steps: {final_t_max} for CosineAnnealingLR.")
            else:
                 logger.info(f"Calculated T_max for CosineAnnealingLR (total steps): {final_t_max}")

        eta_min_from_config = scheduler_args_yaml.get('eta_min', 0.0000001) # Default eta_min
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=final_t_max,
            eta_min=eta_min_from_config #
        )
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={final_t_max} steps and eta_min={eta_min_from_config}.") #
    elif scheduler_config_yaml.get('name'): #
        logger.warning(f"Scheduler '{scheduler_config_yaml.get('name')}' not explicitly handled or configured. Scheduler disabled.") #
    else:
        logger.info("Scheduler not configured. Training without a learning rate scheduler.") #


    # --- TensorBoard Writer ---
    tensorboard_log_dir = os.path.join(config['log_dir'], "tensorboard_stage3") #
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    # --- Gumbel 温度设置 ---
    gumbel_config_yaml = config['train'].get('gumbel', {}) #
    initial_tau = gumbel_config_yaml.get('initial_tau', 1.0) #
    final_tau = gumbel_config_yaml.get('final_tau', 0.1) #
    anneal_schedule_gumbel = gumbel_config_yaml.get('anneal_schedule', 'linear').lower() #
    
    anneal_steps_gumbel_cfg = gumbel_config_yaml.get('anneal_steps') #
    if anneal_steps_gumbel_cfg is not None and anneal_steps_gumbel_cfg > 0 : #
        anneal_steps_gumbel = anneal_steps_gumbel_cfg #
        logger.info(f"Gumbel temperature annealing using anneal_steps from config: {anneal_steps_gumbel} steps.") #
    else:
        anneal_epochs_gumbel = gumbel_config_yaml.get('anneal_epochs', config['train']['epochs']) #
        anneal_steps_gumbel = anneal_epochs_gumbel * len(train_dataloader) #
        logger.info(f"Gumbel anneal_steps not set or invalid in config. Calculated from anneal_epochs ({anneal_epochs_gumbel}) * len(dataloader): {anneal_steps_gumbel} steps.") #

    use_gumbel_annealing = anneal_steps_gumbel > 0 and initial_tau != final_tau #
    logger.info(f"Gumbel Annealing: Use={use_gumbel_annealing}, Initial Tau={initial_tau}, Final Tau={final_tau}, Anneal Steps={anneal_steps_gumbel}, Schedule={anneal_schedule_gumbel}") #


    # --- 训练循环 ---
    logger.info("Starting joint finetuning...") #
    best_map50 = 0.0 # For saving best model based on mAP@0.50
    global_step = 0
    num_epochs_total = config['train']['epochs'] #

    for epoch in range(num_epochs_total): #
        conditional_sr.train()
        epoch_total_loss = 0.0
        epoch_detection_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_smooth_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs_total}", total=len(train_dataloader)) #

        for lr_images_batch, targets_batch in progress_bar: #
            if not lr_images_batch.numel() or not targets_batch: # Skip if collate_fn returned empty batch
                logger.warning(f"Skipping empty batch at step {global_step}.")
                continue

            lr_images_batch = lr_images_batch.to(device) #
            # Targets are a list of dicts; elements need to be moved to device if they are tensors
            targets_on_device_batch = []
            for t_dict_item in targets_batch: #
                target_item_dev = {}
                for k_t, v_t in t_dict_item.items(): #
                    if isinstance(v_t, torch.Tensor): #
                        target_item_dev[k_t] = v_t.to(device) #
                    else:
                        target_item_dev[k_t] = v_t #
                targets_on_device_batch.append(target_item_dev) #

            # Gumbel temperature update
            current_tau_gumbel = initial_tau #
            if use_gumbel_annealing: #
                if global_step < anneal_steps_gumbel: #
                    progress_ratio = global_step / anneal_steps_gumbel #
                    if anneal_schedule_gumbel == 'linear': #
                        current_tau_gumbel = initial_tau - (initial_tau - final_tau) * progress_ratio #
                    elif anneal_schedule_gumbel == 'cosine': #
                        current_tau_gumbel = final_tau + 0.5 * (initial_tau - final_tau) * (1 + math.cos(math.pi * progress_ratio)) #
                else:
                    current_tau_gumbel = final_tau #
                current_tau_gumbel = max(current_tau_gumbel, final_tau) # Ensure it doesn't go below final_tau
            
            optimizer.zero_grad()
            
            outputs_model = conditional_sr(lr_images_batch, targets=targets_on_device_batch, temperature=current_tau_gumbel) #
            
            sr_images_output = outputs_model["sr_image"] #
            mask_coarse_output = outputs_model["mask_coarse"] #
            yolo_raw_preds_output = outputs_model["yolo_raw_predictions"] #

            total_loss_val, loss_dict_vals = calculate_joint_loss( #
                sr_images=sr_images_output,
                mask_coarse=mask_coarse_output,
                targets=targets_on_device_batch, 
                yolo_raw_predictions=yolo_raw_preds_output,
                config=config, #
                logger=logger,
                yolo_model_components_for_loss=yolo_model_components_for_loss_dict, # Pass the prepared dict
                precomputed_detection_loss=None 
            )

            if torch.isnan(total_loss_val): #
                logger.error(f"NaN loss detected at step {global_step}. Stopping training.") #
                if writer: writer.close() #
                return 

            total_loss_val.backward()
            # Optional: Gradient Clipping
            # grad_clip_norm = config['train'].get('gradient_clip_norm', None)
            # if grad_clip_norm:
            #     torch.nn.utils.clip_grad_norm_(conditional_sr.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            if scheduler: #
                scheduler.step() # Update LR per step for CosineAnnealingLR

            epoch_total_loss += total_loss_val.item() #
            epoch_detection_loss += loss_dict_vals.get('loss_detection', 0.0) #
            epoch_sparsity_loss += loss_dict_vals.get('loss_sparsity', 0.0) #
            epoch_smooth_loss += loss_dict_vals.get('loss_smooth', 0.0) #
            
            postfix_str_dict = { #
                "total": f"{total_loss_val.item():.4f}",
                "det": f"{loss_dict_vals.get('loss_detection', 0.0):.4f}",
                "spar": f"{loss_dict_vals.get('loss_sparsity', 0.0):.4f}",
                "tau": f"{current_tau_gumbel:.2f}" #
            }
            if optimizer and optimizer.param_groups: #
                 postfix_str_dict["lr_h"] = f"{optimizer.param_groups[0]['lr']:.1e}" #
                 if len(optimizer.param_groups) > 1: #
                      postfix_str_dict["lr_l"] = f"{optimizer.param_groups[1]['lr']:.1e}" #
            progress_bar.set_postfix(postfix_str_dict) #

            log_interval_cfg = config['train'].get('log_interval_steps', 10) #
            if global_step % log_interval_cfg == 0 and writer: #
                writer.add_scalar("Train/TotalLoss_Step", total_loss_val.item(), global_step) #
                writer.add_scalar("Train/DetectionLoss_Step", loss_dict_vals.get('loss_detection', 0.0), global_step) #
                writer.add_scalar("Train/SparsityLoss_Step", loss_dict_vals.get('loss_sparsity', 0.0), global_step) #
                writer.add_scalar("Train/SmoothnessLoss_Step", loss_dict_vals.get('loss_smooth', 0.0), global_step) #
                writer.add_scalar("Train/ActualSparsity_Step", loss_dict_vals.get("actual_sparsity", 0.0), global_step) #
                writer.add_scalar("Train/GumbelTau_Step", current_tau_gumbel, global_step) #
                if optimizer and optimizer.param_groups: #
                    writer.add_scalar("Train/LR_Group_High", optimizer.param_groups[0]['lr'], global_step) #
                    if len(optimizer.param_groups) > 1: #
                        writer.add_scalar("Train/LR_Group_Low", optimizer.param_groups[1]['lr'], global_step) #
                if mask_coarse_output is not None and mask_coarse_output.numel() > 0: #
                    if mask_coarse_output.min() >= 0 and mask_coarse_output.max() <= 1: #
                        epsilon_entropy = 1e-8 #
                        p_entropy = mask_coarse_output.float() #
                        entropy_val = - (p_entropy * torch.log2(p_entropy + epsilon_entropy) + \
                                       (1 - p_entropy) * torch.log2(1 - p_entropy + epsilon_entropy)).mean().item() #
                        writer.add_scalar("Train/Mask_Entropy_Step", entropy_val, global_step) #

            # Model Evaluation & Saving (Step-based)
            if val_dataloader and args.eval_interval > 0 and global_step > 0 and global_step % args.eval_interval == 0: #
                logger.info(f"--- Running evaluation at step {global_step}... ---") #
                eval_output_dir = os.path.join(config['log_dir'], "eval_results_step_based") #
                map_results_eval, avg_sparsity_eval = run_coco_evaluation( #
                    model=conditional_sr,
                    dataloader=val_dataloader,
                    device=device,
                    annotation_file=eval_config_main['val_annotation_file'], #
                    output_dir=eval_output_dir, #
                    step_or_epoch=f"step_{global_step}",
                    logger=logger,
                    use_hard_mask=True 
                )
                if writer: #
                    writer.add_scalar("Validation/mAP50_Step", map_results_eval.get('map_50', 0.0), global_step) #
                    writer.add_scalar("Validation/mAP_Step", map_results_eval.get('map', 0.0), global_step) #
                    writer.add_scalar("Validation/Sparsity_Step", avg_sparsity_eval, global_step) #

                current_map50_eval = map_results_eval.get('map_50', 0.0) #
                if current_map50_eval > best_map50: #
                    best_map50 = current_map50_eval #
                    save_path_best_model = os.path.join(config['checkpoint_dir'], "joint_best_map50.pth") #
                    os.makedirs(config['checkpoint_dir'], exist_ok=True) # Ensure dir exists
                    torch.save({ #
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': conditional_sr.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, #
                        'map50': best_map50, #
                        'config': config #
                    }, save_path_best_model)
                    logger.info(f"Saved best model (mAP50: {best_map50:.4f}) at step {global_step} to {save_path_best_model}") #

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0: #
                save_path_step_model = os.path.join(config['checkpoint_dir'], f"joint_step{global_step}.pth") #
                os.makedirs(config['checkpoint_dir'], exist_ok=True)
                torch.save({ #
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': conditional_sr.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None, #
                    'config': config #
                }, save_path_step_model)
                logger.info(f"Saved checkpoint at step {global_step} to {save_path_step_model}") #
            
            global_step += 1
        # --- End of Epoch ---
        avg_epoch_loss = epoch_total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0 #
        logger.info(f"Epoch {epoch+1}/{num_epochs_total} Summary: AvgTotalLoss={avg_epoch_loss:.4f}") #
        if writer: #
            writer.add_scalar("Train/TotalLoss_Epoch", avg_epoch_loss, epoch + 1) #
            writer.add_scalar("Train/DetectionLoss_Epoch", epoch_detection_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0, epoch + 1) #
            writer.add_scalar("Train/SparsityLoss_Epoch", epoch_sparsity_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0, epoch + 1) #
            writer.add_scalar("Train/SmoothnessLoss_Epoch", epoch_smooth_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0, epoch + 1) #


    # --- Training Finished ---
    if writer: writer.close() #
    logger.info("Joint finetuning completed.") #
    final_model_save_path = os.path.join(config['checkpoint_dir'], "joint_final.pth") #
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    torch.save({ #
        'step': global_step,
        'epoch': num_epochs_total,
        'model_state_dict': conditional_sr.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None, #
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, #
        'map50': best_map50, # Save last best mAP50
        'config': config #
    }, final_model_save_path)
    logger.info(f"Saved final model to {final_model_save_path}") #


def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f: #
            config = yaml.safe_load(f) #
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}") #
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}") #
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}") #
        exit(1)

    log_dir_config = config.get('log_dir', './temp_logs/stage3_joint') #
    checkpoint_dir_config = config.get('checkpoint_dir', './temp_checkpoints/stage3_joint') #

    os.makedirs(log_dir_config, exist_ok=True) #
    os.makedirs(checkpoint_dir_config, exist_ok=True) #
    
    logger = setup_logger(log_dir_config, "stage3_finetune_joint.log") #
    
    # Set logging level based on command line argument
    numeric_log_level = getattr(logging, args.log_level.upper(), None) #
    if not isinstance(numeric_log_level, int): #
        raise ValueError(f'Invalid log level: {args.log_level}') #
    set_logger_level(logger, numeric_log_level) #
    logger.info(f"Logging level set to: {args.log_level}") #
    
    logger.info("--- Starting Stage 3: Joint Finetuning ---") #
    logger.info(f"Loaded configuration from: {args.config}") #
    logger.info(f"Command line arguments: {args}") #
    if args.eval_interval == 0: logger.info("Step-based evaluation is disabled via --eval_interval 0.") #
    if args.save_interval == 0: logger.info("Step-based model saving is disabled via --save_interval 0.") #


    train_joint(config, logger, args)

if __name__ == "__main__":
    main()