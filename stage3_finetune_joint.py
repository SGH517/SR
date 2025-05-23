# stage3_finetune_joint.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json # 确保导入 json
import math
import logging # 确保导入 logging
from torchvision import transforms # 确保导入 transforms

# 从 utils 导入新模块和现有模块
from utils.logger import setup_logger, set_logger_level
from utils.losses import calculate_joint_loss
from utils.optimizer_utils import get_optimizer_with_differential_lr
from utils.evaluation_utils import run_coco_evaluation
# 导入新的工具函数
from utils.common_utils import get_device
from utils.config_utils import validate_config
from utils.model_utils import load_full_checkpoint # 用于恢复训练

# 从 data 和 models 导入
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Finetuning (ConditionalSR + YOLO)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (stage3_joint_finetune.yaml)")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluation interval in steps. Default: 500. Set to 0 to disable step-based eval.")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Model save interval in steps. Default: 1000. Set to 0 to disable step-based saving.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to a checkpoint to resume training from (optional).")
    return parser.parse_args()

def train_joint(config: Dict, logger: logging.Logger, args: argparse.Namespace):
    # --- 设备选择 ---
    # 使用新的工具函数获取设备
    device = get_device(args.use_gpu, logger)
    logger.info(f"--- 训练将在设备: {device} 上进行 ---")

    # --- 配置校验 ---
    # 将 args 合并到 config 中，以便 validate_config 可以访问它们 (例如 args.enable_eval)
    config['args'] = vars(args) # vars(args) 将 Namespace 转为字典
    if not validate_config(config, "stage3_joint", logger):
        logger.error("配置校验失败。正在退出。")
        return
    # 从已校验的 config 中安全地获取参数
    dataset_config = config['dataset']
    model_config = config['model']
    train_config = config['train']
    eval_config = config.get('evaluation', {}) # evaluation 可能不存在

    # --- 数据加载 ---
    train_batch_size = train_config['batch_size']
    num_workers_cfg = train_config.get('num_workers', 0)

    try:
        # 假设 DetectionDataset 的 image_dir 参数期望的是包含 "LR" 子目录的根目录
        # 例如，如果 LR 图像在 "dataset/date_prepared/LR/"
        # 并且标注文件中的 "file_name" 是 "LR/some_image.jpg"
        # 那么 dataset_config['image_dir'] 应该是 "dataset/date_prepared"
        train_dataset_image_dir = dataset_config['image_dir'] # 这是图像的根目录
        train_dataset = DetectionDataset(
            image_dir=train_dataset_image_dir,
            annotation_file=dataset_config['annotation_file'],
            transform=transforms.ToTensor(), # 基本转换
            return_image_id=True # COCO 评估需要 image_id
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers_cfg,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=DetectionDataset.collate_fn # 使用静态方法处理可能的None样本
        )
        logger.info(f"训练数据加载器已初始化，包含 {len(train_dataset)} 张图像。批次大小: {train_batch_size}。")
    except FileNotFoundError as e:
        logger.error(f"训练数据或标注文件未找到: {e}。正在退出。")
        return
    except Exception as e:
        logger.error(f"初始化训练数据加载器时发生错误: {e}。正在退出。", exc_info=True)
        return

    val_dataloader = None
    if eval_config.get('val_image_dir') and eval_config.get('val_annotation_file'):
        try:
            val_dataset_image_dir = eval_config['val_image_dir']
            val_dataset = DetectionDataset(
                image_dir=val_dataset_image_dir,
                annotation_file=eval_config['val_annotation_file'],
                transform=transforms.ToTensor(),
                return_image_id=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=train_config.get('val_batch_size', 1), # 验证时通常批次为1
                shuffle=False,
                num_workers=num_workers_cfg,
                pin_memory=True if device.type == "cuda" else False,
                collate_fn=DetectionDataset.collate_fn
            )
            logger.info(f"验证数据加载器已初始化，包含 {len(val_dataset)} 张图像。")
        except FileNotFoundError as e:
            logger.error(f"验证数据或标注文件未找到: {e}。将跳过基于步骤的评估。")
            val_dataloader = None
        except Exception as e:
            logger.error(f"初始化验证数据加载器时发生错误: {e}。将跳过基于步骤的评估。", exc_info=True)
            val_dataloader = None
    else:
        logger.warning("配置文件中未指定验证图像目录或标注文件。将跳过基于步骤的评估。")

    # --- 模型初始化 ---
    # SRFast, SRQuality, Masker 的参数从 model_config 中获取
    sr_fast_params = model_config.get('sr_fast', {})
    sr_quality_params = model_config.get('sr_quality', {})
    masker_params = model_config.get('masker', {})

    sr_fast_model = SRFast(**sr_fast_params).to(device)
    sr_quality_model = SRQuality(**sr_quality_params).to(device)
    masker_model = Masker(**masker_params).to(device)

    # ConditionalSR 初始化
    # 权重路径从 model_config['weights'] 获取
    weights_paths = model_config.get('weights', {})
    conditional_sr = ConditionalSR(
        sr_fast=sr_fast_model,
        sr_quality=sr_quality_model,
        masker=masker_model,
        detector_weights=weights_paths.get('detector'), # 可能为 None
        sr_fast_weights=weights_paths.get('sr_fast'),   # 可能为 None
        sr_quality_weights=weights_paths.get('sr_quality'),# 可能为 None
        masker_weights=weights_paths.get('masker'),     # 可能为 None
        device=str(device),
        config=config # 将完整的配置字典传递给 ConditionalSR
    ).to(device)
    logger.info("ConditionalSR 模型已初始化并移至设备。")

    # --- 为YOLO损失计算准备组件 ---
    yolo_model_components_for_loss_dict = None
    # ConditionalSR.detector 是 DetectorWrapper 实例
    # DetectorWrapper.yolo_model_module 是底层的 YOLO nn.Module
    if conditional_sr.detector and conditional_sr.detector.yolo_model_module:
        # 从主配置中获取 YOLO 特定参数
        num_classes_cfg = model_config.get('num_classes')
        yolo_struct_params = model_config.get('yolo_params', {})
        reg_max_cfg = yolo_struct_params.get('reg_max')
        strides_cfg = yolo_struct_params.get('strides') # 这是一个列表
        yolo_hyp_cfg = train_config.get('yolo_hyp', {})

        if num_classes_cfg is None or reg_max_cfg is None or strides_cfg is None:
            logger.error("模型配置中缺少 num_classes, yolo_params.reg_max 或 yolo_params.strides。无法计算YOLO损失。")
            if train_config.get('loss_weights', {}).get('detection', 0.0) > 0:
                logger.error("检测损失权重 > 0，但YOLO组件参数缺失。正在退出。")
                return
        else:
            yolo_model_components_for_loss_dict = {
                'stride': torch.tensor(strides_cfg, device=device, dtype=torch.float), # 确保是浮点张量
                'nc': num_classes_cfg,
                'reg_max': reg_max_cfg,
                'no': num_classes_cfg + reg_max_cfg * 4, # 根据定义计算
                'hyp': yolo_hyp_cfg
            }
            logger.info(f"为损失计算准备的YOLO模型组件: {yolo_model_components_for_loss_dict}")
    else:
        logger.warning("ConditionalSR.detector 或其 yolo_model_module 未初始化。")
        if train_config.get('loss_weights', {}).get('detection', 0.0) > 0:
            logger.error("检测损失权重 > 0，但YOLO检测器组件缺失。正在退出。")
            return


    # --- 优化器和调度器 ---
    optimizer = get_optimizer_with_differential_lr(conditional_sr, config)
    logger.info(f"优化器已创建: {optimizer.__class__.__name__}")

    scheduler = None
    scheduler_config_yaml = train_config.get('scheduler', {})
    if scheduler_config_yaml.get('name', '').lower() == 'cosineannealinglr':
        scheduler_args_yaml = scheduler_config_yaml.get('args', {})
        total_steps_for_scheduler = train_config['epochs'] * len(train_dataloader)

        t_max_from_config = scheduler_args_yaml.get('T_max')
        final_t_max = t_max_from_config if t_max_from_config else total_steps_for_scheduler
        eta_min_from_config = scheduler_args_yaml.get('eta_min', 1e-7) # 使用更小的默认 eta_min

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=final_t_max,
            eta_min=eta_min_from_config
        )
        logger.info(f"使用 CosineAnnealingLR 调度器，T_max={final_t_max} 步, eta_min={eta_min_from_config}。")
    elif scheduler_config_yaml.get('name'):
        logger.warning(f"调度器 '{scheduler_config_yaml.get('name')}' 未显式处理或配置。调度器已禁用。")
    else:
        logger.info("未配置调度器。将在没有学习率调度器的情况下进行训练。")


    # --- TensorBoard Writer ---
    tensorboard_log_dir = os.path.join(config.get('log_dir', './temp_logs/stage3_joint'), "tensorboard_stage3")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard 日志将保存到: {tensorboard_log_dir}")

    # --- Gumbel 温度设置 ---
    gumbel_config_yaml = train_config.get('gumbel', {})
    initial_tau = gumbel_config_yaml.get('initial_tau', 1.0)
    final_tau = gumbel_config_yaml.get('final_tau', 0.1)
    anneal_schedule_gumbel = gumbel_config_yaml.get('anneal_schedule', 'linear').lower()

    anneal_steps_gumbel_cfg = gumbel_config_yaml.get('anneal_steps')
    if anneal_steps_gumbel_cfg is not None and anneal_steps_gumbel_cfg > 0:
        anneal_steps_gumbel = anneal_steps_gumbel_cfg
    else:
        anneal_epochs_gumbel = gumbel_config_yaml.get('anneal_epochs', train_config['epochs'])
        anneal_steps_gumbel = anneal_epochs_gumbel * len(train_dataloader)
    use_gumbel_annealing = anneal_steps_gumbel > 0 and initial_tau != final_tau
    logger.info(f"Gumbel 退火: 使用={use_gumbel_annealing}, 初始 Tau={initial_tau}, "
                f"最终 Tau={final_tau}, 退火步数={anneal_steps_gumbel}, 策略={anneal_schedule_gumbel}")

    # --- 断点续训逻辑 ---
    start_epoch = 0
    global_step = 0
    best_map50 = 0.0

    if args.resume_path:
        logger.info(f"尝试从检查点恢复训练: {args.resume_path}")
        checkpoint = load_full_checkpoint(args.resume_path, device, logger)
        if checkpoint:
            try:
                # 加载模型权重 (注意：ConditionalSR初始化时已经加载了各子模块的预训练权重)
                # 如果 resume_path 是针对整个 ConditionalSR 联合训练的检查点，则应该加载它
                # 假设这里的 resume 是接着之前的联合训练
                if 'model_state_dict' in checkpoint:
                     # 使用我们创建的工具函数，它能处理 'module.' 前缀
                    load_model_weights(conditional_sr, args.resume_path, device, "ConditionalSR (resumed)", logger, strict=False)
                    logger.info(f"已从检查点 {args.resume_path} 加载 ConditionalSR 模型权重。")
                else:
                    logger.warning(f"检查点 {args.resume_path} 中未找到 'model_state_dict'。模型权重可能未恢复。")


                if 'optimizer_state_dict' in checkpoint and optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("已加载优化器状态。")
                if 'scheduler_state_dict' in checkpoint and scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("已加载调度器状态。")

                start_epoch = checkpoint.get('epoch', 0) + 1 # 从下一轮开始
                global_step = checkpoint.get('step', 0)
                best_map50 = checkpoint.get('map50', 0.0)
                # 如果配置也保存在检查点中，可以考虑是否加载或比较
                # config_ckpt = checkpoint.get('config')
                # if config_ckpt:
                #     logger.info("检查点中包含配置信息。") # 可以选择性地使用或与当前配置比较

                logger.info(f"成功从检查点恢复。将从 Epoch {start_epoch + 1}, Global Step {global_step} 开始训练。记录的最佳 mAP50: {best_map50:.4f}")
            except Exception as e:
                logger.error(f"从检查点 {args.resume_path} 恢复时发生错误: {e}。将从头开始训练。", exc_info=True)
                start_epoch = 0
                global_step = 0
                best_map50 = 0.0
        else:
            logger.warning(f"无法加载检查点 {args.resume_path}。将从头开始训练。")


    # --- 训练循环 ---
    logger.info(f"--- 开始联合微调 (将从 Epoch {start_epoch + 1} 开始) ---")
    num_epochs_total = train_config['epochs']

    for epoch in range(start_epoch, num_epochs_total):
        conditional_sr.train() # 设置模型为训练模式
        epoch_total_loss = 0.0
        epoch_detection_loss = 0.0
        epoch_sparsity_loss = 0.0
        epoch_smooth_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs_total}", total=len(train_dataloader))

        for lr_images_batch, targets_batch_list_dict in progress_bar:
            if not lr_images_batch.numel() or not targets_batch_list_dict:
                logger.warning(f"在步骤 {global_step} 跳过空批次。")
                continue

            lr_images_batch = lr_images_batch.to(device)
            # targets_batch_list_dict 是一个字典列表，每个字典包含 'boxes', 'labels', 'image_id' (均为张量)
            # 需要确保这些张量也在正确的设备上
            targets_on_device_batch = []
            for t_dict_item in targets_batch_list_dict:
                target_item_dev = {}
                for k_t, v_t in t_dict_item.items():
                    if isinstance(v_t, torch.Tensor):
                        target_item_dev[k_t] = v_t.to(device)
                    else:
                        target_item_dev[k_t] = v_t # 例如 image_id 可能是非张量
                targets_on_device_batch.append(target_item_dev)

            # Gumbel 温度更新
            current_tau_gumbel = initial_tau
            if use_gumbel_annealing:
                progress_ratio = min(1.0, global_step / anneal_steps_gumbel) if anneal_steps_gumbel > 0 else 1.0
                if anneal_schedule_gumbel == 'linear':
                    current_tau_gumbel = initial_tau - (initial_tau - final_tau) * progress_ratio
                elif anneal_schedule_gumbel == 'cosine':
                    current_tau_gumbel = final_tau + 0.5 * (initial_tau - final_tau) * (1 + math.cos(math.pi * progress_ratio))
                current_tau_gumbel = max(current_tau_gumbel, final_tau) # 确保不低于 final_tau

            optimizer.zero_grad()

            # ConditionalSR.forward 返回一个字典
            outputs_model = conditional_sr(
                lr_images_batch,
                targets=targets_on_device_batch, # COCO 格式的目标列表
                temperature=current_tau_gumbel,
                # hard_mask_inference 在训练时通常为 False (由 Gumbel-Softmax 控制)
            )

            sr_images_output = outputs_model["sr_image"]
            mask_coarse_output = outputs_model["mask_coarse"]
            yolo_raw_preds_output = outputs_model["yolo_raw_predictions"]
            # outputs_model["detection_loss_from_wrapper"] 应该为 None

            total_loss_val, loss_dict_vals = calculate_joint_loss(
                sr_images=sr_images_output,
                mask_coarse=mask_coarse_output,
                targets=targets_on_device_batch, # 传递 COCO 格式的标注
                yolo_raw_predictions=yolo_raw_preds_output, # YOLO 模型的原始输出
                config=config,
                logger=logger,
                yolo_model_components_for_loss=yolo_model_components_for_loss_dict,
                # precomputed_detection_loss 参数已不再需要，因为损失总是在内部计算
            )

            if torch.isnan(total_loss_val):
                logger.error(f"在步骤 {global_step} 检测到 NaN 损失。停止训练。")
                if writer: writer.close()
                return # 停止训练

            total_loss_val.backward()
            # 可选: 梯度裁剪 (从配置中读取)
            grad_clip_norm = train_config.get('gradient_clip_norm', None)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(conditional_sr.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            if scheduler:
                scheduler.step() # 每一步都更新学习率 (对于 CosineAnnealingLR)

            epoch_total_loss += total_loss_val.item()
            epoch_detection_loss += loss_dict_vals.get('loss_detection', 0.0)
            epoch_sparsity_loss += loss_dict_vals.get('loss_sparsity', 0.0)
            epoch_smooth_loss += loss_dict_vals.get('loss_smooth', 0.0)

            postfix_str_dict = {
                "total": f"{total_loss_val.item():.4f}",
                "det": f"{loss_dict_vals.get('loss_detection', 0.0):.4f}",
                "spar": f"{loss_dict_vals.get('loss_sparsity', 0.0):.4f}",
                "tau": f"{current_tau_gumbel:.2f}"
            }
            if optimizer and optimizer.param_groups:
                 postfix_str_dict["lr_h"] = f"{optimizer.param_groups[0]['lr']:.1e}"
                 if len(optimizer.param_groups) > 1:
                      postfix_str_dict["lr_l"] = f"{optimizer.param_groups[1]['lr']:.1e}"
            progress_bar.set_postfix(postfix_str_dict)

            log_interval_cfg = train_config.get('log_interval_steps', 10)
            if global_step % log_interval_cfg == 0 and writer:
                writer.add_scalar("Train/TotalLoss_Step", total_loss_val.item(), global_step)
                writer.add_scalar("Train/DetectionLoss_Step", loss_dict_vals.get('loss_detection', 0.0), global_step)
                writer.add_scalar("Train/SparsityLoss_Step", loss_dict_vals.get('loss_sparsity', 0.0), global_step)
                writer.add_scalar("Train/SmoothnessLoss_Step", loss_dict_vals.get('loss_smooth', 0.0), global_step)
                writer.add_scalar("Train/ActualSparsity_Step", loss_dict_vals.get("actual_sparsity", 0.0), global_step)
                writer.add_scalar("Train/GumbelTau_Step", current_tau_gumbel, global_step)
                if optimizer and optimizer.param_groups:
                    writer.add_scalar("Train/LR_Group_High", optimizer.param_groups[0]['lr'], global_step)
                    if len(optimizer.param_groups) > 1:
                        writer.add_scalar("Train/LR_Group_Low", optimizer.param_groups[1]['lr'], global_step)
                if mask_coarse_output is not None and mask_coarse_output.numel() > 0:
                    if mask_coarse_output.min() >= 0 and mask_coarse_output.max() <= 1:
                        epsilon_entropy = 1e-8
                        p_entropy = mask_coarse_output.float()
                        entropy_val = - (p_entropy * torch.log2(p_entropy + epsilon_entropy) + \
                                       (1 - p_entropy) * torch.log2(1 - p_entropy + epsilon_entropy)).mean().item()
                        writer.add_scalar("Train/Mask_Entropy_Step", entropy_val, global_step)

            # 模型评估和保存 (基于步骤)
            if val_dataloader and args.eval_interval > 0 and global_step > 0 and global_step % args.eval_interval == 0:
                logger.info(f"--- 在步骤 {global_step} 运行评估... ---")
                eval_output_dir = os.path.join(config.get('log_dir', './temp_logs/stage3_joint'), "eval_results_step_based")
                map_results_eval, avg_sparsity_eval = run_coco_evaluation(
                    model=conditional_sr,
                    dataloader=val_dataloader,
                    device=device,
                    annotation_file=eval_config['val_annotation_file'], # 确保这个路径正确
                    output_dir=eval_output_dir,
                    step_or_epoch=f"step_{global_step}",
                    logger=logger,
                    use_hard_mask=True # 评估时通常使用硬掩码
                )
                if writer:
                    writer.add_scalar("Validation/mAP50_Step", map_results_eval.get('map_50', 0.0), global_step)
                    writer.add_scalar("Validation/mAP_Step", map_results_eval.get('map', 0.0), global_step)
                    writer.add_scalar("Validation/Sparsity_Step", avg_sparsity_eval, global_step)

                current_map50_eval = map_results_eval.get('map_50', 0.0)
                if current_map50_eval > best_map50:
                    best_map50 = current_map50_eval
                    save_path_best_model = os.path.join(config.get('checkpoint_dir', './temp_checkpoints/stage3_joint'), "joint_best_map50.pth")
                    os.makedirs(os.path.dirname(save_path_best_model), exist_ok=True)
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': conditional_sr.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'map50': best_map50,
                        'config': config # 保存配置以供后续使用
                    }, save_path_best_model)
                    logger.info(f"在步骤 {global_step} 保存了最佳模型 (mAP50: {best_map50:.4f}) 到 {save_path_best_model}")

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                save_path_step_model = os.path.join(config.get('checkpoint_dir', './temp_checkpoints/stage3_joint'), f"joint_step{global_step}.pth")
                os.makedirs(os.path.dirname(save_path_step_model), exist_ok=True)
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': conditional_sr.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config
                }, save_path_step_model)
                logger.info(f"在步骤 {global_step} 保存了检查点到 {save_path_step_model}")

            global_step += 1
        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs_total} 总结: 平均总损失={avg_epoch_loss:.4f}")
        if writer:
            writer.add_scalar("Train/TotalLoss_Epoch", avg_epoch_loss, epoch + 1)
            writer.add_scalar("Train/DetectionLoss_Epoch", (epoch_detection_loss / len(train_dataloader)) if len(train_dataloader) > 0 else 0.0, epoch + 1)
            writer.add_scalar("Train/SparsityLoss_Epoch", (epoch_sparsity_loss / len(train_dataloader)) if len(train_dataloader) > 0 else 0.0, epoch + 1)
            writer.add_scalar("Train/SmoothnessLoss_Epoch", (epoch_smooth_loss / len(train_dataloader)) if len(train_dataloader) > 0 else 0.0, epoch + 1)


    # --- 训练完成 ---
    if writer:
        writer.close()
    logger.info("联合微调完成。")
    final_model_save_path = os.path.join(config.get('checkpoint_dir', './temp_checkpoints/stage3_joint'), "joint_final.pth")
    os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
    torch.save({
        'step': global_step,
        'epoch': num_epochs_total -1, # 保存完成的 epoch 数
        'model_state_dict': conditional_sr.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'map50': best_map50, # 保存最后记录的最佳 mAP50
        'config': config
    }, final_model_save_path)
    logger.info(f"最终模型已保存到 {final_model_save_path}")


def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件在 {args.config} 未找到")
        exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 配置文件失败: {e}")
        exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件时发生错误: {e}")
        exit(1)

    # 确保日志和检查点目录存在 (使用配置中的路径)
    log_dir_config = config.get('log_dir', './temp_logs/stage3_joint')
    checkpoint_dir_config = config.get('checkpoint_dir', './temp_checkpoints/stage3_joint')
    os.makedirs(log_dir_config, exist_ok=True)
    os.makedirs(checkpoint_dir_config, exist_ok=True)

    logger = setup_logger(log_dir_config, "stage3_finetune_joint.log")

    # 根据命令行参数设置日志级别
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logger.critical(f'无效的日志级别: {args.log_level}') # 使用 logger 记录
        raise ValueError(f'无效的日志级别: {args.log_level}')
    set_logger_level(logger, numeric_log_level)
    logger.info(f"日志级别已设置为: {args.log_level}")

    logger.info("--- 开始阶段 3: 联合微调 ---")
    logger.info(f"已从以下路径加载配置: {args.config}")
    logger.info(f"命令行参数: {args}")
    if args.eval_interval == 0: logger.info("通过 --eval_interval 0 禁用了基于步骤的评估。")
    if args.save_interval == 0: logger.info("通过 --save_interval 0 禁用了基于步骤的模型保存。")

    train_joint(config, logger, args)

if __name__ == "__main__":
    main()