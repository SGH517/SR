# stage3_finetune_joint.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import setup_logger
from utils.losses import calculate_joint_loss # 确保 calculate_joint_loss 被正确导入
from utils.optimizer_utils import get_optimizer_with_differential_lr
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from torch.utils.tensorboard import SummaryWriter
import json
# from pycocotools.coco import COCO # 这两个在 train_joint 中不需要，但在 main 中可能需要
# from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from utils.evaluation_utils import run_coco_evaluation
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Finetuning (ConditionalSR + YOLO)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage3_joint_finetune.yaml)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in steps") # 配置文件中似乎没有，但命令行参数中可以有
    parser.add_argument("--save_interval", type=int, default=1000, help="Model save interval in steps") # 配置文件中似乎没有，但命令行参数中可以有
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def train_joint(config, logger, args):
    # --- 设备选择逻辑 ---
    if args.use_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")
        if num_gpus > 1:
            device_id_to_use = 1  # 目标使用 GPU 1
            try:
                # 设置 PyTorch 当前使用的 CUDA 设备
                torch.cuda.set_device(device_id_to_use)
                device_str = f'cuda:{device_id_to_use}'
                device = torch.device(device_str)
                logger.info(f"Successfully set CUDA device to: GPU {device_id_to_use} ({torch.cuda.get_device_name(device_id_to_use)})")
            except Exception as e:
                logger.error(f"Could not set CUDA device to GPU {device_id_to_use}. Error: {e}. Falling back to cuda:0.")
                device = torch.device('cuda:0') # 出错则回退到 GPU 0
        elif num_gpus == 1:
            device = torch.device('cuda:0') # 只有一个 GPU，使用 GPU 0
            logger.info(f"Only one CUDA device available. Using GPU 0 ({torch.cuda.get_device_name(0)})")
        else: # num_gpus == 0, 但 torch.cuda.is_available() 为 True 的情况不太可能
            logger.warning("torch.cuda.is_available() is True, but torch.cuda.device_count() is 0. This is unusual. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        if args.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
        else:
            logger.info("GPU not requested or CUDA not available. Using CPU.")
        device = torch.device('cpu')

    logger.info(f"--- Final device for training: {device} ---")
    # --- 配置校验 ---
    # ... (配置校验逻辑保持不变) ...
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

    if scale_factor % masker_patch_size != 0: # 已在日志中作为 WARNING 出现
         logger.warning(f"Scale factor ({scale_factor}) is not divisible by masker output_patch_size ({masker_patch_size}). This might lead to issues with mask alignment.")

    masker_threshold = masker_config.get('threshold')
    if not isinstance(masker_threshold, (int, float)) or not (0 <= masker_threshold <= 1):
         logger.warning(f"Masker threshold ({masker_threshold}) is outside the expected range [0, 1].")

    target_sparsity_ratio = config.get('train', {}).get('target_sparsity_ratio')
    if target_sparsity_ratio is not None and (not isinstance(target_sparsity_ratio, (int, float)) or not (0 <= target_sparsity_ratio <= 1)):
         logger.warning(f"Target sparsity ratio ({target_sparsity_ratio}) is outside the expected range [0, 1].")


    required_weights = ['detector', 'sr_fast', 'sr_quality']
    for weight_key in required_weights:
        weight_path = weights_config.get(weight_key)
        if not weight_path or not os.path.exists(weight_path): # 确保文件存在
            logger.error(f"Required weight file not found for '{weight_key}': {weight_path}. Exiting.")
            return # 如果权重文件缺失则退出

    eval_config = config.get('evaluation', {})
    if eval_config.get('val_image_dir') or eval_config.get('val_annotation_file'):
        if not eval_config.get('val_image_dir') or not os.path.exists(eval_config['val_image_dir']):
             logger.warning(f"Validation image directory not found: {eval_config.get('val_image_dir')}. Evaluation might be skipped or fail.")
        if not eval_config.get('val_annotation_file') or not os.path.exists(eval_config['val_annotation_file']):
             logger.warning(f"Validation annotation file not found: {eval_config.get('val_annotation_file')}. Evaluation might be skipped or fail.")
    logger.info("--- Configuration Validated ---")
    # --- End Configuration Validation ---

    # --- 数据加载 ---
    # ... (数据加载逻辑保持不变, 但它们会使用上面定义的 device 变量) ...
    dataset_config = config['dataset']
    try:
        train_dataset = DetectionDataset(
            image_dir=os.path.join(dataset_config['image_dir'], "LR"), # 假设LR图像在 image_dir/LR 下
            annotation_file=dataset_config['annotation_file'],
            transform=transforms.ToTensor(),
            return_image_id=True # 确保COCO评估时image_id可用
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'], # 使用配置文件中的batch_size
            shuffle=True,
            num_workers=config['train'].get('num_workers', 0), # 使用配置或默认为0
            pin_memory=True if str(device).startswith("cuda") else False, # pin_memory只在CUDA上有效
            collate_fn=getattr(train_dataset, 'collate_fn', None) # 使用数据集的collate_fn
        )
        logger.info(f"Train dataloader initialized with {len(train_dataset)} images. Batch size: {config['train']['batch_size']}.")
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
                image_dir=os.path.join(eval_config['val_image_dir'], "LR"), # 假设LR图像在 val_image_dir/LR 下
                annotation_file=eval_config['val_annotation_file'],
                transform=transforms.ToTensor(),
                return_image_id=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['train'].get('val_batch_size', 1), # 验证时通常batch_size为1
                shuffle=False,
                num_workers=config['train'].get('num_workers', 0),
                pin_memory=True if str(device).startswith("cuda") else False,
                collate_fn=getattr(val_dataset, 'collate_fn', None)
            )
            logger.info(f"Validation dataloader initialized with {len(val_dataset)} images.")
        except FileNotFoundError as e:
            logger.error(f"Validation data/annotation file not found: {e}. Evaluation will be skipped.")
            val_dataloader = None # 确保验证dataloader为None
        except Exception as e:
            logger.error(f"Error initializing validation dataloader: {e}. Evaluation will be skipped.")
            val_dataloader = None # 确保验证dataloader为None
    else:
        logger.warning("Validation image directory or annotation file not specified in config. Evaluation will be skipped.")


    # --- 模型初始化 ---
    # 注意: device 对象 (例如 torch.device('cuda:1')) 会被传递给 ConditionalSR
    # ConditionalSR 内部会使用这个 device 来初始化其子模块和 DetectorWrapper
    # ... (模型初始化逻辑保持不变) ...
    sr_fast_config = config['model']['sr_fast']
    sr_quality_config = config['model']['sr_quality']
    masker_config_full = config['model']['masker']

    valid_masker_init_keys = [
        'in_channels', 'base_channels', 'num_blocks', 'output_channels', 'output_patch_size'
    ]
    masker_init_args = {k: masker_config_full[k] for k in valid_masker_init_keys if k in masker_config_full}

    sr_fast_model = SRFast(**sr_fast_config)
    sr_quality_model = SRQuality(**sr_quality_config)
    masker_model = Masker(**masker_init_args)

    # device 变量在这里是 torch.device('cuda:1') 或 torch.device('cpu') 等
    conditional_sr = ConditionalSR(
        sr_fast=sr_fast_model,
        sr_quality=sr_quality_model,
        masker=masker_model,
        detector_weights=config['model']['weights']['detector'],
        sr_fast_weights=config['model']['weights']['sr_fast'],
        sr_quality_weights=config['model']['weights']['sr_quality'],
        masker_weights=config['model']['weights'].get('masker', None),
        device=str(device), # ConditionalSR 和 DetectorWrapper 通常期望字符串 "cuda:1" 或 "cpu"
        config=config
    ).to(device) # 将整个模型移动到目标设备


    # --- 优化器和调度器 ---
    # ... (优化器和调度器逻辑保持不变) ...
    optimizer = get_optimizer_with_differential_lr(conditional_sr, config) # 差分学习率
    scheduler = None
    scheduler_config = config['train'].get('scheduler', {}) # 安全获取调度器配置
    if scheduler_config.get('name', '').lower() == 'cosineannealinglr':
        scheduler_args = scheduler_config.get('args', {})
        # total_steps = config['train']['epochs'] * len(train_dataloader) # 如果T_max是基于step
        
        # T_max 可以是 epoch 数也可以是 step 数，根据 Ultralytics 的 YOLO 行为，通常是 epoch 数
        # 但 stage3_finetune_joint.py 日志显示 "Converting to steps: 300"
        # 这意味着 T_max 在配置文件中可能是 epoch 数
        # 我们需要在配置文件中明确 T_max 的单位，或者让脚本更智能地处理
        
        # 假设 T_max 在配置文件中指的是 epoch 数，并且 anneal_steps 也是基于 epoch
        # 如果是 CosineAnnealingLR，T_max 通常是总的 step 数或 epoch 数
        # 日志说 "Converting to steps: 300"，假设 epochs = 30, len(dataloader) = 10
        # 所以 T_max = 30 * 10 = 300 (steps)
        
        # T_max 单位需要与 scheduler.step() 的调用频率匹配
        # 如果 scheduler.step() 每 step 调用，T_max 就是总 step 数
        # 如果 scheduler.step() 每 epoch 调用，T_max 就是总 epoch 数

        # 根据日志 "WARNING:root:Assuming scheduler T_max (30) is in epochs. Converting to steps: 300"
        # 我们这里让 T_max = 配置文件中的 T_max * len(train_dataloader)
        # 如果配置文件中的 T_max 已经是 step 数，则直接使用
        
        # 简化：我们假设 scheduler.step() 是每个 step 调用一次，所以 T_max 是总的 step 数。
        # 如果不是，需要调整。
        # 但日志中 T_max=300 steps，而 epochs=150（从tqdm看），每个epoch 10个step
        # 这意味着 Anneal Steps 是 1500 (150*10)，而 T_max 只有 300.
        # 这会导致学习率在早期就退火完毕。
        # CosineAnnealingLR 的 T_max 应该是总的迭代次数。
        
        # 我们将使用配置文件中 scheduler:args:T_max，并假设它是 step 数
        # 如果不是，用户需要在配置文件中调整
        if 'T_max' in scheduler_args and 'eta_min' in scheduler_args:
            # 从日志看，T_max=30，但被转换成了300。len(train_dataloader) = 10 (1252 images / batch 8 ~ 157 batches, not 10)
            # 日志的 epochs 是 150. tqdm 是 0/10.
            # 假设配置文件中的 batch_size=8, epochs=30 (配置文件原始值)
            # len(train_dataloader) = ceil(1252 / 8) = 157
            # total_steps_for_config_epochs = 30 * 157 = 4710
            # scheduler_T_max = scheduler_args['T_max'] # 直接使用配置文件中的 T_max
            # if scheduler_T_max <= config['train']['epochs']: # 如果 T_max 看起来像 epoch 数
            #     logger.warning(f"Scheduler T_max ({scheduler_T_max}) seems to be in epochs. Consider setting it in steps or adjusting this logic.")
            #     scheduler_T_max_steps = scheduler_T_max * len(train_dataloader)
            # else: # 假设 T_max 已经是 step 数
            #     scheduler_T_max_steps = scheduler_T_max

            # 根据日志 "Converting to steps: 300"，这里暂时也设为300，但用户应检查配置文件
            # 或根据实际的 epochs * len(train_dataloader) 来设置
            scheduler_T_max_steps = config['train']['epochs'] * len(train_dataloader) # 更安全的方式是基于总的训练迭代次数
            # 如果配置文件中有 T_max，并且它大于 epochs，则假设它是 step 数
            if 'T_max' in scheduler_args and scheduler_args['T_max'] > config['train']['epochs']:
                 scheduler_T_max_steps = scheduler_args['T_max']
            else: # 否则，根据 epochs 和 dataloader 长度计算总 steps
                 scheduler_T_max_steps = config['train']['epochs'] * len(train_dataloader)
                 if 'T_max' in scheduler_args: # 如果配置文件里有 T_max 且较小，则提示
                      logger.warning(f"Scheduler T_max in config ({scheduler_args['T_max']}) is small. Using calculated total steps: {scheduler_T_max_steps}. Ensure T_max in config is total steps if intended.")


            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_T_max_steps, # 总的迭代步数
                eta_min=scheduler_args['eta_min']
            )
            logger.info(f"Using CosineAnnealingLR scheduler with T_max={scheduler_T_max_steps} steps and eta_min={scheduler_args['eta_min']}.")
        else:
            logger.warning("Scheduler config for CosineAnnealingLR missing T_max or eta_min. Scheduler disabled.")
            scheduler = None # 确保禁用
    elif scheduler_config.get('name'): # 如果指定了其他类型的调度器但未处理
        logger.warning(f"Scheduler '{scheduler_config.get('name')}' not explicitly handled. Scheduler disabled.")
        scheduler = None
    else: # 没有配置调度器
        logger.info("Scheduler not configured. Scheduler disabled.")
        scheduler = None


    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], "tensorboard"))

    logger.info("Starting joint finetuning...")
    best_map = 0.0
    global_step = 0

    # --- Gumbel 温度设置 ---
    # ... (Gumbel 设置逻辑保持不变) ...
    gumbel_config = config['train'].get('gumbel', {})
    initial_tau = gumbel_config.get('initial_tau', 1.0)
    final_tau = gumbel_config.get('final_tau', 0.1)
    # anneal_steps 优先于 anneal_epochs
    anneal_steps_from_config = gumbel_config.get('anneal_steps')
    if anneal_steps_from_config is not None:
        anneal_steps = anneal_steps_from_config
    else: # 如果 anneal_steps 未设置，则使用 anneal_epochs
        anneal_epochs = gumbel_config.get('anneal_epochs', config['train']['epochs']) # 默认为总 epochs
        anneal_steps = anneal_epochs * len(train_dataloader)
        logger.info(f"Gumbel anneal_steps not set, using anneal_epochs ({anneal_epochs}) * len(dataloader) = {anneal_steps} steps.")

    anneal_schedule = gumbel_config.get('anneal_schedule', 'linear').lower()
    use_annealing = anneal_steps > 0 and initial_tau != final_tau
    logger.info(f"Gumbel Annealing: Use={use_annealing}, Initial Tau={initial_tau}, Final Tau={final_tau}, Anneal Steps={anneal_steps}, Schedule={anneal_schedule}")


    # --- 训练循环 ---
    # epochs_to_run = config['train']['epochs'] # 从tqdm看是150，但配置文件是30
    # 日志显示 Epoch 1/150，而配置文件是 epochs: 30。
    # 这表明 len(train_dataloader) 可能被错误地计算为 10（在tqdm中）。
    # 或者 epochs 被硬编码或从其他地方读取为150。
    # 为了安全，我们使用配置文件中的 epochs。
    # 如果 tqdm 的总数是基于 step，那它应该是 epochs * len(train_dataloader)
    # 而不是 len(train_dataloader)

    # 实际的 epochs 数应来自 config['train']['epochs']
    num_epochs_from_config = config['train']['epochs']

    for epoch in range(num_epochs_from_config): # 使用配置文件中的 epochs
        conditional_sr.train() # 确保模型在训练模式
        epoch_loss = 0.0
        
        # tqdm 的 total 应该是 len(train_dataloader)
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs_from_config}", total=len(train_dataloader))

        for lr_images, targets in progress_bar:
            # 数据移动到设备
            lr_images = lr_images.to(device)
            targets_on_device = []
            if targets: # targets 是一个字典列表
                for t_dict in targets:
                    # t_dict 是单个图像的标注字典: {'boxes': tensor, 'labels': tensor, 'image_id': tensor}
                    target_item_on_device = {}
                    for k, v_tensor in t_dict.items():
                        if isinstance(v_tensor, torch.Tensor):
                            target_item_on_device[k] = v_tensor.to(device)
                        else: # image_id 可能不是 tensor，如果 collate_fn 没有处理
                            target_item_on_device[k] = v_tensor 
                    targets_on_device.append(target_item_on_device)
            else: # 如果整个批次的 targets 为 None (例如，如果 collate_fn 返回了空 targets)
                targets_on_device = None # 或者一个空列表，取决于后续处理


            # Gumbel 温度更新
            current_tau = initial_tau
            if use_annealing:
                if global_step < anneal_steps:
                    if anneal_schedule == 'linear':
                        anneal_progress = global_step / anneal_steps
                        current_tau = initial_tau - (initial_tau - final_tau) * anneal_progress
                    elif anneal_schedule == 'cosine': # 余弦退火
                        current_tau = final_tau + 0.5 * (initial_tau - final_tau) * (1 + math.cos(math.pi * global_step / anneal_steps))
                    # ... (其他退火策略)
                else:
                    current_tau = final_tau
                current_tau = max(final_tau, current_tau) #确保不低于final_tau


            optimizer.zero_grad()
            
            # 前向传播
            outputs = conditional_sr(lr_images, targets=targets_on_device, temperature=current_tau)



            # --- 调试代码开始 ---
            if global_step == 0: 
                logger.info(f"--- Debugging outputs at global_step {global_step} ---")
                logger.info(f"Type of outputs: {type(outputs)}")
                if isinstance(outputs, dict):
                    logger.info(f"Keys in outputs: {list(outputs.keys())}")
                    for key, value in outputs.items():
                        if key == "yolo_raw_predictions": # <--- 重点关注这个键
                            logger.info(f"  Key: '{key}', Value type: {type(value)}")
                            if isinstance(value, list):
                                logger.info(f"    Length of list: {len(value)}")
                                for i, item in enumerate(value):
                                    if torch.is_tensor(item):
                                        logger.info(f"      Item {i} type: Tensor, Shape: {item.shape}, Device: {item.device}, Dtype: {item.dtype}")
                                    else:
                                        logger.info(f"      Item {i} type: {type(item)}")
                            # 如果 yolo_raw_predictions 直接是张量而不是列表（不太可能，但以防万一）
                            elif torch.is_tensor(value):
                                logger.info(f"  Key: '{key}', Value type: Tensor, Shape: {value.shape}, Device: {value.device}, Dtype: {value.dtype}")

                        elif torch.is_tensor(value):
                            logger.info(f"  Key: '{key}', Value type: Tensor, Shape: {value.shape}, Device: {value.device}")
                        elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                            logger.info(f"  Key: '{key}', Value type: List[dict], Length: {len(value)}")
                        elif value is None:
                            logger.info(f"  Key: '{key}', Value type: NoneType")
                        else:
                            logger.info(f"  Key: '{key}', Value type: {type(value)}")
                else:
                    logger.info(f"Outputs is not a dictionary. Value: {outputs}")
                logger.info(f"--- End Debugging outputs ---")
            # --- 调试代码结束 ---



            sr_images = outputs["sr_image"]
            mask_coarse = outputs["mask_coarse"]
            yolo_raw_predictions_from_model = outputs["yolo_raw_predictions"]

            # 计算损失
            total_loss, loss_dict = calculate_joint_loss(
                sr_images=sr_images,
                mask_coarse=mask_coarse,
                targets=targets_on_device, 
                yolo_raw_predictions=yolo_raw_predictions_from_model,
                config=config,
                logger=logger,
                precomputed_detection_loss=None # 明确设为 None
            )

            if torch.isnan(total_loss):
                logger.error(f"NaN loss detected at step {global_step}. Stopping training.")
                if writer: writer.close()
                return # 终止训练

            total_loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(conditional_sr.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler:
                scheduler.step() # 每个 step 更新学习率

            epoch_loss += total_loss.item()
            
            # 更新进度条显示
            postfix_dict = {
                "total_loss": f"{total_loss.item():.4f}",
                "det": f"{loss_dict.get('loss_detection', 0.0):.4f}",
                "spar": f"{loss_dict.get('loss_sparsity', 0.0):.4f}",
                "smooth": f"{loss_dict.get('loss_smooth', 0.0):.4f}",
                "tau": f"{current_tau:.2f}"
            }
            # 添加学习率到进度条
            if optimizer and optimizer.param_groups:
                 postfix_dict["lr_high"] = f"{optimizer.param_groups[0]['lr']:.1e}" # 高学习率组
                 if len(optimizer.param_groups) > 1:
                      postfix_dict["lr_low"] = f"{optimizer.param_groups[1]['lr']:.1e}" # 低学习率组
            progress_bar.set_postfix(postfix_dict)


            # 日志记录到 TensorBoard
            log_interval_steps = config['train'].get('log_interval_steps', 10) # 从配置读取
            if global_step % log_interval_steps == 0 and writer:
                writer.add_scalar("Train/TotalLoss", total_loss.item(), global_step)
                writer.add_scalar("Train/GumbelTau", current_tau, global_step)
                for loss_name_key, loss_val_item in loss_dict.items():
                    if loss_name_key not in ["total_loss", "actual_sparsity"]: # total_loss 已记录, actual_sparsity 不是损失
                        writer.add_scalar(f"Train/Loss_{loss_name_key.replace('loss_', '')}", loss_val_item, global_step)
                writer.add_scalar("Train/ActualSparsity", loss_dict.get("actual_sparsity", 0.0), global_step)


                # 记录学习率
                if optimizer and optimizer.param_groups:
                    writer.add_scalar("Train/LR_Group_High", optimizer.param_groups[0]['lr'], global_step)
                    if len(optimizer.param_groups) > 1:
                        writer.add_scalar("Train/LR_Group_Low", optimizer.param_groups[1]['lr'], global_step)
                
                # (可选) 记录掩码的熵或均值等
                if mask_coarse is not None:
                    # actual_sparsity = torch.mean(mask_coarse.float()).item() # 已在 loss_dict 中
                    # writer.add_scalar("Train/Mask_ActualSparsity", actual_sparsity, global_step)
                    if mask_coarse.min() >= 0 and mask_coarse.max() <= 1 and mask_coarse.numel() > 0:
                        # 计算二元熵 H(p) = -p*log2(p) - (1-p)*log2(1-p)
                        # 为了数值稳定，添加 epsilon
                        epsilon = 1e-8
                        p = mask_coarse.float()
                        entropy = - (p * torch.log2(p + epsilon) + (1 - p) * torch.log2(1 - p + epsilon)).mean().item()
                        writer.add_scalar("Train/Mask_Entropy", entropy, global_step)


            # 模型评估与保存
            # 使用 args 中的 eval_interval 和 save_interval (命令行参数)
            if val_dataloader and args.eval_interval > 0 and global_step > 0 and global_step % args.eval_interval == 0:
                logger.info(f"Step {global_step}: Running evaluation...")
                map_results, avg_sparsity_val = run_coco_evaluation(
                    model=conditional_sr,
                    dataloader=val_dataloader,
                    device=device, # 使用当前选择的设备
                    annotation_file=config['evaluation']['val_annotation_file'], # 确保路径正确
                    output_dir=os.path.join(config['log_dir'], "eval_results"), # 保存评估结果的子目录
                    step_or_epoch=f"step_{global_step}", # 文件名中包含step
                    logger=logger,
                    use_hard_mask=True # 评估时通常使用硬掩码
                )
                if writer:
                    writer.add_scalar("Validation/mAP50", map_results.get('map_50', 0.0), global_step)
                    writer.add_scalar("Validation/mAP", map_results.get('map', 0.0), global_step)
                    writer.add_scalar("Validation/Sparsity", avg_sparsity_val, global_step)

                current_map50 = map_results.get('map_50', 0.0)
                if current_map50 > best_map:
                    best_map = current_map50
                    save_path_best = os.path.join(config['checkpoint_dir'], "joint_best.pth")
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': conditional_sr.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'map50': best_map,
                        'config': config # 保存配置以备后续加载
                    }, save_path_best)
                    logger.info(f"Saved best model (mAP50: {best_map:.4f}) to {save_path_best}")

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                save_path_step = os.path.join(config['checkpoint_dir'], f"joint_step{global_step}.pth")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': conditional_sr.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config
                }, save_path_step)
                logger.info(f"Saved checkpoint to {save_path_step}")

            global_step += 1
        # --- 每个 epoch 结束 ---
        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs_from_config} Average Loss: {avg_epoch_loss:.4f}")
        # (可选) 如果 scheduler.step() 是每个 epoch 调用，则放在这里
        # if scheduler and config['train']['scheduler'].get('step_per_epoch', True):
        #     scheduler.step()


    # --- 训练结束 ---
    if writer: writer.close()
    logger.info("Joint finetuning completed.")
    final_save_path = os.path.join(config['checkpoint_dir'], "joint_final.pth")
    torch.save({
        'step': global_step,
        'epoch': num_epochs_from_config,
        'model_state_dict': conditional_sr.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config
    }, final_save_path)
    logger.info(f"Saved final model to {final_save_path}")


def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f: # 确保UTF-8编码
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e: # 更具体的YAML错误捕获
        print(f"Error parsing YAML configuration file: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # 创建日志和检查点目录 (如果不存在)
    # 日志目录从配置中读取
    log_dir_from_config = config.get('log_dir', './temp_logs/stage3_joint') # 提供默认值
    checkpoint_dir_from_config = config.get('checkpoint_dir', './temp_checkpoints/stage3_joint')

    os.makedirs(log_dir_from_config, exist_ok=True)
    os.makedirs(checkpoint_dir_from_config, exist_ok=True)
    
    # 使用配置中的目录设置日志记录器
    logger = setup_logger(log_dir_from_config, "stage3_finetune.log")
    logger.info("Starting Stage 3: Joint Finetuning")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Command line arguments: {args}")


    train_joint(config, logger, args)

if __name__ == "__main__":
    main()