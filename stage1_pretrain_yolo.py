# stage1_pretrain_yolo.py
import os
import argparse
import yaml
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch # 确保导入 torch
import logging # 导入 logging

# 从 utils 导入
from utils.logger import setup_logger, set_logger_level
from utils.common_utils import get_device
from utils.config_utils import validate_config

# 该脚本用于预训练YOLO检测器。

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain YOLO on High-Resolution Data")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file (stage1_yolo_pretrain.yaml)")
    parser.add_argument("--save_interval", type=int, default=1000, # 与原始脚本一致
                        help="Save model interval in steps (step-based for callback). Ultralytics default is epoch-based.")
    parser.add_argument("--eval_interval", type=int, default=500, # 与原始脚本一致
                        help="Evaluation interval in steps (step-based for callback). Ultralytics default is epoch-based.")
    parser.add_argument("--enable_eval", action="store_true",
                        help="Enable evaluation during training via callbacks (if supported and configured).")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    # YOLO 通常自己处理断点续训，通过 resume=True 或 resume='path/to/last.pt' 参数
    # parser.add_argument("--resume_path", type=str, default=None, help="Path to a YOLO checkpoint to resume training from (optional).")
    return parser.parse_args()

def train_yolo(config: Dict, logger: logging.Logger, args: argparse.Namespace, device: torch.device):
    """使用 ultralytics API 训练 YOLO 模型"""
    # 从配置中获取路径
    # log_dir 和 ckpt_dir 现在从 config['train'] 或 config 的顶层获取
    # (与 stage2, stage3 保持一致，在 main 函数中处理)
    log_dir_main = config.get('log_dir', './temp_logs/stage1_yolo')
    ckpt_dir_main = config.get('checkpoint_dir', './temp_checkpoints/stage1_yolo')
    # setup_logger 已经在 main 中调用，这里直接使用传入的 logger

    # TensorBoard writer
    tensorboard_log_dir = os.path.join(log_dir_main, "tensorboard_stage1_yolo")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard 日志将保存到: {tensorboard_log_dir}")

    # --- 数据集准备 ---
    dataset_config = config['dataset']
    class_names = dataset_config.get('class_names', ['object']) # 从config读取，提供默认值
    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        logger.error(f"配置中的 class_names 格式不正确，应为字符串列表: {class_names}")
        writer.close()
        return
    num_classes = len(class_names)

    # 强制使用绝对路径并处理路径分隔符，确保在所有操作系统上一致
    try:
        train_path_abs = os.path.abspath(dataset_config['train_image_dir']).replace('\\', '/')
        val_path_abs = ""
        if args.enable_eval and dataset_config.get('val_image_dir'):
            val_path_abs = os.path.abspath(dataset_config['val_image_dir']).replace('\\', '/')
        elif args.enable_eval:
            logger.warning("启用了评估，但 dataset.val_image_dir 未在配置中提供。评估可能无法运行。")
    except KeyError as e:
        logger.error(f"配置 dataset 部分缺少键: {e}。请检查 train_image_dir 或 val_image_dir。")
        writer.close()
        return

    # 动态生成 data_stage1.yaml 的内容
    # 将其保存在主日志目录下，而不是 TensorBoard 子目录
    data_yaml_path = os.path.join(log_dir_main, "data_stage1_yolo.yaml")
    data_yaml_content = f"""
train: {train_path_abs}
val: {val_path_abs if args.enable_eval and val_path_abs else '# No validation path specified or eval disabled'}
nc: {num_classes}
names: {class_names}
"""
    try:
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            f.write(data_yaml_content)
        logger.info(f"已为 YOLO 生成数据配置文件: {data_yaml_path}")
        logger.info(f"请确保 {data_yaml_path} 中的类别名称 ({class_names}) 和数量 nc ({num_classes}) 与您的数据集匹配！")
        logger.info(f"并且图像目录 ({train_path_abs}, {val_path_abs if val_path_abs else 'N/A'}) 下应有 'labels' 子目录包含标注文件。")
    except IOError as e:
        logger.error(f"无法写入 YOLO 数据配置文件 {data_yaml_path}: {e}")
        writer.close()
        return

    # --- 模型加载 ---
    model_version_or_path = config['model']['version'] # 例如 "yolov8n.pt" 或预训练模型路径
    try:
        model = YOLO(model_version_or_path)
        logger.info(f"已加载 YOLO 模型: {model_version_or_path}")
    except Exception as e:
        logger.error(f"加载 YOLO 模型 {model_version_or_path} 失败: {e}", exc_info=True)
        writer.close()
        return

    # --- 训练参数 ---
    train_params_cfg = config['train']
    optimizer_cfg = train_params_cfg.get('optimizer', {})
    scheduler_cfg = train_params_cfg.get('scheduler', {})

    # Ultralytics 的 train 方法接受许多参数，这里映射配置文件中的值
    # 对于 'lr0' 和 'lrf' (学习率相关):
    # lr0 是初始学习率。
    # lrf 是最终学习率与初始学习率的比率 (lr_final = lr0 * lrf)。
    # 如果使用 CosineAnnealingLR，lrf 约等于 eta_min / lr0。
    initial_lr = optimizer_cfg.get('args', {}).get('lr', 0.01) # 默认一个初始学习率
    final_lr_factor = 0.01 # 默认的最终学习率因子 (YOLO 常用)
    if scheduler_cfg.get('name', '').lower() == 'cosineannealinglr':
        eta_min = scheduler_cfg.get('args', {}).get('eta_min', 0.0001) # 假设的 eta_min
        if initial_lr > 0: # 避免除零
            final_lr_factor = eta_min / initial_lr
        else:
            logger.warning("初始学习率为0，无法计算 final_lr_factor。使用默认值。")

    yolo_train_args = {
        'data': data_yaml_path,
        'epochs': train_params_cfg['epochs'],
        'batch': train_params_cfg['batch_size'],
        'imgsz': dataset_config.get('input_size', 640), # 图像尺寸
        'optimizer': optimizer_cfg.get('name', 'AdamW'), # 'SGD', 'Adam', 'AdamW' 等
        'lr0': initial_lr,
        'lrf': final_lr_factor, # (float) final OneCycleLR learning rate (lr0 * lrf)
        'weight_decay': optimizer_cfg.get('args', {}).get('weight_decay', 0.0005),
        'device': device.type if device.type == "cuda" else 'cpu', # YOLO期望 'cpu' 或 '0', '0,1' 等
        'workers': train_params_cfg.get('num_workers', 8),
        'project': ckpt_dir_main, # 保存到主检查点目录
        'name': 'yolo_pretrain_run', # 运行的子目录名
        'seed': train_params_cfg.get('seed', 42),
        'save_period': -1, # Ultralytics 默认按 epoch 保存最好的和最后的，-1 禁用基于epoch的定期保存，依赖回调
        'val': args.enable_eval, # 根据命令行参数决定是否在训练时进行验证
        # 'resume': args.resume_path if args.resume_path else False, # YOLO的resume参数
        # 其他可配置参数: patience, momentum, warmup_epochs, etc.
    }
    # 如果是 CUDA，可以指定具体设备索引，或者 YOLO 会自动选择
    if device.type == "cuda":
        yolo_train_args['device'] = ','.join(map(str, range(torch.cuda.device_count()))) if torch.cuda.device_count() > 0 else 'cpu'
        # 或者更简单地： yolo_train_args['device'] = 'cuda' 让 ultralytics 处理

    logger.info(f"开始 YOLO 训练，参数: {yolo_train_args}")
    logger.info(f"基于步骤的回调保存间隔: {args.save_interval}")
    logger.info(f"基于步骤的回调评估间隔: {args.eval_interval}")
    logger.info(f"通过回调启用评估: {args.enable_eval}")

    # --- 定义回调函数 ---
    # global_step 需要在回调函数外部定义，并通过 nonlocal 访问
    # 但 YOLO 的回调是实例方法或静态方法，直接修改外部变量不直接
    # 通常，YOLO 的 trainer 对象会包含当前步数等信息
    # 我们将依赖 trainer.epoch 和 trainer.batch (当前 epoch 内的 batch 索引)
    # 以及 trainer.stopper.current_epoch 和 trainer.stopper.max_epochs

    # 使用一个小的包装类来维护 global_step，因为回调是作为方法添加到模型中的
    class StepTracker:
        def __init__(self):
            self.global_step = 0
            self.last_saved_step = -args.save_interval #确保第一次保存能触发
            self.last_eval_step = -args.eval_interval  #确保第一次评估能触发


    step_tracker = StepTracker()

    def on_train_batch_end_callback(trainer):
        try:
            step_tracker.global_step += 1 # trainer.total_batches_seen or similar might be better if available

            # 记录训练损失 (trainer.loss 是最近一次迭代的损失)
            if trainer.loss is not None:
                writer.add_scalar("Train/Loss_Step", trainer.loss.item(), step_tracker.global_step)
            else:
                logger.warning(f"步骤 {step_tracker.global_step}: Trainer.loss 为 None。跳过损失记录。")

            # 记录学习率
            if trainer.optimizer:
                current_lr = trainer.optimizer.param_groups[0]['lr']
                writer.add_scalar("Train/LearningRate_Step", current_lr, step_tracker.global_step)
            else:
                logger.warning(f"步骤 {step_tracker.global_step}: Trainer.optimizer 为 None。跳过学习率记录。")


            # 基于步骤的评估 (如果启用了主验证 `val=True`，YOLO会在每个epoch结束时评估)
            #这里的回调是补充性的基于步骤的评估
            if args.enable_eval and args.eval_interval > 0 and \
               (step_tracker.global_step - step_tracker.last_eval_step >= args.eval_interval):
                logger.info(f"回调：步骤 {step_tracker.global_step}: 运行基于步骤的评估...")
                # trainer.validator.run_callbacks("on_val_start") # 可能需要
                # val_results = trainer.validator() # 执行验证
                # trainer.validator.run_callbacks("on_val_end") # 可能需要
                # if val_results and hasattr(val_results, 'box') and hasattr(val_results.box, 'map50'):
                #     writer.add_scalar("Validation/mAP50_StepCallback", val_results.box.map50, step_tracker.global_step)
                #     writer.add_scalar("Validation/mAP_StepCallback", val_results.box.map, step_tracker.global_step)
                # logger.info(f"回调：步骤 {step_tracker.global_step}: 基于步骤的评估完成。mAP50: {val_results.box.map50:.4f}")
                # 当前YOLO版本 (8.x) 的回调中直接触发验证比较复杂，
                # 依赖于 trainer.val() 或类似方法，这通常会执行完整的验证流程。
                # 更简单的方式是依赖YOLO在epoch结束时的标准验证流程，并通过 `val=True` 启用。
                # 如果确实需要严格按step评估，可能需要更深入地定制Trainer。
                # 此处我们仅记录一个信息，真正的验证由YOLO的 `val=True` 控制。
                logger.info(f"回调：达到步骤 {step_tracker.global_step} 的评估间隔。YOLO将在epoch结束时进行标准验证。")
                step_tracker.last_eval_step = step_tracker.global_step


            # 基于步骤的模型保存
            if args.save_interval > 0 and \
               (step_tracker.global_step - step_tracker.last_saved_step >= args.save_interval) :
                # trainer.save_model() 会保存到默认的 run 目录的 weights 下
                # 如果要保存到特定路径，可能需要 trainer.ckpt_path = new_path
                current_epoch_for_save = trainer.epoch +1 # 当前正在进行的 epoch (0-indexed)
                save_path = os.path.join(trainer.save_dir, f"weights", f"step_{step_tracker.global_step}_epoch_{current_epoch_for_save}.pt")
                trainer.save_model(save_path) # Ultralytics >=8.0.106 之后 trainer.save_model() 可以接受路径
                                              # 老版本可能需要 trainer.model.save(save_path)
                logger.info(f"回调：模型检查点已在步骤 {step_tracker.global_step} (Epoch {current_epoch_for_save}) 保存到: {save_path}")
                step_tracker.last_saved_step = step_tracker.global_step


        except Exception as e_callback:
            logger.error(f"on_train_batch_end 回调函数中发生错误 (步骤 {step_tracker.global_step}): {e_callback}", exc_info=True)

    # 添加回调
    model.add_callback("on_train_batch_end", on_train_batch_end_callback)
    # 其他有用的回调: on_epoch_end, on_fit_epoch_end, on_val_end

    # --- 开始训练 ---
    try:
        model.train(**yolo_train_args)
        logger.info("YOLO 训练完成。")

        # 训练完成后，YOLO 会自动将 best.pt 和 last.pt 保存在 project/name/weights/ 目录下
        # 将 best.pt 复制到目标路径
        # yolo_train_args['project'] 是 ckpt_dir_main
        # yolo_train_args['name'] 是 'yolo_pretrain_run'
        default_best_model_path = os.path.join(ckpt_dir_main, 'yolo_pretrain_run', 'weights', 'best.pt')
        target_final_model_path = os.path.join(ckpt_dir_main, "yolo_pretrained_hr.pt") # 与配置中的目标名称一致

        if os.path.exists(default_best_model_path):
            shutil.copyfile(default_best_model_path, target_final_model_path)
            logger.info(f"最佳模型已从 {default_best_model_path} 复制到: {target_final_model_path}")
        else:
            logger.warning(f"训练后未找到最佳模型 {default_best_model_path}。请检查训练过程和保存路径。")
            # 可以考虑复制 last.pt 作为备用
            default_last_model_path = os.path.join(ckpt_dir_main, 'yolo_pretrain_run', 'weights', 'last.pt')
            if os.path.exists(default_last_model_path):
                shutil.copyfile(default_last_model_path, target_final_model_path)
                logger.info(f"最终模型已从 {default_last_model_path} 复制到: {target_final_model_path} (作为 best.pt 的备用)")
            else:
                 logger.error(f"也未找到最终模型 {default_last_model_path}。")


    except Exception as e_train:
        logger.error(f"YOLO 训练过程中发生严重错误: {e_train}", exc_info=True)
    finally:
        if writer:
            writer.close()


def main():
    args = parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件在 {args.config} 未找到")
        exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件时发生错误: {e}")
        exit(1)

    # 日志和检查点目录
    log_dir_cfg = config.get('log_dir', './temp_logs/stage1_yolo')
    ckpt_dir_cfg = config.get('checkpoint_dir', './temp_checkpoints/stage1_yolo')
    os.makedirs(log_dir_cfg, exist_ok=True)
    os.makedirs(ckpt_dir_cfg, exist_ok=True)

    # 将配置中的路径也更新为绝对路径，如果它们是相对的
    config['log_dir'] = os.path.abspath(log_dir_cfg)
    config['checkpoint_dir'] = os.path.abspath(ckpt_dir_cfg)


    logger = setup_logger(config['log_dir'], "stage1_yolo_pretrain.log") # 使用更新后的绝对路径

    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logger.critical(f'无效的日志级别: {args.log_level}')
        raise ValueError(f'无效的日志级别: {args.log_level}')
    set_logger_level(logger, numeric_log_level)
    logger.info(f"日志级别已设置为: {args.log_level}")

    logger.info("--- 开始阶段 1: YOLO 在高分辨率数据上预训练 ---")
    logger.info(f"已从以下路径加载配置: {args.config}")
    logger.info(f"命令行参数: {args}")

    # 设备选择
    device = get_device(args.use_gpu, logger) # device 是 torch.device 对象
    logger.info(f"将使用设备: {device} (类型: {device.type}) 进行训练。")


    # 配置校验
    config['args'] = vars(args) # 将命令行参数合并到配置中，供校验函数使用
    if not validate_config(config, "stage1_yolo", logger):
        logger.error("配置校验失败。正在退出。")
        return

    train_yolo(config, logger, args, device)

if __name__ == "__main__":
    main()