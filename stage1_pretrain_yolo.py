import os
import argparse
import yaml
from ultralytics import YOLO
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
import shutil  # Import shutil
import torch  # Import torch for cuda check

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain YOLO on High-Resolution Data")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (stage1_yolo_pretrain.yaml)")
    # --- 修改默认值 ---
    parser.add_argument("--save_interval", type=int, default=1000, help="保存模型的间隔步数 (step-based)")
    parser.add_argument("--eval_interval", type=int, default=500, help="评估模型的间隔步数 (step-based)")
    parser.add_argument("--enable_eval", action="store_true", help="是否启用评估")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    return parser.parse_args()

def train_yolo(config, args):
    """使用 ultralytics API 训练 YOLO 模型"""
    log_dir = config['log_dir']
    ckpt_dir = config['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(log_dir, "stage1_train.log")

    # --- 配置校验 ---
    logger.info("--- Validating Configuration ---")
    dataset_config = config.get('dataset', {})
    train_image_dir = dataset_config.get('train_image_dir')
    val_image_dir = dataset_config.get('val_image_dir')
    # 注意: 原始脚本中 train_annotation_file 和 val_annotation_file 未在YOLO数据准备逻辑中使用
    # train_annotation_file = dataset_config.get('train_annotation_file')
    # val_annotation_file = dataset_config.get('val_annotation_file')

    if not train_image_dir or not os.path.exists(train_image_dir):
        logger.error(f"Training image directory not found: {train_image_dir}")
        return # Exit if critical path is missing
    # if not train_annotation_file or not os.path.exists(train_annotation_file): # 这段检查可以移除，因为YOLO依赖labels文件夹
    #     logger.error(f"Training annotation file not found: {train_annotation_file}")
    #     return
    if args.enable_eval: # Only check validation paths if evaluation is enabled
        if not val_image_dir or not os.path.exists(val_image_dir):
            logger.warning(f"Validation image directory not found: {val_image_dir}. Evaluation might fail.")
        # if not val_annotation_file or not os.path.exists(val_annotation_file): # 这段检查可以移除
        #     logger.warning(f"Validation annotation file not found: {val_annotation_file}. Evaluation might fail.")

    logger.info("--- Configuration Validated ---")
    # --- End Configuration Validation ---

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    # --- 数据集准备 ---
    class_names = config.get('class_names', ['block','bolt']) # 从config读取，提供默认值
    num_classes = len(class_names)
    
    # 强制使用绝对路径并处理中文问题
    train_path = os.path.abspath(config['dataset']['train_image_dir']).replace('\\', '/')
    val_path = os.path.abspath(config['dataset']['val_image_dir']).replace('\\', '/')
    
    # data_yaml_path = os.path.abspath("dataset/Data partitioning/data.yaml").replace('\\','/') # 这行通常是动态生成的，而不是写死的

    # 动态生成 data_stage1.yaml 的内容
    data_yaml_content = f"""
train: {train_path}
val: {val_path}
nc: {num_classes}
names: {class_names}
"""
    # 将 data_stage1.yaml 保存在log_dir中
    data_yaml_path = os.path.join(log_dir, "data_stage1.yaml") # 确保这行是在正确的缩进级别
    with open(data_yaml_path, 'w', encoding='utf-8') as f: # 修正缩进
        f.write(data_yaml_content) # 修正缩进
    logger.info(f"Generated data YAML for YOLO: {data_yaml_path}") # 修正缩进
    logger.info(f"Ensure class names ({class_names}) and nc ({num_classes}) in {data_yaml_path} are correct for your dataset!") # 修正缩进

    # --- 模型加载 (保持不变) ---
    model_version = config['model']['version'] # 修正缩进
    try: # 修正缩进
        model = YOLO(model_version) # 修正缩进 (在try块内)
        logger.info(f"Loaded YOLO model: {model_version}") # 修正缩进 (在try块内)
    except Exception as e: # 修正缩进
        logger.error(f"Failed to load YOLO model {model_version}: {e}") # 修正缩进 (在except块内)
        writer.close()  # Ensure writer is closed on error
        return

    # --- 训练参数 ---
    # 设置device参数，优先使用命令行参数
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu' # 修正缩进
    if args.use_gpu and not torch.cuda.is_available(): # 修正缩进
        logger.warning("GPU requested but not available, falling back to CPU") # 修正缩进 (在if块内)
    
    train_params = { # 修正缩进
        'data': data_yaml_path,
        'epochs': config['train']['epochs'],
        'batch': config['train']['batch_size'],
        'imgsz': config['dataset'].get('input_size', 640),
        'optimizer': config['train']['optimizer']['name'],
        'lr0': config['train']['optimizer']['args']['lr'],
        'lrf': config['train']['scheduler']['args'].get('eta_min', 0.01) / config['train']['optimizer']['args']['lr'] if config['train']['scheduler']['name'] == 'CosineAnnealingLR' else 0.1,
        'weight_decay': config['train']['optimizer']['args']['weight_decay'],
        'device': device,
        'workers': config['train']['num_workers'],
        'project': ckpt_dir, 
        'name': 'yolo_pretrain_run', 
        'seed': config['train']['seed'],
        'save_period': -1, 
    }
    logger.info(f"Starting YOLO training with parameters: {train_params}") # 修正缩进
    logger.info(f"Step-based saving interval: {args.save_interval}") # 修正缩进
    logger.info(f"Step-based evaluation interval: {args.eval_interval}") # 修正缩进
    logger.info(f"Evaluation enabled via callback: {args.enable_eval}") # 修正缩进

    # --- 开始训练 ---
    try: # 修正缩进
        global_step = 0 # 修正缩进 (在try块内)

        def on_train_batch_end(trainer): # 修正缩进 (在try块内，def是新的作用域)
            try:
                nonlocal global_step
                global_step += 1

                if trainer.loss is not None:
                    writer.add_scalar("Train/Loss", trainer.loss.item(), global_step)
                else:
                    logger.warning("Trainer loss is None. Skipping logging.")
                if trainer.optimizer is not None:
                    current_lr = trainer.optimizer.param_groups[0]['lr']
                    writer.add_scalar("Train/LearningRate", current_lr, global_step)
                else:
                    logger.warning("Trainer optimizer is None. Skipping learning rate logging.")

                if args.enable_eval and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                    logger.info(f"Step {global_step}: Running evaluation...")
                    trainer.model.eval()
                    val_results = trainer.model.val(data=data_yaml_path) 
                    writer.add_scalar("Validation/mAP50", val_results.box.map50, global_step)
                    writer.add_scalar("Validation/mAP50-95", val_results.box.map, global_step)
                    trainer.model.train()

                if args.save_interval > 0 and global_step % args.save_interval == 0:
                    save_path = os.path.join(ckpt_dir, f"yolo_step{global_step}.pt")
                    trainer.save(save_path) 
                    logger.info(f"Model checkpoint saved at step {global_step}: {save_path}")

            except Exception as e:
                logger.error(f"Error in on_train_batch_end callback at step {global_step}: {e}", exc_info=True)

        model.add_callback("on_train_batch_end", on_train_batch_end) # 修正缩进 (在try块内)

        model.train(**train_params) # 修正缩进 (在try块内)
        logger.info("YOLO training finished.") # 修正缩进 (在try块内)

        final_model_path = os.path.join(ckpt_dir, 'yolo_pretrain_run', 'weights', 'best.pt') # 修正缩进 (在try块内)
        target_model_path = os.path.join(ckpt_dir, "yolo_pretrained_hr.pth") # 修正缩进 (在try块内)
        if os.path.exists(final_model_path): # 修正缩进 (在try块内)
            shutil.copyfile(final_model_path, target_model_path) # 修正缩进 (在if块内)
            logger.info(f"Saved best model to: {target_model_path}") # 修正缩进 (在if块内)
        else: # 修正缩进 (在try块内)
            last_model_path = os.path.join(ckpt_dir, 'yolo_pretrain_run', 'weights', 'last.pt') # 修正缩进 (在else块内)
            if os.path.exists(last_model_path): # 修正缩进 (在else块内)
                shutil.copyfile(last_model_path, target_model_path) # 修正缩进 (在if块内)
                logger.warning(f"Best model not found at {final_model_path}. Saved last model instead to: {target_model_path}") # 修正缩进 (在if块内)
            else: # 修正缩进 (在else块内)
                logger.error(f"Final model not found at {final_model_path} or {last_model_path}.") # 修正缩进 (在else块内)

    except Exception as e: # 修正缩进
        logger.error(f"YOLO training failed: {e}", exc_info=True) # 修正缩进 (在except块内)

    finally: # 修正缩进
        writer.close() # 修正缩进 (在finally块内)

if __name__ == "__main__":
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

    train_yolo(config, args)