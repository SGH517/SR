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
    train_annotation_file = dataset_config.get('train_annotation_file')
    val_annotation_file = dataset_config.get('val_annotation_file')

    if not train_image_dir or not os.path.exists(train_image_dir):
        logger.error(f"Training image directory not found: {train_image_dir}")
        return # Exit if critical path is missing
    if not train_annotation_file or not os.path.exists(train_annotation_file):
        logger.error(f"Training annotation file not found: {train_annotation_file}")
        return # Exit if critical path is missing
    if args.enable_eval: # Only check validation paths if evaluation is enabled
        if not val_image_dir or not os.path.exists(val_image_dir):
            logger.warning(f"Validation image directory not found: {val_image_dir}. Evaluation might fail.")
        if not val_annotation_file or not os.path.exists(val_annotation_file):
            logger.warning(f"Validation annotation file not found: {val_annotation_file}. Evaluation might fail.")

    logger.info("--- Configuration Validated ---")
    # --- End Configuration Validation ---

    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    # --- 数据集准备 (保持不变) ---
    # 默认使用COCO数据集类别
    class_names = ['block','bolt']
    num_classes = 2  # COCO数据集类别数
    
    # 强制使用绝对路径并处理中文问题
    train_path = os.path.abspath(config['dataset']['train_image_dir']).replace('\\', '/')
    val_path = os.path.abspath(config['dataset']['val_image_dir']).replace('\\', '/')
    
    data_yaml_path = os.path.abspath("dataset/Data partitioning/data.yaml").replace('\\','/')

    #     data_yaml_content = f"""
    # train: {train_path}
    # val: {val_path}
    # nc: {num_classes}
    # names: {class_names}
    # """
    #     data_yaml_path = os.path.join(log_dir, "data_stage1.yaml")
    #     with open(data_yaml_path, 'w') as f:
    #         f.write(data_yaml_content)
    #     logger.info(f"Generated data YAML for YOLO: {data_yaml_path}")
    #     logger.info(f"Ensure class names and nc in {data_yaml_path} are correct for your dataset!")

        # --- 模型加载 (保持不变) ---
        model_version = config['model']['version']
        try:
            model = YOLO(model_version)
            logger.info(f"Loaded YOLO model: {model_version}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_version}: {e}")
            writer.close()  # Ensure writer is closed on error
            return

        # --- 训练参数 ---
        # 设置device参数，优先使用命令行参数
        device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
        if args.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available, falling back to CPU")
        
        train_params = {
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
            'project': ckpt_dir, # YOLO will create a subdir here
            'name': 'yolo_pretrain_run', # Subdir name within project dir
            'seed': config['train']['seed'],
            # --- 禁用内置的基于轮数的保存 ---
            'save_period': -1, # <--- IMPORTANT: Disable epoch-based saving
            # --- 禁用内置的基于轮数的评估 (如果使用回调评估) ---
            # 'val': False, # Or rely on callback eval
            # 其他 YOLO 支持的参数...
        }
        logger.info(f"Starting YOLO training with parameters: {train_params}")
        logger.info(f"Step-based saving interval: {args.save_interval}")
        logger.info(f"Step-based evaluation interval: {args.eval_interval}")
        logger.info(f"Evaluation enabled via callback: {args.enable_eval}")

        # --- 开始训练 ---
        try:
            global_step = 0

            def on_train_batch_end(trainer):
                try:
                    nonlocal global_step
                    global_step += 1

                    # 记录损失和学习率
                    if trainer.loss is not None:
                        writer.add_scalar("Train/Loss", trainer.loss.item(), global_step)
                    else:
                        logger.warning("Trainer loss is None. Skipping logging.")
                    if trainer.optimizer is not None:
                        current_lr = trainer.optimizer.param_groups[0]['lr']
                        writer.add_scalar("Train/LearningRate", current_lr, global_step)
                    else:
                        logger.warning("Trainer optimizer is None. Skipping learning rate logging.")

                    # 每 eval_interval 步评估 (保持不变)
                    if args.enable_eval and args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        logger.info(f"Step {global_step}: Running evaluation...")
                        # 评估逻辑（可以调用 YOLO 的验证方法）
                        # Ensure model is in eval mode for validation
                        trainer.model.eval()
                        val_results = trainer.model.val(data=data_yaml_path)  # Pass data explicitly if needed
                        writer.add_scalar("Validation/mAP50", val_results.box.map50, global_step)
                        writer.add_scalar("Validation/mAP50-95", val_results.box.map, global_step)
                        # Set back to train mode
                        trainer.model.train()

                    # --- 基于步数保存模型 ---
                    if args.save_interval > 0 and global_step % args.save_interval == 0:
                        save_path = os.path.join(ckpt_dir, f"yolo_step{global_step}.pt")
                        # 使用 trainer.save() 或 trainer.model.save()
                        trainer.save(save_path)  # trainer.save might save more state
                        # 或者 trainer.model.save(save_path) # 只保存模型权重
                        logger.info(f"Model checkpoint saved at step {global_step}: {save_path}")

                except Exception as e:
                    logger.error(f"Error in on_train_batch_end callback at step {global_step}: {e}", exc_info=True)

            # 绑定回调
            model.add_callback("on_train_batch_end", on_train_batch_end)

            model.train(**train_params)
            logger.info("YOLO training finished.")

            # --- 保存最终最佳模型 (保持不变) ---
            final_model_path = os.path.join(ckpt_dir, 'yolo_pretrain_run', 'weights', 'best.pt')
            target_model_path = os.path.join(ckpt_dir, "yolo_pretrained_hr.pth")
            if os.path.exists(final_model_path):
                shutil.copyfile(final_model_path, target_model_path)
                logger.info(f"Saved best model to: {target_model_path}")
            else:
                # Try saving the last model state if best.pt doesn't exist
                last_model_path = os.path.join(ckpt_dir, 'yolo_pretrain_run', 'weights', 'last.pt')
                if os.path.exists(last_model_path):
                    shutil.copyfile(last_model_path, target_model_path)
                    logger.warning(f"Best model not found at {final_model_path}. Saved last model instead to: {target_model_path}")
                else:
                    logger.error(f"Final model not found at {final_model_path} or {last_model_path}.")

        except Exception as e:
            logger.error(f"YOLO training failed: {e}", exc_info=True)

        finally:
            writer.close()

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
