# evaluate.py
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm # 确保导入 tqdm
import json
import logging # 确保导入 logging

# 从 torchvision 导入
from torchvision import transforms

# 从 utils 导入
from utils.logger import setup_logger, set_logger_level # 导入 set_logger_level
from utils.evaluation_utils import run_coco_evaluation
# 导入新的工具函数
from utils.common_utils import get_device
from utils.model_utils import load_full_checkpoint, load_model_weights

# 从 data 和 models 导入
from data.detection_dataset import DetectionDataset
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR


# 该脚本用于评估联合网络的目标检测性能。

def load_model_for_eval(checkpoint_path: str, device: torch.device, logger_instance: logging.Logger) -> Optional[ConditionalSR]:
    """
    为评估加载 ConditionalSR 模型。
    它首先尝试从检查点加载配置来实例化模型，然后加载模型权重。
    """
    logger_instance.info(f"正在从检查点加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger_instance.error(f"错误: 检查点文件在 {checkpoint_path} 未找到。")
        return None

    checkpoint = load_full_checkpoint(checkpoint_path, device, logger_instance)
    if not checkpoint:
        logger_instance.error(f"无法从 {checkpoint_path} 加载检查点内容。")
        return None

    # --- 尝试从检查点加载配置来实例化模型 ---
    config_from_checkpoint = checkpoint.get('config')
    model_instance: Optional[ConditionalSR] = None

    if config_from_checkpoint:
        logger_instance.info("从检查点加载模型配置。")
        try:
            model_cfg = config_from_checkpoint.get('model', {})
            sr_fast_args = model_cfg.get('sr_fast', {})
            sr_quality_args = model_cfg.get('sr_quality', {})
            masker_args = model_cfg.get('masker', {})
            detector_weights_path = model_cfg.get('weights', {}).get('detector') # 获取检测器权重路径

            # 确保必要的参数存在，否则使用合理的默认值或抛出错误
            if not all([sr_fast_args, sr_quality_args, masker_args]):
                logger_instance.warning("检查点配置中 SRFast, SRQuality 或 Masker 的参数不完整。")
                # 可以选择在这里中断或尝试使用默认参数

            sr_fast = SRFast(**sr_fast_args).to(device)
            sr_quality = SRQuality(**sr_quality_args).to(device)
            masker = Masker(**masker_args).to(device)

            model_instance = ConditionalSR(
                sr_fast=sr_fast,
                sr_quality=sr_quality,
                masker=masker,
                detector_weights=detector_weights_path, # 从配置中获取
                sr_fast_weights=None,  # 权重将通过下面的 load_model_weights 加载
                sr_quality_weights=None,
                masker_weights=None,
                device=str(device),
                config=config_from_checkpoint # 将从检查点加载的完整配置传递下去
            ).to(device)
            logger_instance.info("已使用检查点中的配置成功实例化 ConditionalSR 模型。")
        except KeyError as e_key:
            logger_instance.error(f"使用检查点配置实例化模型时发生 KeyError: {e_key}。可能缺少必要的配置项。")
            model_instance = None
        except Exception as e_instantiate:
            logger_instance.error(f"使用检查点配置实例化模型时发生错误: {e_instantiate}", exc_info=True)
            model_instance = None
    else:
        logger_instance.warning(f"检查点 {checkpoint_path} 中未找到 'config'。")

    if model_instance is None:
        logger_instance.warning("无法使用检查点中的配置实例化模型。将尝试使用默认参数创建模型结构，并加载权重。")
        # 如果无法从配置实例化，创建一个具有默认结构的 ConditionalSR 模型
        # 这假设检查点中的 model_state_dict 与默认结构兼容，或者 load_model_weights(strict=False) 能处理
        try:
            sr_fast_default = SRFast().to(device) # 使用默认参数
            sr_quality_default = SRQuality().to(device)
            masker_default = Masker().to(device)
            # 创建一个最小化的 mock_config，因为 ConditionalSR 的 __init__ 需要它
            mock_config_for_init = {
                'model': {
                    'masker': {'threshold': 0.5}, # ConditionalSR._validate_config 可能需要
                    'weights': {} # 避免在 ConditionalSR 初始化时因缺少 weights 键而警告/错误
                },
                'train': {} # ConditionalSR._validate_config 可能需要
            }
            model_instance = ConditionalSR(
                sr_fast=sr_fast_default,
                sr_quality=sr_quality_default,
                masker=masker_default,
                detector_weights=None, # 权重将从检查点加载
                sr_fast_weights=None,
                sr_quality_weights=None,
                masker_weights=None,
                device=str(device),
                config=mock_config_for_init
            ).to(device)
            logger_instance.info("已创建具有默认参数的 ConditionalSR 模型结构。")
        except Exception as e_default_init:
            logger_instance.error(f"创建具有默认参数的 ConditionalSR 模型失败: {e_default_init}", exc_info=True)
            return None


    # --- 加载模型权重 ---
    # load_model_weights 会处理 'model_state_dict' 或直接的 state_dict
    # 以及 'module.' 前缀
    weights_loaded = load_model_weights(
        model=model_instance,
        weights_path=checkpoint_path, # 直接传递检查点路径，函数内部会解析
        device=device,
        model_name="ConditionalSR (for eval)",
        logger_instance=logger_instance,
        strict=False # 通常评估时允许非严格加载，以防模型结构略有调整
    )

    if not weights_loaded:
        logger_instance.error(f"未能从检查点 {checkpoint_path} 为 ConditionalSR 加载模型权重。")
        return None

    model_instance.eval() # 设置为评估模式
    logger_instance.info(f"模型已成功从 {checkpoint_path} 加载并设置为评估模式。")
    return model_instance


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConditionalSR model for Detection Performance")
    parser.add_argument("--config", type=str, help="Path to the main configuration file (e.g., stage3_joint_finetune.yaml) "
                                                   "This can be used to get dataset paths if not provided directly.")
    parser.add_argument("--lr_dir", type=str,
                        help="Directory containing low-resolution images for evaluation. Overrides config if provided.")
    parser.add_argument("--annotation_file", type=str,
                        help="Path to the ground truth annotation file (JSON format). Overrides config if provided.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the ConditionalSR checkpoint (.pth file) to evaluate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (cuda or cpu).")
    parser.add_argument("--hard_mask", action='store_true',
                        help="Use hard mask during inference (thresholding sigmoid output).")
    parser.add_argument("--output_dir", type=str, default="evaluation_output",
                        help="Directory to save detection results JSON and logs.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    args = parser.parse_args()

    # --- 日志设置 ---
    os.makedirs(args.output_dir, exist_ok=True)
    logger_main = setup_logger(args.output_dir, "evaluation_main.log")
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logger_main.critical(f'无效的日志级别: {args.log_level}')
        raise ValueError(f'无效的日志级别: {args.log_level}')
    set_logger_level(logger_main, numeric_log_level)
    logger_main.info(f"评估脚本参数: {args}")


    # --- 设备选择 ---
    # 命令行 --device 参数优先于 --use_gpu (如果也存在的话)
    # get_device 通常期望一个布尔值 use_gpu_arg
    use_gpu_flag = True if args.device == "cuda" else False
    if args.device == "cpu" and args.use_gpu is True: # 假设 args 可能有 use_gpu (虽然当前没有)
         logger_main.warning("命令行指定 --device cpu 但 --use_gpu 也被设置。将使用 CPU。")
         use_gpu_flag = False
    device = get_device(use_gpu_flag, logger_main)


    # --- 获取数据集路径 ---
    eval_lr_dir = args.lr_dir
    eval_annotation_file = args.annotation_file

    if not eval_lr_dir or not eval_annotation_file:
        if args.config:
            logger_main.info(f"尝试从主配置文件 {args.config} 获取评估数据集路径。")
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    main_config = yaml.safe_load(f)
                eval_config_from_yaml = main_config.get('evaluation', {})
                if not eval_lr_dir:
                    eval_lr_dir = eval_config_from_yaml.get('val_image_dir')
                if not eval_annotation_file:
                    eval_annotation_file = eval_config_from_yaml.get('val_annotation_file')
                logger_main.info(f"从配置文件中获取到 LR 目录: {eval_lr_dir}, 标注文件: {eval_annotation_file}")
            except Exception as e_cfg:
                logger_main.error(f"加载主配置文件 {args.config} 或提取评估路径时出错: {e_cfg}")
        else:
            logger_main.error("未直接提供评估数据集路径，也未提供主配置文件以供查找。")
            return

    if not eval_lr_dir or not os.path.exists(eval_lr_dir):
        logger_main.error(f"评估用 LR 图像目录无效或未找到: {eval_lr_dir}")
        return
    if not eval_annotation_file or not os.path.exists(eval_annotation_file):
        logger_main.error(f"评估用标注文件无效或未找到: {eval_annotation_file}")
        return


    # --- 加载模型 ---
    model = load_model_for_eval(args.checkpoint, device, logger_main)
    if model is None:
        logger_main.error("模型加载失败。正在退出评估。")
        return

    # --- 数据加载器 ---
    try:
        # 评估时通常使用 ToTensor，归一化等应在模型内部或加载权重时考虑
        eval_transform = transforms.ToTensor()
        eval_dataset = DetectionDataset(
            image_dir=eval_lr_dir, # 直接使用LR图像目录
            annotation_file=eval_annotation_file,
            transform=eval_transform,
            return_image_id=True # 对于COCO评估是必需的
        )
        # 使用 DetectionDataset 的 collate_fn 来处理可能的 None 项
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers= main_config.get('train',{}).get('num_workers', 2) if 'main_config' in locals() else 2, #尝试从config获取
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=DetectionDataset.collate_fn
        )
        logger_main.info(f"已从 {eval_lr_dir} 加载评估数据集，共 {len(eval_dataset)} 张图像。")
    except FileNotFoundError as e_data:
        logger_main.error(f"加载评估数据集时发生错误 (文件未找到): {e_data}")
        return
    except Exception as e_data_other:
        logger_main.error(f"加载评估数据集时发生未知错误: {e_data_other}", exc_info=True)
        return

    # --- 运行评估 ---
    logger_main.info("开始在评估数据集上运行 COCO 评估...")
    # 确定文件名的时间戳或标识符
    checkpoint_basename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    eval_step_or_epoch_name = f"ckpt_{checkpoint_basename}"


    map_results_output, avg_quality_path_usage_output = run_coco_evaluation(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        annotation_file=eval_annotation_file, # 使用已确定的 GT 标注文件
        output_dir=args.output_dir,           # 保存检测结果 JSON 的目录
        step_or_epoch=eval_step_or_epoch_name, # 用于结果文件名
        logger=logger_main,
        use_hard_mask=args.hard_mask          # 推理时是否使用硬掩码
    )

    logger_main.info("--- 评估完成 ---")
    logger_main.info(f"Quality 路径平均使用率 (1-稀疏度): {avg_quality_path_usage_output:.4f}")
    if map_results_output:
        logger_main.info("检测 mAP 结果:")
        logger_main.info(f"  mAP@0.50:0.95 = {map_results_output.get('map', 0.0):.4f}")
        logger_main.info(f"  mAP@0.50      = {map_results_output.get('map_50', 0.0):.4f}")
        logger_main.info(f"  mAP@0.75      = {map_results_output.get('map_75', 0.0):.4f}")

        # 将评估结果也保存到一个 JSON 文件中
        eval_summary_path = os.path.join(args.output_dir, f"evaluation_summary_{eval_step_or_epoch_name}.json")
        summary_data = {
            "checkpoint_evaluated": args.checkpoint,
            "mAP_0.50_0.95": map_results_output.get('map', 0.0),
            "mAP_0.50": map_results_output.get('map_50', 0.0),
            "mAP_0.75": map_results_output.get('map_75', 0.0),
            "quality_path_usage_ratio": avg_quality_path_usage_output,
            "hard_mask_used": args.hard_mask,
            "dataset_lr_dir": eval_lr_dir,
            "dataset_annotation_file": eval_annotation_file
        }
        try:
            with open(eval_summary_path, 'w') as f_summary:
                json.dump(summary_data, f_summary, indent=4)
            logger_main.info(f"评估摘要已保存到: {eval_summary_path}")
        except Exception as e_save_summary:
            logger_main.error(f"保存评估摘要时出错: {e_save_summary}")

    else:
        logger_main.warning("mAP 结果不可用。")


if __name__ == "__main__":
    main()