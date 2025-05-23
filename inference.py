# inference.py
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage # 确保导入 ToPILImage
import argparse
import logging # 导入 logging

# 从 models 导入
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from models.detector import DetectorWrapper # DetectorWrapper 用于 'yolo' 模式

# 从 utils 导入
from utils.common_utils import get_device # 用于获取设备
from utils.model_utils import load_full_checkpoint, load_model_weights # 用于加载模型

# 设置一个简单的 logger
logger_inf = logging.getLogger(__name__)
if not logger_inf.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


def load_model(weights_path: str, device: torch.device, mode: str = "joint", logger_instance: logging.Logger = logger_inf):
    """
    加载模型，根据模式选择加载 SR_Fast、SR_Quality、YOLO 或联合网络 ConditionalSR。
    """
    if not os.path.exists(weights_path):
        logger_instance.error(f"模型权重文件在 {weights_path} 未找到。")
        raise FileNotFoundError(f"模型权重文件在 {weights_path} 未找到。")

    model: Optional[torch.nn.Module] = None
    checkpoint: Optional[dict] = None

    if mode == "joint":
        checkpoint = load_full_checkpoint(weights_path, device, logger_instance)
        if not checkpoint:
            logger_instance.error(f"无法从 {weights_path} 加载 'joint' 模式的检查点。")
            return None

        config_from_checkpoint = checkpoint.get('config')
        if config_from_checkpoint:
            logger_instance.info("从检查点加载 'joint' 模型的配置。")
            try:
                model_cfg = config_from_checkpoint.get('model', {})
                sr_fast_args = model_cfg.get('sr_fast', {})
                sr_quality_args = model_cfg.get('sr_quality', {})
                masker_args = model_cfg.get('masker', {})
                # 推理时，detector_weights 可以为空字符串或 None，因为权重已在 ConditionalSR 的 state_dict 中
                # 但如果 ConditionalSR 内部依赖 config 中的路径来初始化 DetectorWrapper，则需要传递
                detector_weights_cfg = model_cfg.get('weights', {}).get('detector')


                sr_fast_instance = SRFast(**sr_fast_args).to(device)
                sr_quality_instance = SRQuality(**sr_quality_args).to(device)
                masker_instance = Masker(**masker_args).to(device)

                model = ConditionalSR(
                    sr_fast=sr_fast_instance,
                    sr_quality=sr_quality_instance,
                    masker=masker_instance,
                    detector_weights=detector_weights_cfg, # 传递配置中的路径
                    sr_fast_weights=None,  # 权重将从主检查点加载
                    sr_quality_weights=None,
                    masker_weights=None,
                    device=str(device),
                    config=config_from_checkpoint
                ).to(device)
                logger_instance.info("已使用检查点中的配置实例化 'joint' (ConditionalSR) 模型。")
            except Exception as e_inst:
                logger_instance.error(f"使用检查点配置实例化 'joint' 模型时发生错误: {e_inst}。将尝试默认结构。", exc_info=True)
                model = None # 发生错误则重置

        if model is None: # 如果从配置实例化失败或无配置
            logger_instance.warning("无法从检查点配置实例化 'joint' 模型或配置缺失。将尝试使用默认参数创建模型结构。")
            sr_fast_default = SRFast().to(device)
            sr_quality_default = SRQuality().to(device)
            masker_default = Masker().to(device)
            # 提供一个最小化的 mock_config
            mock_cfg = {'model': {'masker': {'threshold': 0.5}, 'weights':{}}, 'train': {}}
            model = ConditionalSR(sr_fast_default, sr_quality_default, masker_default,
                                  None, None, None, None, str(device), mock_cfg).to(device)

        # 加载模型权重 (state_dict)
        weights_loaded = load_model_weights(model, weights_path, device, f"ConditionalSR ({mode})", logger_instance, strict=False)
        if not weights_loaded:
            logger_instance.error(f"未能从 {weights_path} 为 'joint' (ConditionalSR) 模型加载权重。")
            return None

    elif mode == "sr_fast":
        # 假设 SRFast 检查点直接是 state_dict 或包含 model_state_dict
        model = SRFast(scale_factor=4).to(device) # 使用默认参数或从配置推断
        load_model_weights(model, weights_path, device, "SR_Fast", logger_instance, strict=False)
    elif mode == "sr_quality":
        model = SRQuality(scale_factor=4).to(device) # 使用默认参数或从配置推断
        load_model_weights(model, weights_path, device, "SR_Quality", logger_instance, strict=False)
    elif mode == "yolo":
        # DetectorWrapper 构造函数会加载 YOLO 模型
        model = DetectorWrapper(model_path=weights_path, device=str(device))
        if not model.model and not model.yolo_model_module: # 检查是否成功加载
            logger_instance.error(f"加载 YOLO 模型 ({weights_path}) 失败。DetectorWrapper 未能初始化模型。")
            return None
    else:
        logger_instance.error(f"不支持的推理模式: {mode}")
        raise ValueError(f"不支持的推理模式: {mode}")

    if model:
        model.eval()  # 设置为推理模式
        logger_instance.info(f"模型 ({mode}) 已成功加载并设置为评估模式。")
    return model


def preprocess_image(image_path: str, device: torch.device) -> Tuple[torch.Tensor, Image.Image]:
    """
    预处理输入图像，将其转换为张量。
    返回张量和原始 PIL 图像。
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger_inf.error(f"图像文件未找到: {image_path}")
        raise
    except Exception as e:
        logger_inf.error(f"打开或转换图像 {image_path} 时出错: {e}")
        raise

    transform = ToTensor() # 将 PIL 图像 (H, W, C) 或 (H, W) 转换为 (C, H, W) 张量，并缩放到 [0.0, 1.0]
    image_tensor = transform(pil_image).unsqueeze(0).to(device)  # 添加 batch 维度并移至设备
    return image_tensor, pil_image

def postprocess_sr_image(tensor: torch.Tensor) -> Image.Image:
    """
    后处理超分辨率输出图像，将张量转换为 PIL 图像。
    """
    if tensor.ndim == 4 and tensor.size(0) == 1: # 移除批次维度 (B, C, H, W) -> (C, H, W)
        tensor_squeezed = tensor.squeeze(0)
    elif tensor.ndim == 3: # (C, H, W)
        tensor_squeezed = tensor
    else:
        logger_inf.error(f"后处理SR图像时遇到非预期的张量维度: {tensor.shape}")
        # 根据情况返回默认图像或引发错误
        return Image.new('RGB', (64,64), color = 'red')


    # 确保张量在 CPU 上且为浮点类型，并反归一化（如果需要）
    # ToPILImage 期望 [0,1] 范围的浮点张量或 [0,255] 范围的整数张量
    # ToTensor 已经将 PIL 图像转为 [0,1] 范围的浮点张量
    image_pil = ToPILImage()(tensor_squeezed.cpu())
    return image_pil

def run_sr_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用 SR_Fast 或 SR_Quality 网络运行推理。
    """
    with torch.no_grad():
        sr_image_tensor = model(image_tensor)
    return sr_image_tensor

def run_yolo_inference(model: DetectorWrapper, image_tensor: torch.Tensor, original_pil_image: Image.Image) -> List[Dict[str, Any]]:
    """
    使用 YOLO 网络运行推理。 DetectorWrapper 的 forward 在推理时直接返回检测结果列表。
    original_pil_image 用于可能的绘图（如果需要）。
    """
    with torch.no_grad():
        # DetectorWrapper.forward 在 eval 模式下返回 (detections_list, None)
        # detections_list 是一个列表，每个元素是 {'boxes': tensor, 'scores': tensor, 'labels': tensor}
        detections_list, _ = model(image_tensor) # targets 为 None

    # detections_list 已经是处理好的结果
    # 如果需要进一步处理或绘图，可以在这里进行
    # 例如，将归一化坐标转回绝对坐标（如果YOLO输出的是归一化坐标且需要绝对坐标）
    # 但 DetectorWrapper 现在返回的是 xyxy 绝对（相对于输入给它的图像）坐标
    formatted_detections = []
    if detections_list: # detections_list 是批次的列表，这里假设批次大小为1
        img_detections = detections_list[0] # 取出该图像的检测结果字典
        formatted_detections.append({
            "boxes": img_detections["boxes"].cpu().numpy().tolist(), # [[x1,y1,x2,y2], ...]
            "scores": img_detections["scores"].cpu().numpy().tolist(), # [score1, score2, ...]
            "labels": img_detections["labels"].cpu().numpy().tolist(), # [label1, label2, ...]
        })
    return formatted_detections


def run_joint_inference(model: ConditionalSR, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    使用联合网络 ConditionalSR 运行推理。
    """
    with torch.no_grad():
        # hard_mask_inference=True 是 ConditionalSR 的参数
        # 假设在 config 中有 model.masker.threshold
        output_dict = model(image_tensor, hard_mask_inference=True)
        sr_image_tensor = output_dict["sr_image"]
        mask_fused_tensor = output_dict["mask_fused"] # (B, 1, H_sr, W_sr)
    return sr_image_tensor, mask_fused_tensor


def process_single_image(input_path: str, output_folder: str,
                         model: torch.nn.Module, mode: str, device: torch.device,
                         logger_instance: logging.Logger):
    """处理单张图像文件，运行推理并将结果保存到输出文件夹。"""
    file_name = os.path.basename(input_path)
    base_name, ext = os.path.splitext(file_name)
    
    try:
        logger_instance.info(f"正在处理图像: {input_path} (模式: {mode})")
        image_tensor, original_pil_image = preprocess_image(input_path, device)

        if mode == "sr_fast" or mode == "sr_quality":
            sr_image_tensor = run_sr_inference(model, image_tensor)
            sr_pil_image = postprocess_sr_image(sr_image_tensor)
            output_file_path = os.path.join(output_folder, f"{base_name}_sr_{mode}{ext}")
            sr_pil_image.save(output_file_path)
            logger_instance.info(f"{mode} 超分结果已保存到: {output_file_path}")

        elif mode == "yolo":
            # DetectorWrapper 的 forward 返回的是处理好的检测结果列表
            detections_output = run_yolo_inference(model, image_tensor, original_pil_image) # model 是 DetectorWrapper
            output_json_path = os.path.join(output_folder, f"{base_name}_yolo_detections.json")
            with open(output_json_path, 'w') as f:
                json.dump(detections_output, f, indent=4)
            logger_instance.info(f"YOLO 检测结果已保存到: {output_json_path}")
            # 可选：在此处添加绘制检测框到图像并保存的逻辑

        elif mode == "joint":
            sr_image_tensor, mask_fused_tensor = run_joint_inference(model, image_tensor) # model 是 ConditionalSR
            sr_pil_image = postprocess_sr_image(sr_image_tensor)
            output_sr_file_path = os.path.join(output_folder, f"{base_name}_sr_joint{ext}")
            sr_pil_image.save(output_sr_file_path)
            logger_instance.info(f"Joint (ConditionalSR) 超分结果已保存到: {output_sr_file_path}")

            if mask_fused_tensor is not None:
                mask_pil_image = postprocess_sr_image(mask_fused_tensor) # 同样使用 postprocess_sr_image 处理单通道掩码
                output_mask_file_path = os.path.join(output_folder, f"{base_name}_mask_joint{ext}")
                mask_pil_image.save(output_mask_file_path)
                logger_instance.info(f"Joint (ConditionalSR) 融合掩码已保存到: {output_mask_file_path}")
            
            # 如果 ConditionalSR 内部有检测器，并且配置为在 joint 模式下也输出检测结果
            if isinstance(model, ConditionalSR) and model.detector is not None:
                # ConditionalSR 的 forward 返回的 yolo_raw_predictions 在推理时是格式化结果
                # 这里需要再次调用 detector，或者修改 ConditionalSR 的 forward 使其也返回检测结果
                # 当前 ConditionalSR 的 forward 返回了 "yolo_raw_predictions"
                # 我们需要确保它在推理模式下是 DetectorWrapper 返回的格式化检测列表
                output_dict = model(image_tensor, hard_mask_inference=True) # 再次调用或从上次结果获取
                detections_from_joint = output_dict.get("yolo_raw_predictions")
                
                if detections_from_joint:
                    # detections_from_joint 已经是 DetectorWrapper 返回的格式化列表
                    output_joint_det_json_path = os.path.join(output_folder, f"{base_name}_joint_detections.json")
                    # detections_from_joint 是一个批次的列表，取第一个元素（假设批次大小为1）
                    # 并将其转换为可序列化的格式
                    serializable_detections = []
                    if detections_from_joint and isinstance(detections_from_joint, list) and len(detections_from_joint)>0:
                        img_dets = detections_from_joint[0]
                        serializable_detections.append({
                            "boxes": img_dets["boxes"].cpu().numpy().tolist() if torch.is_tensor(img_dets["boxes"]) else img_dets["boxes"],
                            "scores": img_dets["scores"].cpu().numpy().tolist() if torch.is_tensor(img_dets["scores"]) else img_dets["scores"],
                            "labels": img_dets["labels"].cpu().numpy().tolist() if torch.is_tensor(img_dets["labels"]) else img_dets["labels"],
                        })

                    with open(output_joint_det_json_path, 'w') as f_joint_det:
                        json.dump(serializable_detections, f_joint_det, indent=4)
                    logger_instance.info(f"Joint (ConditionalSR) 检测结果已保存到: {output_joint_det_json_path}")


    except Exception as e_proc:
        logger_instance.error(f"处理文件 {file_name} 时发生错误: {e_proc}", exc_info=True)


def process_folder(input_folder_path: str, output_folder_path: str,
                   model_instance: torch.nn.Module, inference_mode: str,
                   device_obj: torch.device, logger_instance: logging.Logger):
    """
    遍历输入文件夹中的所有图像文件，运行推理并将结果保存到输出文件夹。
    """
    if not os.path.isdir(input_folder_path):
        logger_instance.error(f"输入文件夹路径无效: {input_folder_path}")
        return
    os.makedirs(output_folder_path, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        logger_instance.warning(f"在输入文件夹 {input_folder_path} 中未找到支持的图像文件。")
        return

    for img_file_name in tqdm(image_files, desc=f"处理 {inference_mode} 推理"):
        img_full_path = os.path.join(input_folder_path, img_file_name)
        process_single_image(img_full_path, output_folder_path, model_instance, inference_mode, device_obj, logger_instance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 SR_Fast, SR_Quality, YOLO, 或 Joint (ConditionalSR) 网络进行推理。")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["sr_fast", "sr_quality", "yolo", "joint"],
                        help="选择推理模式: sr_fast, sr_quality, yolo, 或 joint (ConditionalSR)。")
    parser.add_argument("--weights", type=str, required=True, help="模型权重文件的路径。")
    parser.add_argument("--input", type=str, required=True,
                        help="输入图像文件或包含图像的文件夹的路径。")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="保存推理结果的输出文件夹路径。")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="选择运行设备 (cuda 或 cpu)。")
    # ConditionalSR 的 hard_mask 参数可以保留，但对于其他模式可能不需要
    parser.add_argument("--hard_mask", action='store_true',
                        help="在 'joint' (ConditionalSR) 推理模式下使用硬掩码。")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别。")
    args_cli = parser.parse_args()

    # --- 日志设置 ---
    # 主脚本的 logger
    main_logger = logging.getLogger("inference_script")
    if not main_logger.hasHandlers():
        handler = logging.StreamHandler() # 输出到控制台
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        main_logger.addHandler(handler)
    try:
        main_logger.setLevel(args_cli.log_level.upper())
    except ValueError:
        main_logger.setLevel(logging.INFO)
        main_logger.warning(f"无效的日志级别 '{args_cli.log_level}'。使用 INFO 级别。")

    main_logger.info(f"推理脚本参数: {args_cli}")


    # --- 设备选择 ---
    use_gpu = True if args_cli.device == "cuda" else False
    selected_device = get_device(use_gpu, main_logger) # 使用 common_utils 获取设备

    # --- 加载模型 ---
    try:
        loaded_model = load_model(args_cli.weights, selected_device, mode=args_cli.mode, logger_instance=main_logger)
    except Exception as e_load:
        main_logger.error(f"加载模型时发生致命错误: {e_load}", exc_info=True)
        exit(1)

    if loaded_model is None:
        main_logger.error("模型未能加载。退出推理。")
        exit(1)

    # --- 处理输入 (单个文件或文件夹) ---
    os.makedirs(args_cli.output_folder, exist_ok=True) # 确保输出文件夹存在

    if os.path.isfile(args_cli.input):
        process_single_image(args_cli.input, args_cli.output_folder, loaded_model, args_cli.mode, selected_device, main_logger)
    elif os.path.isdir(args_cli.input):
        process_folder(args_cli.input, args_cli.output_folder, loaded_model, args_cli.mode, selected_device, main_logger)
    else:
        main_logger.error(f"输入路径无效: {args_cli.input}。它既不是文件也不是文件夹。")
        exit(1)

    main_logger.info(f"推理完成。结果已保存到: {args_cli.output_folder}")