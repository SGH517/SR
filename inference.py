import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from models.sr_fast import SRFast
from models.sr_quality import SRQuality
from models.masker import Masker
from models.conditional_sr import ConditionalSR
from models.detector import DetectorWrapper
import argparse

def load_model(weights_path, device, mode="joint"):
    """
    加载模型，根据模式选择加载 SR_Fast、SR_Quality、YOLO 或联合网络。
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    if mode == "sr_fast":
        model = SRFast(scale_factor=4).to(device)
    elif mode == "sr_quality":
        model = SRQuality(scale_factor=4).to(device)
    elif mode == "yolo":
        model = DetectorWrapper(model_path=weights_path, device=device)
    elif mode == "joint":
        # 尝试从 checkpoint 加载配置
        checkpoint = torch.load(weights_path, map_location=device)
        config = checkpoint.get('config', None)

        if config:
            print("Loading model configuration from checkpoint.")
            try:
                sr_fast_args = config.get('model', {}).get('sr_fast', {})
                sr_quality_args = config.get('model', {}).get('sr_quality', {})
                masker_args = config.get('model', {}).get('masker', {})

                sr_fast = SRFast(**sr_fast_args).to(device)
                sr_quality = SRQuality(**sr_quality_args).to(device)
                masker = Masker(**masker_args).to(device)

                model = ConditionalSR(
                    sr_fast=sr_fast,
                    sr_quality=sr_quality,
                    masker=masker,
                    detector_weights="",  # 推理时可以不加载检测器权重
                    sr_fast_weights="",
                    sr_quality_weights="",
                    masker_weights=None,
                    device=device,
                    config=config  # 使用从 checkpoint 加载的配置
                ).to(device)
                print("Model instantiated using config from checkpoint.")
            except Exception as e:
                print(f"Error instantiating model using config from checkpoint: {e}. Falling back to default parameters.")
                model = None

        if model is None:
            # 如果无法从 checkpoint 加载配置，则使用默认参数
            print("Warning: Could not load config from checkpoint. Using default parameters.")
            sr_fast = SRFast(scale_factor=4).to(device)
            sr_quality = SRQuality(scale_factor=4).to(device)
            masker = Masker(in_channels=3, base_channels=32, num_blocks=4).to(device)
            mock_config = {'model': {'masker': {'threshold': 0.5}}, 'train': {}}
            model = ConditionalSR(sr_fast, sr_quality, masker, "", "", "", None, device, mock_config).to(device)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state_dict from {weights_path}")
        else:
            print("Error: Checkpoint does not contain 'model_state_dict'.")
            return None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    model.eval()  # 设置为推理模式
    return model

def preprocess_image(image_path, device):
    """
    预处理输入图像，将其转换为张量。
    """
    image = Image.open(image_path).convert("RGB")
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度
    return image_tensor, image

def postprocess_image(tensor):
    """
    后处理输出图像，将张量转换为 PIL 图像。
    """
    transform = ToPILImage()
    image = transform(tensor.squeeze(0).cpu())  # 移除 batch 维度
    return image

def run_sr_fast(model, image_tensor):
    """
    使用 SR_Fast 网络运行推理。
    """
    with torch.no_grad():
        sr_image = model(image_tensor)
    return sr_image

def run_sr_quality(model, image_tensor):
    """
    使用 SR_Quality 网络运行推理。
    """
    with torch.no_grad():
        sr_image = model(image_tensor)
    return sr_image

def run_yolo(model, image_tensor):
    """
    使用 YOLO 网络运行推理。
    """
    with torch.no_grad():
        results = model(image_tensor)
    detections = [{"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "labels": r.boxes.cls} for r in results]
    return detections

def run_joint(model, image_tensor):
    """
    使用联合网络运行推理。
    """
    with torch.no_grad():
        output = model(image_tensor, hard_mask_inference=True)  # 使用硬掩码
        sr_image = output["sr_image"]
        mask = output["mask_fused"]  # 可视化掩码（可选）
    return sr_image, mask

def process_folder(input_folder, output_folder, model, mode, device):
    """
    遍历输入文件夹中的所有图像文件，运行推理并将结果保存到输出文件夹。
    """
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            # 预处理图像
            image_tensor, original_image = preprocess_image(input_path, device)

            # 根据模式运行推理
            if mode == "sr_fast":
                sr_image_tensor = run_sr_fast(model, image_tensor)
                sr_image = postprocess_image(sr_image_tensor)
                output_path = os.path.join(output_folder, file_name)
                if os.path.exists(output_path):
                    base, ext = os.path.splitext(file_name)
                    output_path = os.path.join(output_folder, f"{base}_output{ext}")
                sr_image.save(output_path)
                print(f"SR_Fast 结果已保存到: {output_path}")

            elif mode == "sr_quality":
                sr_image_tensor = run_sr_quality(model, image_tensor)
                sr_image = postprocess_image(sr_image_tensor)
                output_path = os.path.join(output_folder, file_name)
                if os.path.exists(output_path):
                    base, ext = os.path.splitext(file_name)
                    output_path = os.path.join(output_folder, f"{base}_output{ext}")
                sr_image.save(output_path)
                print(f"SR_Quality 结果已保存到: {output_path}")

            elif mode == "yolo":
                detections = run_yolo(model, image_tensor)
                print(f"YOLO 检测结果 ({file_name}):", detections)

            elif mode == "joint":
                sr_image_tensor, mask_tensor = run_joint(model, image_tensor)
                sr_image = postprocess_image(sr_image_tensor)
                output_path = os.path.join(output_folder, file_name)
                if os.path.exists(output_path):
                    base, ext = os.path.splitext(file_name)
                    output_path = os.path.join(output_folder, f"{base}_output{ext}")
                sr_image.save(output_path)
                print(f"联合网络超分结果已保存到: {output_path}")

                # 可视化掩码
                mask_image = postprocess_image(mask_tensor)
                mask_output_path = os.path.join(output_folder, file_name.replace(".jpg", "_mask.jpg"))
                mask_image.save(mask_output_path)
                print(f"掩码图像已保存到: {mask_output_path}")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Run inference with SR_Fast, SR_Quality, YOLO, or Joint network.")
    parser.add_argument("--mode", type=str, required=True, choices=["sr_fast", "sr_quality", "yolo", "joint"],
                        help="Choose the mode: sr_fast, sr_quality, yolo, or joint.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder to save results.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference if available")
    args = parser.parse_args()

    # 配置
    # 默认使用CPU，除非明确指定--use_gpu参数
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(args.weights, device, mode=args.mode)

    # 处理文件夹中的所有图像
    process_folder(args.input_folder, args.output_folder, model, args.mode, device)
