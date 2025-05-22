import os
from PIL import Image
import argparse

# 该脚本用于准备超分辨率训练数据，包括生成低分辨率图像。

def prepare_sr_data(input_dir, output_dir, scale_factor):
    """
    生成 LR-HR 图像对。

    参数：
        input_dir (str): 高分辨率图像的输入目录。
        output_dir (str): 输出目录，包含生成的 LR 和 HR 图像。
        scale_factor (int): 下采样因子，例如 4 表示 x4 下采样。
    """
    hr_dir = os.path.join(output_dir, "HR")
    lr_dir = os.path.join(output_dir, "LR")

    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            with Image.open(img_path) as img:
                # 确保图像是 RGB 格式
                img = img.convert("RGB")

                # 保存 HR 图像
                hr_path = os.path.join(hr_dir, img_name)
                img.save(hr_path)

                # 生成并保存 LR 图像
                lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)
                lr_path = os.path.join(lr_dir, img_name)
                lr_img.save(lr_path)

                print(f"Processed {img_name}: HR -> {hr_path}, LR -> {lr_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

def validate_input_directory(input_dir):
    """
    验证输入目录是否存在且包含有效图像文件。

    参数：
        input_dir (str): 输入目录路径。
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(input_dir)):
        raise ValueError(f"No valid image files found in directory: {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LR-HR image pairs for super-resolution.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing high-resolution images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated LR and HR images.")
    parser.add_argument("--scale_factor", type=int, default=4, help="Downscaling factor (e.g., 4 for x4).")

    args = parser.parse_args()

    validate_input_directory(args.input_dir)
    prepare_sr_data(args.input_dir, args.output_dir, args.scale_factor)