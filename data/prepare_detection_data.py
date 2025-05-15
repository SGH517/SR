import os
import json
from PIL import Image
import argparse

def prepare_detection_data(input_dir, annotation_file, output_dir, scale_factor):
    """
    对数据集图像进行下采样，生成 LR 图像，并调整标注文件中的 BBox 坐标。

    参数:
        input_dir (str): 高分辨率图像的输入目录。
        annotation_file (str): 标注文件路径 (JSON 格式)。
        output_dir (str): 输出目录，包含生成的 LR 图像和调整后的标注文件。
        scale_factor (int): 下采样因子，例如 4 表示 x4 下采样。
    """
    lr_dir = os.path.join(output_dir, "LR")
    os.makedirs(lr_dir, exist_ok=True)

    # 加载标注文件
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    updated_annotations = {"images": [], "annotations": []}

    for img_info in annotations["images"]:
        img_name = img_info["file_name"]
        img_path = os.path.join(input_dir, img_name)
        try:
            with Image.open(img_path) as img:
                # 确保图像是 RGB 格式
                img = img.convert("RGB")

                # 生成并保存 LR 图像
                lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)
                lr_path = os.path.join(lr_dir, img_name)
                lr_img.save(lr_path)

                # 更新图像信息
                updated_img_info = img_info.copy()
                updated_img_info["width"] = lr_img.width
                updated_img_info["height"] = lr_img.height
                updated_img_info["file_name"] = os.path.join("LR", img_name)
                updated_annotations["images"].append(updated_img_info)

                print(f"Processed {img_name}: LR -> {lr_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            # 可选：记录到日志文件
            continue

    # 更新标注信息
    for ann in annotations["annotations"]:
        updated_ann = ann.copy()
        bbox = ann["bbox"]
        x_min, y_min, width, height = bbox
        updated_ann["bbox"] = [
            x_min / scale_factor,
            y_min / scale_factor,
            width / scale_factor,
            height / scale_factor
        ]
        updated_annotations["annotations"].append(updated_ann)

    # 保存更新后的标注文件
    updated_annotation_file = os.path.join(output_dir, "annotations.json")
    with open(updated_annotation_file, 'w') as f:
        json.dump(updated_annotations, f, indent=4)
    print(f"Updated annotations saved to {updated_annotation_file}")

def validate_annotation_file(annotation_file):
    """
    验证标注文件是否存在且格式正确。

    参数：
        annotation_file (str): 标注文件路径。
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        if not all(key in data for key in ["images", "annotations"]):
            raise ValueError("Annotation file is missing required keys: 'images' or 'annotations'")
    except json.JSONDecodeError:
        raise ValueError(f"Annotation file is not a valid JSON: {annotation_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LR images and adjust annotations for detection.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing high-resolution images.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation file (JSON format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated LR images and annotations.")
    parser.add_argument("--scale_factor", type=int, default=4, help="Downscaling factor (e.g., 4 for x4).")

    args = parser.parse_args()

    validate_annotation_file(args.annotation_file)
    prepare_detection_data(args.input_dir, args.annotation_file, args.output_dir, args.scale_factor)
