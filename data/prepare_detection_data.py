import os
import json
from PIL import Image
import argparse
import logging # 使用 logging 模块

# 设置一个简单的 logger (或者你可以从 utils.logger 导入)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def prepare_detection_data(input_dir_hr, annotation_file_hr, output_dir_lr, scale_factor):
    """
    对数据集图像进行下采样，生成 LR 图像，并调整标注文件中的 BBox 坐标。
    同时复制 'categories' 字段。

    参数:
        input_dir_hr (str): 高分辨率图像的输入目录。
        annotation_file_hr (str): 原始高分辨率图像的标注文件路径 (JSON 格式)。
        output_dir_lr (str): 输出目录，将包含生成的 LR 图像和调整后的标注文件。
        scale_factor (int): 下采样因子，例如 4 表示 x4 下采样。
    """
    lr_images_subdir = os.path.join(output_dir_lr, "LR") # LR图像的子目录
    os.makedirs(lr_images_subdir, exist_ok=True)

    # 加载原始标注文件
    try:
        with open(annotation_file_hr, 'r', encoding='utf-8') as f:
            original_annotations = json.load(f)
    except FileNotFoundError:
        logger.error(f"Original annotation file not found: {annotation_file_hr}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from original annotation file: {annotation_file_hr}")
        return

    updated_annotations = {"images": [], "annotations": []}

    # 复制 'categories' 字段 (如果存在)
    if 'categories' in original_annotations:
        updated_annotations['categories'] = original_annotations['categories']
        logger.info("Copied 'categories' field from original annotations.")
    else:
        logger.warning(f"'categories' key not found in the original annotation file: {annotation_file_hr}. "
                       "The output annotation file will be missing it, which might cause issues for COCO evaluation.")
        updated_annotations['categories'] = [] # 或者你可以根据需要提供一个默认的 categories 列表

    # 处理图像并更新图像信息
    for img_info in original_annotations.get("images", []):
        img_name = img_info.get("file_name", "")
        if not img_name:
            logger.warning(f"Missing 'file_name' in image info: {img_info}. Skipping.")
            continue

        # 原始HR图像的完整路径
        hr_img_path = os.path.join(input_dir_hr, img_name)
        
        # 生成的LR图像的相对路径 (相对于 output_dir_lr 下的 'LR' 子目录)
        # 和完整保存路径
        lr_img_relative_path = img_name # 将直接保存在 LR 子目录下
        lr_img_save_path = os.path.join(lr_images_subdir, lr_img_relative_path)
        
        # 确保LR图像的子目录存在 (如果图像名包含子路径)
        os.makedirs(os.path.dirname(lr_img_save_path), exist_ok=True)

        try:
            with Image.open(hr_img_path) as img:
                img = img.convert("RGB") # 确保是 RGB 格式

                # 生成并保存 LR 图像
                lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)
                lr_img.save(lr_img_save_path)

                # 更新图像信息
                updated_img_info = img_info.copy()
                updated_img_info["width"] = lr_img.width
                updated_img_info["height"] = lr_img.height
                # file_name 在 COCO json 中通常是相对于图像根目录的路径
                # 如果你的 LR 图像都放在 output_dir_lr/LR/ 下，那么 file_name 应该是 "LR/original_img_name.jpg"
                updated_img_info["file_name"] = os.path.join("LR", lr_img_relative_path).replace("\\", "/") # 使用正斜杠
                updated_annotations["images"].append(updated_img_info)

                logger.info(f"Processed {img_name}: HR -> {hr_img_path}, LR -> {lr_img_save_path}")
        except FileNotFoundError:
            logger.warning(f"Image file not found: {hr_img_path} (original name: {img_name}). Skipping this image.")
            continue
        except Exception as e:
            logger.warning(f"Error processing image {img_name}: {e}")
            continue

    # 更新标注信息中的 bbox
    for ann in original_annotations.get("annotations", []):
        updated_ann = ann.copy()
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x_min, y_min, width, height = bbox
            updated_ann["bbox"] = [
                x_min / scale_factor,
                y_min / scale_factor,
                width / scale_factor,
                height / scale_factor
            ]
            updated_annotations["annotations"].append(updated_ann)
        else:
            logger.warning(f"Skipping annotation due to missing or invalid bbox: {ann}")


    # 保存更新后的标注文件到 output_dir_lr 的根目录
    updated_annotation_file_path = os.path.join(output_dir_lr, "annotations.json")
    try:
        with open(updated_annotation_file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_annotations, f, indent=4)
        logger.info(f"Updated annotations saved to {updated_annotation_file_path}")
    except Exception as e:
        logger.error(f"Error saving updated annotation file: {e}")


def validate_input_paths(input_dir, annotation_file):
    """
    验证输入路径是否存在。
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")
    if not os.path.isfile(annotation_file):
        raise FileNotFoundError(f"Input annotation file does not exist: {annotation_file}")
    # 简单检查输入目录是否包含图片，可根据需要增强
    if not any(fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) for fname in os.listdir(input_dir)):
        logger.warning(f"No common image files found in input directory: {input_dir}. Ensure it contains HR images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LR images and adjust annotations for detection, including categories.")
    # 注意：命令行参数名已修正（移除了空格）
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original high-resolution images.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the original annotation file (JSON format for HR images).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save generated LR images (in 'LR' subdir) and the new annotations.json.")
    parser.add_argument("--scale_factor", type=int, default=4, help="Downscaling factor (e.g., 4 for x4).")

    args = parser.parse_args()

    try:
        validate_input_paths(args.input_dir, args.annotation_file)
        # output_dir 将是新标注文件和 LR 子目录的根目录
        prepare_detection_data(args.input_dir, args.annotation_file, args.output_dir, args.scale_factor)
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

# 该脚本用于准备目标检测训练数据，包括生成低分辨率图像和更新标注文件。