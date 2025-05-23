# data/prepare_detection_data.py
import os
import json
from PIL import Image, UnidentifiedImageError # 导入 UnidentifiedImageError
import argparse
import logging
from typing import List, Dict, Any, Optional # 增加类型提示

# 为此模块设置一个 logger
logger_prep_det = logging.getLogger(__name__)
if not logger_prep_det.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _is_image_file(filename: str) -> bool:
    """辅助函数，检查文件是否是支持的图像格式。"""
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])

def prepare_detection_data(input_dir_hr: str,
                           annotation_file_hr_path: str,
                           output_dir_base: str, # LR图像和新标注文件的基础输出目录
                           scale_factor: int,
                           logger: logging.Logger):
    """
    为目标检测任务准备数据：
    1. 对高分辨率图像进行下采样以生成低分辨率（LR）图像。
    2. 调整COCO格式标注文件中的边界框（BBox）坐标以匹配LR图像。
    3. 复制原始标注中的 'categories' 字段。

    参数:
        input_dir_hr (str): 包含原始高分辨率图像的输入目录。
        annotation_file_hr_path (str): 原始高分辨率图像的COCO标注文件路径 (JSON格式)。
        output_dir_base (str): LR图像将保存在此目录下的 'LR' 子目录中，
                               调整后的标注文件 (annotations.json) 将直接保存在此目录中。
        scale_factor (int): 下采样因子 (例如, 4 表示 x4 下采样)。
        logger (logging.Logger): 用于记录日志的 logger 实例。
    """
    lr_images_subdir = os.path.join(output_dir_base, "LR") # LR图像的子目录
    try:
        os.makedirs(lr_images_subdir, exist_ok=True)
        logger.info(f"LR图像输出子目录已创建/确认: {lr_images_subdir}")
    except OSError as e:
        logger.error(f"创建LR图像输出子目录 {lr_images_subdir} 失败: {e}")
        return False # 指示操作失败

    # 加载原始标注文件
    try:
        with open(annotation_file_hr_path, 'r', encoding='utf-8') as f:
            original_annotations_data: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        logger.error(f"原始标注文件未找到: {annotation_file_hr_path}")
        return False
    except json.JSONDecodeError:
        logger.error(f"解码原始标注文件 {annotation_file_hr_path} 中的 JSON 时出错。")
        return False
    except Exception as e:
        logger.error(f"加载原始标注文件 {annotation_file_hr_path} 时发生未知错误: {e}", exc_info=True)
        return False


    updated_annotations_data: Dict[str, List[Any]] = {"images": [], "annotations": []}
    num_images_processed = 0
    num_image_errors = 0
    num_annotation_errors = 0

    # 复制 'categories' 字段 (如果存在)
    if 'categories' in original_annotations_data:
        updated_annotations_data['categories'] = original_annotations_data['categories']
        logger.info("已从原始标注复制 'categories' 字段。")
    else:
        logger.warning(f"原始标注文件 {annotation_file_hr_path} 中未找到 'categories' 键。"
                       "输出的标注文件将缺少此字段，这可能导致COCO评估等后续步骤出现问题。")
        updated_annotations_data['categories'] = [] # 提供一个空的 categories 列表

    # --- 处理图像并更新图像信息 ---
    original_images_info: List[Dict] = original_annotations_data.get("images", [])
    logger.info(f"开始处理 {len(original_images_info)} 张图像的下采样和信息更新...")

    for img_info_hr in original_images_info:
        hr_img_filename = img_info_hr.get("file_name", "")
        if not hr_img_filename:
            logger.warning(f"图像信息中缺少 'file_name' 字段: {img_info_hr}。跳过此图像。")
            num_image_errors += 1
            continue

        # 原始HR图像的完整路径
        # input_dir_hr 是HR图像所在的目录, hr_img_filename 是相对于该目录的文件名（可能包含子路径）
        hr_img_full_path = os.path.join(input_dir_hr, hr_img_filename)
        hr_img_full_path = os.path.normpath(hr_img_full_path)


        # 生成的LR图像将保存在 output_dir_base/LR/ 下，文件名与原始文件名（可能包含的子路径）相同
        # 例如，如果 hr_img_filename 是 "subdir/image.jpg"，则 lr_img_relative_to_lr_subdir 也是 "subdir/image.jpg"
        lr_img_relative_to_lr_subdir = hr_img_filename
        lr_img_save_full_path = os.path.join(lr_images_subdir, lr_img_relative_to_lr_subdir)

        # 确保LR图像的保存路径中的任何子目录都存在
        try:
            os.makedirs(os.path.dirname(lr_img_save_full_path), exist_ok=True)
        except OSError as e_mkdir: # 防御性编程，通常 exist_ok=True 会处理
            logger.warning(f"为LR图像 {lr_img_save_full_path} 创建子目录失败: {e_mkdir}。跳过此图像。")
            num_image_errors +=1
            continue

        try:
            with Image.open(hr_img_full_path) as img_pil:
                img_pil_rgb = img_pil.convert("RGB") # 确保是 RGB 格式

                original_hr_width = img_info_hr.get("width", img_pil_rgb.width)
                original_hr_height = img_info_hr.get("height", img_pil_rgb.height)

                # 确保原始图像尺寸足够进行下采样
                if original_hr_width < scale_factor or original_hr_height < scale_factor:
                    logger.warning(f"HR图像 {hr_img_filename} 的尺寸 ({original_hr_width}x{original_hr_height}) "
                                   f"过小，无法进行 {scale_factor}x 下采样。跳过此图像。")
                    num_image_errors += 1
                    continue

                lr_width = original_hr_width // scale_factor
                lr_height = original_hr_height // scale_factor

                if lr_width == 0 or lr_height == 0:
                    logger.warning(f"HR图像 {hr_img_filename} 下采样后的尺寸为零 ({lr_width}x{lr_height})。跳过此图像。")
                    num_image_errors += 1
                    continue

                lr_img_pil = img_pil_rgb.resize((lr_width, lr_height), Image.BICUBIC)
                lr_img_pil.save(lr_img_save_full_path) # 可以指定格式, e.g., "PNG"

                # 更新图像信息
                updated_img_info_lr = img_info_hr.copy()
                updated_img_info_lr["width"] = lr_width
                updated_img_info_lr["height"] = lr_height
                # COCO file_name 应该是相对于图像根目录（即 output_dir_base）的路径
                # 所以，如果LR图像保存在 output_dir_base/LR/subdir/image.jpg
                # 那么 file_name 应该是 "LR/subdir/image.jpg"
                updated_img_info_lr["file_name"] = os.path.join("LR", lr_img_relative_to_lr_subdir).replace("\\", "/")
                updated_annotations_data["images"].append(updated_img_info_lr)
                num_images_processed += 1

        except FileNotFoundError:
            logger.warning(f"HR图像文件未找到: {hr_img_full_path} (原始文件名: {hr_img_filename})。跳过此图像。")
            num_image_errors += 1
            continue
        except UnidentifiedImageError:
            logger.warning(f"无法识别的HR图像文件 (可能已损坏): {hr_img_full_path}。跳过。")
            num_image_errors +=1
            continue
        except Exception as e_img:
            logger.error(f"处理HR图像 {hr_img_filename} ({hr_img_full_path}) 时发生错误: {e_img}", exc_info=True)
            num_image_errors += 1
            continue
    logger.info(f"图像处理完成。成功处理 {num_images_processed} 张图像，发生 {num_image_errors} 个错误。")


    # --- 更新标注信息中的 bbox ---
    original_annotations: List[Dict] = original_annotations_data.get("annotations", [])
    logger.info(f"开始更新 {len(original_annotations)} 条标注的边界框...")

    for ann_hr in original_annotations:
        # 确保标注对应一个已被成功处理并添加到 updated_annotations_data["images"] 中的图像
        # 这可以通过 image_id 匹配来实现
        image_id_ann = ann_hr.get("image_id")
        if image_id_ann is None:
            logger.warning(f"标注中缺少 'image_id': {ann_hr}。跳过此标注。")
            num_annotation_errors+=1
            continue

        # 检查此 image_id 是否存在于我们更新后的图像列表中
        if not any(img_lr_info['id'] == image_id_ann for img_lr_info in updated_annotations_data["images"]):
            logger.debug(f"标注 (id: {ann_hr.get('id')}) 对应的图像 (image_id: {image_id_ann}) 未被成功处理或包含在更新后的图像列表中。跳过此标注。")
            # num_annotation_errors+=1 # 这个可能不算错误，只是图像未被处理
            continue


        updated_ann_lr = ann_hr.copy()
        bbox_hr = ann_hr.get("bbox") # COCO format: [x_min, y_min, width, height]

        if bbox_hr and len(bbox_hr) == 4:
            x_min_hr, y_min_hr, width_hr, height_hr = bbox_hr
            # 缩放 bbox 坐标和尺寸
            x_min_lr = x_min_hr / scale_factor
            y_min_lr = y_min_hr / scale_factor
            width_lr = width_hr / scale_factor
            height_lr = height_hr / scale_factor

            # 可选：检查缩放后的 width_lr 和 height_lr 是否过小，如果需要可以过滤
            # min_bbox_dim_lr = 1.0 # 例如，LR图像上的最小宽高为1像素
            # if width_lr < min_bbox_dim_lr or height_lr < min_bbox_dim_lr:
            #     logger.debug(f"标注 (id: {ann_hr.get('id')}) 在缩放后尺寸过小 (w={width_lr:.2f}, h={height_lr:.2f})。跳过。")
            #     num_annotation_errors +=1 # 可以选择是否算作错误
            #     continue

            updated_ann_lr["bbox"] = [x_min_lr, y_min_lr, width_lr, height_lr]

            # （可选）COCO 格式也包含 "area" 字段，它也应该被缩放
            if "area" in updated_ann_lr:
                updated_ann_lr["area"] = (width_lr * height_lr) # 或者 area_hr / (scale_factor**2)

            updated_annotations_data["annotations"].append(updated_ann_lr)
        else:
            logger.warning(f"标注 (id: {ann_hr.get('id')}) 缺少有效的 'bbox' 字段或格式不正确。跳过此标注。")
            num_annotation_errors += 1

    logger.info(f"标注更新完成。成功转换 {len(updated_annotations_data['annotations'])} 条标注。"
                f"处理过程中跳过或发生错误 {num_annotation_errors} 条。")

    # --- 保存更新后的标注文件到 output_dir_base 的根目录 ---
    updated_annotation_file_path = os.path.join(output_dir_base, "annotations_lr.json") # 建议文件名包含 "lr"
    try:
        with open(updated_annotation_file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_annotations_data, f, indent=2) # indent=2 减小文件大小
        logger.info(f"已更新的LR标注文件已保存到: {updated_annotation_file_path}")
    except Exception as e_save:
        logger.error(f"保存已更新的标注文件时发生错误: {e_save}", exc_info=True)
        return False
    return True


def validate_detection_input_paths(input_dir: str, annotation_file: str, logger: logging.Logger) -> bool:
    """
    验证检测数据准备脚本的输入路径。
    """
    valid = True
    if not os.path.isdir(input_dir):
        logger.error(f"输入HR图像目录不存在或不是一个目录: {input_dir}")
        valid = False
    elif not any(_is_image_file(fname) for fname in os.listdir(input_dir)):
        logger.warning(f"在输入HR图像目录 {input_dir} 中未找到支持的图像文件。")
        # 这不一定是致命错误，所以不将 valid 设为 False

    if not os.path.isfile(annotation_file):
        logger.error(f"输入HR标注文件不存在或不是一个文件: {annotation_file}")
        valid = False
    return valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为目标检测准备LR图像和调整后的COCO标注，包括categories字段。")
    parser.add_argument("--input_dir_hr", type=str, required=True, help="包含原始高分辨率图像的目录。")
    parser.add_argument("--annotation_file_hr", type=str, required=True,
                        help="原始高分辨率图像的COCO格式标注文件路径。")
    parser.add_argument("--output_dir_base", type=str, required=True,
                        help="基础输出目录。LR图像将保存在其 'LR' 子目录中，新的 'annotations_lr.json' 将保存在此目录下。")
    parser.add_argument("--scale_factor", type=int, default=4, help="下采样因子 (例如, 4 代表 x4)。")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别。")
    args = parser.parse_args()

    # 设置主脚本的 logger
    script_logger = logging.getLogger("prepare_detection_data_script")
    script_logger.handlers.clear()
    ch = logging.StreamHandler()
    try:
        script_logger.setLevel(args.log_level.upper())
        ch.setLevel(args.log_level.upper())
    except ValueError:
        script_logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        script_logger.warning(f"无效的日志级别 '{args.log_level}'。使用 INFO 级别。")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    script_logger.addHandler(ch)
    script_logger.propagate = False

    script_logger.info(f"脚本参数: {args}")

    if not validate_detection_input_paths(args.input_dir_hr, args.annotation_file_hr, script_logger):
        script_logger.error("输入路径校验失败。退出脚本。")
        exit(1)

    success = prepare_detection_data(args.input_dir_hr, args.annotation_file_hr,
                                     args.output_dir_base, args.scale_factor, script_logger)

    if success:
        script_logger.info("检测数据准备成功完成。")
    else:
        script_logger.error("检测数据准备过程中发生错误。")