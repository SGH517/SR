# data/prepare_sr_data.py
import os
from PIL import Image, UnidentifiedImageError # 导入 UnidentifiedImageError
import argparse
import logging # 导入 logging

# 为此模块设置一个 logger
logger_prep_sr = logging.getLogger(__name__)
if not logger_prep_sr.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _is_image_file(filename: str) -> bool:
    """辅助函数，检查文件是否是支持的图像格式。"""
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])

def prepare_sr_data(input_dir: str, output_dir: str, scale_factor: int, logger: logging.Logger):
    """
    从输入目录中的高分辨率图像生成 LR-HR 图像对。

    参数：
        input_dir (str): 包含高分辨率图像的输入目录。
        output_dir (str): 输出目录，将在此目录下创建 'HR' 和 'LR' 子目录。
        scale_factor (int): 下采样因子，例如 4 表示 x4 下采样。
        logger (logging.Logger): 用于记录日志的 logger 实例。
    """
    hr_output_dir = os.path.join(output_dir, "HR")
    lr_output_dir = os.path.join(output_dir, "LR")

    try:
        os.makedirs(hr_output_dir, exist_ok=True)
        os.makedirs(lr_output_dir, exist_ok=True)
        logger.info(f"输出目录已创建/确认: HR -> {hr_output_dir}, LR -> {lr_output_dir}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {e}")
        return # 无法创建目录，则无法继续

    num_processed = 0
    num_errors = 0

    image_filenames = [f for f in os.listdir(input_dir) if _is_image_file(f)]
    if not image_filenames:
        logger.warning(f"输入目录 {input_dir} 中未找到支持的图像文件。")
        return

    logger.info(f"开始处理 {input_dir} 中的 {len(image_filenames)} 个图像文件...")

    for img_name in image_filenames:
        hr_img_path_original = os.path.join(input_dir, img_name)
        # 输出文件名保持与输入一致
        hr_img_path_dest = os.path.join(hr_output_dir, img_name)
        lr_img_path_dest = os.path.join(lr_output_dir, img_name)

        try:
            with Image.open(hr_img_path_original) as img:
                img_rgb = img.convert("RGB") # 确保图像是 RGB 格式

                # 保存 HR 图像 (直接从源复制或重新保存以统一格式)
                # 为简单起见，这里重新保存，可以确保所有输出HR图像的格式（例如都转为PNG或JPG）
                # 如果希望保持原始格式，可以考虑使用 shutil.copy2
                img_rgb.save(hr_img_path_dest) # 可以指定格式, e.g., img_rgb.save(hr_img_path_dest, "PNG")

                # 生成并保存 LR 图像
                # 确保原始图像尺寸足够进行下采样
                if img_rgb.width < scale_factor or img_rgb.height < scale_factor:
                    logger.warning(f"图像 {img_name} 的尺寸 ({img_rgb.width}x{img_rgb.height}) "
                                   f"过小，无法进行 {scale_factor}x 下采样。跳过此图像的LR生成。")
                    # HR 图像仍然被保存了，但没有对应的 LR
                    # 可以选择删除已保存的HR，或保留它（取决于策略）
                    # os.remove(hr_img_path_dest) # 如果希望严格成对
                    continue # 跳过此图像的LR部分

                lr_width = img_rgb.width // scale_factor
                lr_height = img_rgb.height // scale_factor

                if lr_width == 0 or lr_height == 0:
                    logger.warning(f"图像 {img_name} 下采样后的尺寸为零 ({lr_width}x{lr_height})。跳过此图像的LR生成。")
                    continue

                lr_img = img_rgb.resize((lr_width, lr_height), Image.BICUBIC)
                lr_img.save(lr_img_path_dest) # 可以指定格式

                # logger.debug(f"已处理 {img_name}: 源 HR -> {hr_img_path_original}, "
                #            f"目标 HR -> {hr_img_path_dest}, 目标 LR -> {lr_img_path_dest}")
                num_processed += 1

        except FileNotFoundError:
            logger.error(f"处理图像 {img_name} 时源文件未找到: {hr_img_path_original}")
            num_errors += 1
        except UnidentifiedImageError:
            logger.warning(f"无法识别的图像文件 (可能已损坏): {hr_img_path_original}。跳过。")
            num_errors += 1
        except Exception as e:
            logger.error(f"处理图像 {img_name} ({hr_img_path_original}) 时发生错误: {e}", exc_info=True)
            num_errors += 1
            # 如果发生错误，确保部分创建的文件被清理（可选）
            if os.path.exists(hr_img_path_dest): os.remove(hr_img_path_dest)
            if os.path.exists(lr_img_path_dest): os.remove(lr_img_path_dest)
            continue
    
    logger.info(f"SR数据准备完成。成功处理 {num_processed} 张图像。发生 {num_errors} 个错误。")


def validate_input_directory(input_dir_path: str, logger: logging.Logger) -> bool:
    """
    验证输入目录是否存在且包含有效图像文件。
    """
    if not os.path.isdir(input_dir_path): # 使用 isdir 检查目录
        logger.error(f"输入目录不存在或不是一个目录: {input_dir_path}")
        return False
    if not any(_is_image_file(fname) for fname in os.listdir(input_dir_path)):
        logger.warning(f"在目录 {input_dir_path} 中未找到支持的图像文件。")
        # 根据需求，这可能不是一个致命错误，所以返回 True 但记录警告
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为超分辨率任务准备 LR-HR 图像对。")
    parser.add_argument("--input_dir", type=str, required=True, help="包含原始高分辨率图像的目录。")
    parser.add_argument("--output_dir", type=str, required=True, help="保存生成的 LR 和 HR 图像的目录。")
    parser.add_argument("--scale_factor", type=int, default=4, help="下采样因子 (例如, 4 代表 x4)。")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别。")
    args = parser.parse_args()

    # 设置主脚本的 logger
    script_logger = logging.getLogger("prepare_sr_data_script")
    script_logger.handlers.clear() # 清除已存在的处理器，以防多次运行脚本时重复添加
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
    script_logger.propagate = False # 防止日志消息向上传播到根logger，避免重复打印

    script_logger.info(f"脚本参数: {args}")

    if not validate_input_directory(args.input_dir, script_logger):
        script_logger.error("输入目录校验失败。退出脚本。")
        exit(1)

    prepare_sr_data(args.input_dir, args.output_dir, args.scale_factor, script_logger)