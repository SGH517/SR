# data/sr_dataset.py
import os
from PIL import Image, UnidentifiedImageError # 导入 UnidentifiedImageError
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging # 导入 logging

# 为此模块设置一个 logger
logger_sr_dataset = logging.getLogger(__name__)
if not logger_sr_dataset.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


class SRDataset(Dataset):
    """
    用于超分辨率任务的数据集类。
    支持从LR和HR图像目录加载数据，并可选择性地进行图像块裁剪和基本增强。
    """
    def __init__(self, lr_dir: str, hr_dir: str, patch_size: Optional[int] = None,
                 scale_factor: int = 4, transform: Optional[callable] = None, augment: bool = True):
        """
        初始化数据集。

        参数:
            lr_dir (str): 低分辨率图像的目录。
            hr_dir (str): 高分辨率图像的目录。
            patch_size (int, optional): 低分辨率图像块的边长。如果为 None，则使用完整图像。默认为 None。
            scale_factor (int): 超分辨率的放大倍数。默认为 4。
            transform (callable, optional): 应用于LR和HR图像（Tensor格式）的转换操作，
                                          通常是归一化。ToTensor() 会在内部应用。默认为 None。
            augment (bool): 是否应用数据增强（随机翻转）。默认为 True。
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size_lr = patch_size  # LR patch size
        self.scale_factor = scale_factor
        self.transform = transform # 用于Tensor格式图像的额外转换 (例如归一化)
        self.augment = augment

        if not os.path.isdir(self.lr_dir):
            logger_sr_dataset.error(f"LR图像目录未找到: {self.lr_dir}")
            raise FileNotFoundError(f"LR图像目录未找到: {self.lr_dir}")
        if not os.path.isdir(self.hr_dir):
            logger_sr_dataset.error(f"HR图像目录未找到: {self.hr_dir}")
            raise FileNotFoundError(f"HR图像目录未找到: {self.hr_dir}")

        try:
            self.lr_filenames = sorted([f for f in os.listdir(lr_dir) if self._is_image_file(f)])
            self.hr_filenames = sorted([f for f in os.listdir(hr_dir) if self._is_image_file(f)])
        except Exception as e:
            logger_sr_dataset.error(f"列出目录 {lr_dir} 或 {hr_dir} 中的文件时出错: {e}")
            raise

        if not self.lr_filenames:
            logger_sr_dataset.warning(f"LR图像目录 {self.lr_dir} 中未找到图像文件。")
        if not self.hr_filenames:
            logger_sr_dataset.warning(f"HR图像目录 {self.hr_dir} 中未找到图像文件。")


        if len(self.lr_filenames) != len(self.hr_filenames):
            logger_sr_dataset.error(f"LR 和 HR 图像数量不匹配: {len(self.lr_filenames)} (LR) vs {len(self.hr_filenames)} (HR)")
            # 可以选择是否在此处引发错误，取决于是否允许不匹配
            # raise ValueError("LR 和 HR 图像数量不匹配")

        # 验证文件名是否一一对应 (简单检查文件名主体是否相同)
        # 这一步可以根据实际数据集的文件命名规则调整或移除
        # for lr_fn, hr_fn in zip(self.lr_filenames, self.hr_filenames):
        #     if os.path.splitext(lr_fn)[0] != os.path.splitext(hr_fn)[0]:
        #         logger_sr_dataset.warning(f"文件名主体可能不匹配: LR='{lr_fn}', HR='{hr_fn}'")
        #         # 根据严格程度，可以选择是否在此处引发错误

        self.to_tensor = transforms.ToTensor() # 将 PIL 图像转换为张量

    def _is_image_file(self, filename: str) -> bool:
        """检查文件是否是支持的图像格式。"""
        return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取指定索引处的LR和HR图像对。"""
        if index >= len(self.lr_filenames) or index >= len(self.hr_filenames):
            logger_sr_dataset.error(f"索引 {index} 超出范围。LR图像数量: {len(self.lr_filenames)}, HR图像数量: {len(self.hr_filenames)}")
            # 对于 collate_fn，返回 None, None 会被过滤掉
            return None, None


        lr_img_name = self.lr_filenames[index]
        # 假设HR文件名与LR文件名（除扩展名外）相同，或者它们是按顺序对应的
        # 如果文件名不完全对应，需要更复杂的匹配逻辑
        hr_img_name = self.hr_filenames[index] # 如果数量相同且排序一致

        lr_img_path = os.path.join(self.lr_dir, lr_img_name)
        hr_img_path = os.path.join(self.hr_dir, hr_img_name)

        try:
            lr_img_pil = Image.open(lr_img_path).convert('RGB')
            hr_img_pil = Image.open(hr_img_path).convert('RGB')
        except FileNotFoundError:
            logger_sr_dataset.warning(f"找不到图像文件: {lr_img_path} 或 {hr_img_path} (索引: {index})")
            return None, None # collate_fn 会处理这个
        except UnidentifiedImageError: # PIL 无法识别图像格式
            logger_sr_dataset.warning(f"无法识别的图像文件 (可能已损坏): {lr_img_path} 或 {hr_img_path} (索引: {index})")
            return None, None
        except Exception as e:
            logger_sr_dataset.warning(f"加载图像 {lr_img_name} 或 {hr_img_name} (索引: {index}) 时出错: {e}")
            return None, None

        # --- Patch 裁剪 ---
        if self.patch_size_lr and self.patch_size_lr > 0:
            lr_w, lr_h = lr_img_pil.size
            patch_lr_w_eff = min(lr_w, self.patch_size_lr)
            patch_lr_h_eff = min(lr_h, self.patch_size_lr)

            # 随机选择裁剪的左上角坐标 (针对LR图像)
            left_lr = random.randint(0, lr_w - patch_lr_w_eff) if lr_w > patch_lr_w_eff else 0
            top_lr = random.randint(0, lr_h - patch_lr_h_eff) if lr_h > patch_lr_h_eff else 0

            # 裁剪LR图像块
            lr_patch_pil = lr_img_pil.crop((left_lr, top_lr, left_lr + patch_lr_w_eff, top_lr + patch_lr_h_eff))

            # 计算HR图像块的对应参数
            # 确保HR的裁剪坐标和尺寸是整数，并且在HR图像范围内
            hr_w_img, hr_h_img = hr_img_pil.size
            left_hr = left_lr * self.scale_factor
            top_hr = top_lr * self.scale_factor
            patch_hr_w_eff = patch_lr_w_eff * self.scale_factor
            patch_hr_h_eff = patch_lr_h_eff * self.scale_factor

            # 防止HR裁剪区域超出HR图像边界
            left_hr = min(left_hr, hr_w_img - patch_hr_w_eff)
            top_hr = min(top_hr, hr_h_img - patch_hr_h_eff)
            # 再次确保不会因浮点精度导致负值 (尽管上面min应该处理了)
            left_hr = max(0, left_hr)
            top_hr = max(0, top_hr)


            # 裁剪HR图像块
            # crop方法的坐标是 (left, upper, right, lower)
            hr_patch_pil = hr_img_pil.crop((left_hr, top_hr,
                                          min(left_hr + patch_hr_w_eff, hr_w_img), # 确保 right 不超界
                                          min(top_hr + patch_hr_h_eff, hr_h_img)  # 确保 lower 不超界
                                          ))
        else:
            # 如果不使用patch，则使用完整图像
            lr_patch_pil = lr_img_pil
            hr_patch_pil = hr_img_pil

        # --- 数据增强 (在转换为Tensor之前对PIL Image进行) ---
        if self.augment:
            # 随机水平翻转
            if random.random() < 0.5:
                lr_patch_pil = lr_patch_pil.transpose(Image.FLIP_LEFT_RIGHT)
                hr_patch_pil = hr_patch_pil.transpose(Image.FLIP_LEFT_RIGHT)

            # (可选) 随机旋转 90, 180, 270 度
            # rot_k = random.randint(0, 3)
            # if rot_k > 0:
            #     # Image.rotate 不会改变图像尺寸，而是用填充或裁剪来适应旋转
            #     # 对于SR，我们通常希望保持尺寸对应，所以 Image.transpose(Image.ROTATE_90/180/270) 可能更好
            #     if rot_k == 1:
            #         lr_patch_pil = lr_patch_pil.transpose(Image.ROTATE_90)
            #         hr_patch_pil = hr_patch_pil.transpose(Image.ROTATE_90)
            #     elif rot_k == 2:
            #         lr_patch_pil = lr_patch_pil.transpose(Image.ROTATE_180)
            #         hr_patch_pil = hr_patch_pil.transpose(Image.ROTATE_180)
            #     elif rot_k == 3:
            #         lr_patch_pil = lr_patch_pil.transpose(Image.ROTATE_270)
            #         hr_patch_pil = hr_patch_pil.transpose(Image.ROTATE_270)

        # --- 转换为Tensor ---
        try:
            lr_tensor = self.to_tensor(lr_patch_pil)
            hr_tensor = self.to_tensor(hr_patch_pil)
        except Exception as e_to_tensor:
            logger_sr_dataset.warning(f"将图像 {lr_img_name} 或 {hr_img_name} (索引: {index}) 转换为张量时出错: {e_to_tensor}")
            return None, None


        # --- 应用额外的transform (例如归一化) ---
        if self.transform:
            try:
                lr_tensor = self.transform(lr_tensor)
                hr_tensor = self.transform(hr_tensor) # 通常HR图像也需要同样的归一化
            except Exception as e_transform:
                logger_sr_dataset.warning(f"对图像 {lr_img_name} 或 {hr_img_name} (索引: {index}) 应用额外转换时出错: {e_transform}")
                # 根据策略，可能返回 None, None 或未转换的张量
                return None, None # 或者 lr_tensor, hr_tensor (未进一步转换)

        # 确保裁剪后的LR和HR尺寸仍然满足缩放关系
        # 这在 patch_size_lr > lr_img.size 时可能不满足，但之前的 min() 处理了这个问题
        if self.patch_size_lr and self.patch_size_lr > 0 :
             if lr_tensor.shape[1] * self.scale_factor != hr_tensor.shape[1] or \
                lr_tensor.shape[2] * self.scale_factor != hr_tensor.shape[2]:
                 logger_sr_dataset.warning(f"图像 {lr_img_name} (索引: {index}): 裁剪/缩放后的 LR/HR 尺寸不匹配预期的 scale_factor。"
                                           f" LR形状: {lr_tensor.shape}, HR形状: {hr_tensor.shape}, 缩放因子: {self.scale_factor}")
                 # 根据策略决定是否返回 None 或引发错误
                 # return None, None


        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        # 返回 LR 和 HR 文件列表中较短的那个的长度，以避免因数量不匹配导致的索引错误
        # 但在 __init__ 中已经检查了数量，理论上应该相等
        return min(len(self.lr_filenames), len(self.hr_filenames))