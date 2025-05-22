import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SRDataset(Dataset):
    """
    用于超分辨率任务的数据集类。
    支持从LR和HR图像目录加载数据，并可选择性地进行图像块裁剪和基本增强。
    """
    def __init__(self, lr_dir, hr_dir, patch_size=None, scale_factor=4, transform=None, augment=True):
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
        self.transform = transform # For normalization after ToTensor
        self.augment = augment

        self.lr_filenames = sorted([f for f in os.listdir(lr_dir) if self._is_image_file(f)])
        self.hr_filenames = sorted([f for f in os.listdir(hr_dir) if self._is_image_file(f)])

        if len(self.lr_filenames) != len(self.hr_filenames):
            raise ValueError(f"LR 和 HR 图像数量不匹配: {len(self.lr_filenames)} vs {len(self.hr_filenames)}")

        # 验证文件名是否一一对应 (简单检查)
        for lr_fn, hr_fn in zip(self.lr_filenames, self.hr_filenames):
            if os.path.splitext(lr_fn)[0] != os.path.splitext(hr_fn)[0]:
                # 如果文件名（不含扩展名）需要严格一致，可以在这里添加更严格的检查或报错
                pass # print(f"Warning: Filename mismatch? {lr_fn} vs {hr_fn}")

        self.to_tensor = transforms.ToTensor() # To convert PIL images to tensors

    def _is_image_file(self, filename):
        return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif'])

    def __getitem__(self, index):
        lr_img_path = os.path.join(self.lr_dir, self.lr_filenames[index])
        hr_img_path = os.path.join(self.hr_dir, self.hr_filenames[index % len(self.hr_filenames)]) # 使用 % 防止索引越界，并假设文件名对应

        try:
            lr_img = Image.open(lr_img_path).convert('RGB')
            hr_img = Image.open(hr_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 找不到图像文件 {lr_img_path} 或 {hr_img_path}")
            # 返回None或引发异常，或者返回一个占位符（如果dataloader的collate_fn能处理）
            # 这里简单返回None，您可能需要更健壮的错误处理
            return None, None
        except Exception as e:
            print(f"加载图像时出错 {lr_img_path} 或 {hr_img_path}: {e}")
            return None, None


        if self.patch_size_lr:
            # 获取随机裁剪参数 (针对LR图像)
            lr_w, lr_h = lr_img.size
            
            # 如果图像本身小于patch_size，则直接使用原图（或按需进行其他处理如padding或resize）
            patch_lr_eff_w = min(lr_w, self.patch_size_lr)
            patch_lr_eff_h = min(lr_h, self.patch_size_lr)

            if lr_w > patch_lr_eff_w:
                left_lr = random.randint(0, lr_w - patch_lr_eff_w)
            else:
                left_lr = 0
            
            if lr_h > patch_lr_eff_h:
                top_lr = random.randint(0, lr_h - patch_lr_eff_h)
            else:
                top_lr = 0
            
            # 裁剪LR图像块
            lr_patch = lr_img.crop((left_lr, top_lr, left_lr + patch_lr_eff_w, top_lr + patch_lr_eff_h))

            # 计算HR图像块的对应参数
            left_hr = left_lr * self.scale_factor
            top_hr = top_lr * self.scale_factor
            patch_hr_eff_w = patch_lr_eff_w * self.scale_factor
            patch_hr_eff_h = patch_lr_eff_h * self.scale_factor
            
            # 裁剪HR图像块
            hr_patch = hr_img.crop((left_hr, top_hr, left_hr + patch_hr_eff_w, top_hr + patch_hr_eff_h))
        else:
            # 如果不使用patch，则使用完整图像
            lr_patch = lr_img
            hr_patch = hr_img

        # 数据增强 (在转换为Tensor之前对PIL Image进行)
        if self.augment:
            # 随机水平翻转
            if random.random() < 0.5:
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            
            # (可选) 随机旋转 90, 180, 270 度
            # rot_k = random.randint(0, 3)
            # if rot_k > 0:
            #     lr_patch = lr_patch.rotate(90 * rot_k)
            #     hr_patch = hr_patch.rotate(90 * rot_k)

        # 转换为Tensor
        lr_tensor = self.to_tensor(lr_patch)
        hr_tensor = self.to_tensor(hr_patch)

        # 应用额外的transform (例如归一化)
        if self.transform:
            # 注意：如果transform包含随机操作，需要确保它能同步处理一对图像，
            # 或者只包含确定性操作如Normalization。
            # 这里假设transform是类似Normalization的操作。
            lr_tensor = self.transform(lr_tensor)
            hr_tensor = self.transform(hr_tensor) # 通常HR图像也需要同样的归一化

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.lr_filenames)

# 该模块定义了超分辨率数据集类，用于加载和处理训练/验证数据。
# SRDataset 类是一个 PyTorch 数据集，用于加载低分辨率（LR）和高分辨率（HR）图像对，
# 主要用于超分辨率任务的训练和评估。该类支持从指定目录加载图像文件，
# 可选地对图像进行裁剪（生成图像块）和增强（如随机翻转），
# 并将图像转换为张量格式以供模型训练使用。