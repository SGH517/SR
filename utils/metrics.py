import torch
import torch.nn.functional as F
import math

# 该模块提供了评估指标的计算函数，例如PSNR和SSIM。

def calculate_psnr(sr, hr, max_pixel_value=1.0):
    """
    计算 PSNR (峰值信噪比)。
    """
    mse = F.mse_loss(sr, hr)
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(sr, hr, max_pixel_value=1.0):
    """
    计算 SSIM (结构相似性)。
    支持多通道图像 (如 RGB)。
    """
    C1 = (0.01 * max_pixel_value) ** 2
    C2 = (0.03 * max_pixel_value) ** 2

    if sr.ndim == 4:  # Batch 处理
        ssim_values = []
        for i in range(sr.size(0)):
            ssim_values.append(calculate_ssim(sr[i], hr[i], max_pixel_value))
        return torch.tensor(ssim_values).mean().item()

    # 单张图像处理
    if sr.size(1) > 1:  # 多通道图像处理
        ssim_channels = []
        for channel in range(sr.size(1)):
            ssim_channels.append(calculate_ssim(sr[:, channel:channel+1, :, :], hr[:, channel:channel+1, :, :], max_pixel_value))
        return torch.tensor(ssim_channels).mean().item()

    mu_sr = F.avg_pool2d(sr, kernel_size=11, stride=1, padding=5)
    mu_hr = F.avg_pool2d(hr, kernel_size=11, stride=1, padding=5)

    sigma_sr = F.avg_pool2d(sr * sr, kernel_size=11, stride=1, padding=5) - mu_sr ** 2
    sigma_hr = F.avg_pool2d(hr * hr, kernel_size=11, stride=1, padding=5) - mu_hr ** 2
    sigma_sr_hr = F.avg_pool2d(sr * hr, kernel_size=11, stride=1, padding=5) - mu_sr * mu_hr

    ssim = ((2 * mu_sr * mu_hr + C1) * (2 * sigma_sr_hr + C2)) / ((mu_sr ** 2 + mu_hr ** 2 + C1) * (sigma_sr + sigma_hr + C2))
    return ssim.mean().item()