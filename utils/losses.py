import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import logging  # Import logging

def calculate_joint_loss(
    sr_images: torch.Tensor,
    mask_coarse: Optional[torch.Tensor],
    targets: Optional[List[Dict]],
    detector: Optional[torch.nn.Module],  # Assuming DetectorWrapper or similar
    config: Dict,
    logger: Optional[logging.Logger] = None,  # Add logger as argument
    precomputed_detection_loss: Optional[Union[torch.Tensor, Dict]] = None  # Add precomputed loss arg
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算联合训练的总损失。

    参数:
        sr_images (torch.Tensor): 超分辨率图像 (B, C, H, W)。
        mask_coarse (Optional[torch.Tensor]): 掩码 (训练时可能为软掩码 [0, 1]) (B, 1, H', W') or similar。
        targets (Optional[list]): 目标检测的真实标注列表 (仅训练时需要)。
        detector (Optional[nn.Module]): 检测器包装器 (DetectorWrapper 实例)。
        config (dict): 配置字典。
        logger (Optional[logging.Logger]): 日志记录器实例。
        precomputed_detection_loss (Optional[Union[torch.Tensor, Dict]]): 预先计算的检测损失。

    返回:
        torch.Tensor: 总损失值。
        dict: 包含各分量损失值的字典 (用于日志记录)。
    """
    loss_dict: Dict[str, float] = {}
    device = sr_images.device  # 获取设备

    # 1. 计算检测损失
    detection_weight = config['train']['loss_weights'].get('detection', 1.0)
    loss_detection = torch.tensor(0.0, device=device)

    if detection_weight > 0:
        if precomputed_detection_loss is not None:
            # --- 使用预计算的损失 ---
            if isinstance(precomputed_detection_loss, dict):
                loss_detection = sum(loss for loss in precomputed_detection_loss.values() if torch.is_tensor(loss))
            elif torch.is_tensor(precomputed_detection_loss):
                loss_detection = precomputed_detection_loss
            else:
                if logger: logger.warning(f"Unexpected precomputed_detection_loss format: {type(precomputed_detection_loss)}")
        elif detector is not None and detector.model is not None and targets is not None:
            # --- 如果没有预计算损失，则尝试在此计算 ---
            if hasattr(detector, 'training') and not detector.training:
                if logger: logger.warning("Detector is not in training mode during loss calculation.")
            try:
                _preds, loss_result = detector(sr_images, targets=targets)
                if isinstance(loss_result, dict):
                    loss_detection = sum(loss for loss in loss_result.values() if torch.is_tensor(loss))
                elif torch.is_tensor(loss_result):
                    loss_detection = loss_result
                else:
                    if logger: logger.warning(f"Unexpected loss format from detector: {type(loss_result)}")
            except Exception as e:
                if logger: logger.error(f"Error calculating detection loss: {e}", exc_info=True)  # Log traceback
        else:
            if logger: logger.warning("Detection loss calculation skipped: No precomputed loss, detector unavailable, or targets missing.")

    # Store detection loss value
    loss_dict["loss_detection"] = loss_detection.item() if torch.is_tensor(loss_detection) and loss_detection.numel() == 1 else 0.0

    # 2. 计算稀疏度损失 (与目标比率的 MSE)
    sparsity_weight = config['train']['loss_weights'].get('sparsity', 0.0)
    loss_sparsity = torch.tensor(0.0, device=device)
    actual_sparsity = torch.tensor(0.0, device=device)  # 初始化
    if sparsity_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        actual_sparsity = torch.mean(mask_coarse.float())
        target_sparsity_ratio = config['train'].get('target_sparsity_ratio', 0.0)
        target_sparsity_tensor = torch.tensor(target_sparsity_ratio, device=device, dtype=actual_sparsity.dtype)
        loss_sparsity = F.mse_loss(actual_sparsity, target_sparsity_tensor)
    loss_dict["loss_sparsity"] = loss_sparsity.item() if torch.is_tensor(loss_sparsity) else 0.0
    loss_dict["actual_sparsity"] = actual_sparsity.item() if torch.is_tensor(actual_sparsity) else 0.0

    # 3. 计算平滑度损失 (TV Loss)
    smoothness_weight = config['train']['loss_weights'].get('smoothness', 0.0)
    loss_smooth = torch.tensor(0.0, device=device)
    if smoothness_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        if mask_coarse.dim() == 4 and mask_coarse.shape[1] == 1:
            dh = torch.abs(mask_coarse[:, :, :-1, :] - mask_coarse[:, :, 1:, :])
            dw = torch.abs(mask_coarse[:, :, :, :-1] - mask_coarse[:, :, :, 1:])
            loss_smooth = torch.mean(torch.sum(dh, dim=[1, 2, 3]) + torch.sum(dw, dim=[1, 2, 3]))
        else:
            if logger: logger.warning(f"Smoothness loss expects mask_coarse with shape (B, 1, H, W), got {mask_coarse.shape}. Skipping loss.")
    loss_dict["loss_smooth"] = loss_smooth.item() if torch.is_tensor(loss_smooth) else 0.0

    # 4. 加权求和
    total_loss = (
        detection_weight * loss_detection +
        sparsity_weight * loss_sparsity +
        smoothness_weight * loss_smooth
    )
    if not (torch.is_tensor(total_loss) and total_loss.numel() == 1):
        if logger: logger.warning(f"total_loss is not a scalar tensor. Type: {type(total_loss)}. Setting to 0.")
        total_loss = torch.tensor(0.0, device=device)

    loss_dict["total_loss"] = total_loss.item()

    # 5. 使用传入的 logger 进行日志记录
    if logger and hasattr(logger, 'info'):  # Check if logger is valid
        log_str = f"Loss (unweighted): Det={loss_dict.get('loss_detection', 0.0):.4f} Spar={loss_dict.get('loss_sparsity', 0.0):.4f} (Act={loss_dict.get('actual_sparsity', 0.0):.4f}) Smooth={loss_dict.get('loss_smooth', 0.0):.4f}"
        logger.debug(log_str)  # Use debug level for more verbose logs if needed
        log_str_weighted = f"Loss (weighted): Det={detection_weight * loss_dict.get('loss_detection', 0.0):.4f} Spar={sparsity_weight * loss_dict.get('loss_sparsity', 0.0):.4f} Smooth={smoothness_weight * loss_dict.get('loss_smooth', 0.0):.4f}"
        logger.debug(log_str_weighted)

    return total_loss, loss_dict
