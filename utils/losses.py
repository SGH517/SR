import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import logging  # Import logging

def calculate_joint_loss(
    sr_images: torch.Tensor, 
    mask_coarse: Optional[torch.Tensor],
    targets: Optional[List[Dict]], 
    detector: Optional[torch.nn.Module], # 这个参数可能不再需要，因为损失已经预计算
    config: Dict,
    logger: Optional[logging.Logger] = None,
    precomputed_detection_loss: Optional[Union[torch.Tensor, Dict]] = None # 这是 DetectorWrapper 返回的 loss_value
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_dict: Dict[str, float] = {}
    # device = sr_images.device # 从 precomputed_detection_loss 获取 device 如果它是 tensor
    device = None
    if torch.is_tensor(precomputed_detection_loss):
        device = precomputed_detection_loss.device
    elif isinstance(precomputed_detection_loss, dict) and precomputed_detection_loss:
        for v_loss in precomputed_detection_loss.values():
            if torch.is_tensor(v_loss):
                device = v_loss.device
                break
    if device is None: # Fallback
        device = sr_images.device

    # 1. 计算检测损失
    detection_weight = config['train']['loss_weights'].get('detection', 1.0)
    loss_detection = torch.tensor(0.0, device=device) 

    if detection_weight > 0 and precomputed_detection_loss is not None:
        if isinstance(precomputed_detection_loss, dict):
            current_loss_sum = torch.tensor(0.0, device=device)
            valid_loss_found = False
            for loss_key, loss_val in precomputed_detection_loss.items():
                if torch.is_tensor(loss_val) and loss_val.requires_grad: 
                    current_loss_sum += loss_val
                    loss_dict[f"det_{loss_key}"] = loss_val.item() # 记录分量损失
                    valid_loss_found = True
            if valid_loss_found:
                loss_detection = current_loss_sum
            else:
                if logger: logger.warning(f"Precomputed detection loss (dict) did not contain valid tensor losses: {precomputed_detection_loss}")
        elif torch.is_tensor(precomputed_detection_loss):
            loss_detection = precomputed_detection_loss
        else:
            if logger: logger.warning(f"Unexpected precomputed_detection_loss format: {type(precomputed_detection_loss)}. Detection loss will be 0.")
    elif detection_weight > 0 and precomputed_detection_loss is None:
        if logger: logger.warning("Detection loss calculation skipped: precomputed_detection_loss was None from DetectorWrapper.")
    
    loss_dict["loss_detection"] = loss_detection.item() if torch.is_tensor(loss_detection) and loss_detection.numel() == 1 else 0.0
    
    # 2. 计算稀疏度损失 (与目标比率的 MSE)
    sparsity_weight = config['train']['loss_weights'].get('sparsity', 0.0)
    loss_sparsity = torch.tensor(0.0, device=device)
    actual_sparsity = torch.tensor(0.0, device=device)
    if sparsity_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        actual_sparsity = torch.mean(mask_coarse.float())
        target_sparsity_ratio = config['train'].get('target_sparsity_ratio', 0.0)
        target_sparsity_tensor = torch.tensor(target_sparsity_ratio, device=device, dtype=actual_sparsity.dtype)
        loss_sparsity = torch.nn.functional.mse_loss(actual_sparsity, target_sparsity_tensor) # 使用 F.mse_loss
    loss_dict["loss_sparsity"] = loss_sparsity.item() if torch.is_tensor(loss_sparsity) else 0.0
    loss_dict["actual_sparsity"] = actual_sparsity.item() if torch.is_tensor(actual_sparsity) else 0.0

    # 3. 计算平滑度损失 (TV Loss)
    smoothness_weight = config['train']['loss_weights'].get('smoothness', 0.0)
    loss_smooth = torch.tensor(0.0, device=device)
    if smoothness_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        if mask_coarse.dim() == 4 and mask_coarse.shape[1] == 1: # B, 1, H, W
            dh = torch.abs(mask_coarse[:, :, :-1, :] - mask_coarse[:, :, 1:, :])
            dw = torch.abs(mask_coarse[:, :, :, :-1] - mask_coarse[:, :, :, 1:])
            # Sum over H, W, C (channel is 1) then mean over Batch
            loss_smooth = (torch.sum(dh, dim=[1,2,3]) + torch.sum(dw, dim=[1,2,3])).mean()
            # 或者像之前那样 sum over all then mean
            # loss_smooth = torch.mean(torch.sum(dh, dim=[1, 2, 3]) + torch.sum(dw, dim=[1, 2, 3]))
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
        if logger: logger.warning(f"total_loss is not a scalar tensor. Type: {type(total_loss)}. Value: {total_loss}. Setting to 0.")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True if loss_detection.requires_grad or loss_sparsity.requires_grad or loss_smooth.requires_grad else False)


    loss_dict["total_loss"] = total_loss.item()

    if logger and hasattr(logger, 'info'):
        log_str = f"Loss (unweighted): Det={loss_dict.get('loss_detection', 0.0):.4f} Spar={loss_dict.get('loss_sparsity', 0.0):.4f} (Act={loss_dict.get('actual_sparsity', 0.0):.4f}) Smooth={loss_dict.get('loss_smooth', 0.0):.4f}"
        logger.debug(log_str)
        log_str_weighted = f"Loss (weighted): Det={detection_weight * loss_dict.get('loss_detection', 0.0):.4f} Spar={sparsity_weight * loss_dict.get('loss_sparsity', 0.0):.4f} Smooth={smoothness_weight * loss_dict.get('loss_smooth', 0.0):.4f} TOTAL={total_loss.item():.4f}"
        logger.debug(log_str_weighted)

    return total_loss, loss_dict
