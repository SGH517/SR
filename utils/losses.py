# utils/losses.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import logging

# 确保 DetectorWrapper._format_targets_for_yolo 在某个地方可以被调用
# 或者将类似的功能直接实现在这里或 ConditionalSR 中
# from models.detector import DetectorWrapper # 假设可以访问到


def compute_yolo_loss_from_predictions(
    yolo_raw_predictions: Any, # 来自 DetectorWrapper 的原始输出
    targets: List[Dict],       # COCO 格式的标注
    sr_image_shape: Tuple[int, int, int, int], # (B, C, H, W) SR 图像的形状，用于归一化
    device: torch.device,
    config: Dict, # 可能包含类别数等信息
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """
    使用 YOLO 的原始预测和目标来计算检测损失。
    【【【这部分是核心，需要根据 YOLO 模型的具体输出格式和损失函数来详细实现】】】
    """
    if logger: logger.debug(f"Attempting to compute YOLO loss from raw predictions. SR image shape: {sr_image_shape}")
    
    total_detection_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 1. 准备 Targets
    #    你需要将 COCO targets 转换为 YOLO 损失函数期望的格式。
    #    这通常包括：
    #    - 对每个图像，将 GT boxes 转换为 (class_idx, x_center_norm, y_center_norm, w_norm, h_norm)
    #    - 可能需要根据预测的 anchor/grid 进行匹配 (对于 anchor-based)
    #    - 或者准备用于 anchor-free 方法的 target tensor。
    
    #    一个简化的 target 格式化 (基于 DetectorWrapper._format_targets_for_yolo)
    yolo_formatted_targets_for_loss_calculation = []
    batch_size, _, img_h, img_w = sr_image_shape

    if not targets:
        if logger: logger.warning("No targets provided to compute_yolo_loss_from_predictions.")
        return total_detection_loss # 返回零损失

    for i, target_dict in enumerate(targets):
        boxes_abs_coco = target_dict.get('boxes') # COCO: [x_min, y_min, width, height]
        labels = target_dict.get('labels')

        if boxes_abs_coco is None or labels is None or boxes_abs_coco.numel() == 0:
            yolo_formatted_targets_for_loss_calculation.append(torch.empty((0, 5), device=device, dtype=torch.float32))
            continue
        
        boxes_abs_coco = boxes_abs_coco.float().to(device)
        labels = labels.to(device)

        boxes_xywh_abs = torch.zeros_like(boxes_abs_coco)
        boxes_xywh_abs[:, 0] = boxes_abs_coco[:, 0] + boxes_abs_coco[:, 2] / 2
        boxes_xywh_abs[:, 1] = boxes_abs_coco[:, 1] + boxes_abs_coco[:, 3] / 2
        boxes_xywh_abs[:, 2] = boxes_abs_coco[:, 2]
        boxes_xywh_abs[:, 3] = boxes_abs_coco[:, 3]

        boxes_xywh_norm = boxes_xywh_abs.clone()
        boxes_xywh_norm[:, [0, 2]] /= img_w
        boxes_xywh_norm[:, [1, 3]] /= img_h
        boxes_xywh_norm[:, 0:4] = torch.clamp(boxes_xywh_norm[:, 0:4], min=0.0, max=1.0)
        
        valid_indices = (boxes_xywh_norm[:, 2] > 1e-4) & (boxes_xywh_norm[:, 3] > 1e-4)
        if not valid_indices.all():
            boxes_xywh_norm = boxes_xywh_norm[valid_indices]
            labels_filtered = labels[valid_indices]
        else:
            labels_filtered = labels
        
        if boxes_xywh_norm.numel() == 0:
            yolo_formatted_targets_for_loss_calculation.append(torch.empty((0, 5), device=device, dtype=torch.float32))
            continue
            
        yolo_target_for_image = torch.cat((labels_filtered.float().unsqueeze(1), boxes_xywh_norm), dim=1)
        yolo_formatted_targets_for_loss_calculation.append(yolo_target_for_image)

    # 2. 解析 YOLO Raw Predictions
    #    yolo_raw_predictions 的格式取决于你的 YOLO 模型 (特别是 Detect head)。
    #    对于 YOLOv8, 它通常是包含 (batch_size, num_outputs, num_ detección_features_per_level) 的张量列表/元组，
    #    或者是一个拼接后的大张量 (batch_size, sum_of_all_predictions, num_outputs_per_prediction)。
    #    num_outputs_per_prediction 通常是 4 (bbox) + 1 (obj_conf) + num_classes。
    #    或者对于DFL，bbox部分可能是 4 * reg_max。

    #    【【你需要根据实际的 yolo_raw_predictions 格式来解析它】】
    #    例如，如果 yolo_raw_predictions 是一个列表，每个元素对应一个FPN层级的预测：
    #    pred_s, pred_m, pred_l = yolo_raw_predictions # 假设有三个层级

    # 3. 计算损失组件 (Box, Class, Objectness/DFL)
    #    这需要你参考 Ultralytics YOLO 的损失实现（例如 v8DetectionLoss）
    #    或者使用标准的 torchvision 损失函数。
    
    #    伪代码/占位符：
    #    loss_box = torch.tensor(0.0, device=device)
    #    loss_cls = torch.tensor(0.0, device=device)
    #    loss_dfl_or_obj = torch.tensor(0.0, device=device)

    #    for i in range(batch_size): # 遍历批次中的每张图像
    #        current_preds = ... # 从 yolo_raw_predictions 中获取第 i 张图像的预测
    #        current_targets = yolo_formatted_targets_for_loss_calculation[i]
    #
    #        if current_targets.numel() == 0: continue # 没有 GT 目标
    #
    #        # 进行预测和目标的匹配 (例如，基于 IoU)
    #        matched_preds, matched_targets = match_predictions_to_targets(current_preds, current_targets, ...)
    #
    #        # 计算各部分损失
    #        loss_box += calculate_box_loss(matched_preds_bbox, matched_targets_bbox, ...)
    #        loss_cls += calculate_cls_loss(matched_preds_cls, matched_targets_cls, ...)
    #        loss_dfl_or_obj += calculate_dfl_obj_loss(matched_preds_dfl_obj, matched_targets_obj, ...)
    
    # total_detection_loss = (loss_box + loss_cls + loss_dfl_or_obj) / batch_size

    if logger: 
        logger.warning("YOLO loss computation from raw predictions is NOT YET FULLY IMPLEMENTED in compute_yolo_loss_from_predictions.")
        logger.warning("Returning ZERO detection loss as a placeholder.")
        
    # 【【【占位符 - 返回零损失，直到上面实现完成】】】
    # 为了让训练能跑起来，暂时返回一个需要梯度的零张量
    if total_detection_loss.grad_fn is None and yolo_raw_predictions is not None:
        # 如果 yolo_raw_predictions 是一个张量列表/元组，取第一个元素的和乘以0来赋予梯度依赖
        if isinstance(yolo_raw_predictions, (list, tuple)) and len(yolo_raw_predictions) > 0 and torch.is_tensor(yolo_raw_predictions[0]):
            total_detection_loss = (yolo_raw_predictions[0].sum() * 0.0).requires_grad_()
        elif torch.is_tensor(yolo_raw_predictions):
            total_detection_loss = (yolo_raw_predictions.sum() * 0.0).requires_grad_()
        # 否则，它可能无法获得梯度

    return total_detection_loss


def calculate_joint_loss(
    sr_images: torch.Tensor,
    mask_coarse: Optional[torch.Tensor],
    targets: Optional[List[Dict]],      # COCO 格式的 targets
    yolo_raw_predictions: Optional[Any], # ConditionalSR forward 方法中 'yolo_raw_predictions' 的值
    config: Dict,
    logger: Optional[logging.Logger] = None,
    # precomputed_detection_loss 参数现在不再可靠，因为 DetectorWrapper 返回 None
    # 我们将依赖 yolo_raw_predictions 和 targets 来计算损失
    precomputed_detection_loss: Optional[Union[torch.Tensor, Dict]] = None 
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    loss_dict: Dict[str, float] = {}
    device = sr_images.device # 主设备

    # 1. 计算检测损失
    detection_weight = config['train']['loss_weights'].get('detection', 1.0)
    loss_detection = torch.tensor(0.0, device=device) 

    if detection_weight > 0 and yolo_raw_predictions is not None and targets is not None:
        loss_detection = compute_yolo_loss_from_predictions(
            yolo_raw_predictions,
            targets,
            sr_images.shape, # Pass SR image shape for normalization context
            device,
            config,
            logger
        )
        if not (torch.is_tensor(loss_detection) and loss_detection.requires_grad):
            if logger: logger.warning(f"Computed detection loss is not a tensor or does not require grad. Value: {loss_detection}")
            loss_detection = torch.tensor(0.0, device=device) # Fallback
            
    elif detection_weight > 0:
        if logger: logger.warning("Detection loss calculation skipped: yolo_raw_predictions or targets were None.")
    
    loss_dict["loss_detection"] = loss_detection.item() if torch.is_tensor(loss_detection) else 0.0
    
    # 2. 计算稀疏度损失 (与目标比率的 MSE)
    # ... (这部分逻辑不变) ...
    sparsity_weight = config['train']['loss_weights'].get('sparsity', 0.0)
    loss_sparsity = torch.tensor(0.0, device=device)
    actual_sparsity_val = 0.0 # 用于记录
    if sparsity_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        actual_sparsity_tensor = torch.mean(mask_coarse.float())
        actual_sparsity_val = actual_sparsity_tensor.item()
        target_sparsity_ratio = config['train'].get('target_sparsity_ratio', 0.0)
        target_sparsity_tensor = torch.tensor(target_sparsity_ratio, device=device, dtype=actual_sparsity_tensor.dtype)
        loss_sparsity = torch.nn.functional.mse_loss(actual_sparsity_tensor, target_sparsity_tensor)
    loss_dict["loss_sparsity"] = loss_sparsity.item() if torch.is_tensor(loss_sparsity) else 0.0
    loss_dict["actual_sparsity"] = actual_sparsity_val

    # 3. 计算平滑度损失 (TV Loss)
    # ... (这部分逻辑不变) ...
    smoothness_weight = config['train']['loss_weights'].get('smoothness', 0.0)
    loss_smooth = torch.tensor(0.0, device=device)
    if smoothness_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        if mask_coarse.dim() == 4 and mask_coarse.shape[1] == 1: # B, 1, H, W
            dh = torch.abs(mask_coarse[:, :, :-1, :] - mask_coarse[:, :, 1:, :])
            dw = torch.abs(mask_coarse[:, :, :, :-1] - mask_coarse[:, :, :, 1:])
            loss_smooth = (torch.sum(dh) + torch.sum(dw)) / mask_coarse.size(0) # Sum over all then mean over Batch
        else:
            if logger: logger.warning(f"Smoothness loss expects mask_coarse with shape (B, 1, H, W), got {mask_coarse.shape}. Skipping loss.")
    loss_dict["loss_smooth"] = loss_smooth.item() if torch.is_tensor(loss_smooth) else 0.0
    
    # 4. 加权求和
    # 确保所有参与加法的损失都是标量张量
    if not (torch.is_tensor(loss_detection) and loss_detection.numel() == 1): loss_detection = torch.tensor(loss_detection, device=device)
    if not (torch.is_tensor(loss_sparsity) and loss_sparsity.numel() == 1): loss_sparsity = torch.tensor(loss_sparsity, device=device)
    if not (torch.is_tensor(loss_smooth) and loss_smooth.numel() == 1): loss_smooth = torch.tensor(loss_smooth, device=device)
    
    total_loss = (
        detection_weight * loss_detection +
        sparsity_weight * loss_sparsity +
        smoothness_weight * loss_smooth
    )

    loss_dict["total_loss"] = total_loss.item()

    if logger and hasattr(logger, 'info'): # 确保 logger 有 info 方法
        # Format for better readability and ensure values are numbers
        det_val = loss_dict.get('loss_detection', 0.0)
        spar_val = loss_dict.get('loss_sparsity', 0.0)
        act_spar_val = loss_dict.get('actual_sparsity', 0.0)
        smooth_val = loss_dict.get('loss_smooth', 0.0)
        total_val = loss_dict.get('total_loss', 0.0)

        log_str = f"Loss (unweighted): Det={det_val:.4f} Spar={spar_val:.4f} (Act={act_spar_val:.4f}) Smooth={smooth_val:.4f}"
        logger.debug(log_str)
        log_str_weighted = (f"Loss (weighted): Det={detection_weight * det_val:.4f} "
                            f"Spar={sparsity_weight * spar_val:.4f} "
                            f"Smooth={smoothness_weight * smooth_val:.4f} "
                            f"TOTAL={total_val:.4f}")
        logger.debug(log_str_weighted)

    return total_loss, loss_dict