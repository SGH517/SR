import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import math # For isnan

# 从新的辅助模块导入核心组件
from .yolo_loss_utils import (
    TaskAlignedAssigner,
    BboxLoss,
    make_anchors,
    dist2bbox, # dist2bbox is used by _bbox_decode_local
    xywh2xyxy, # Used in target preparation
    # bbox_iou is used internally by TaskAlignedAssigner and BboxLoss
)

# 确保 DetectorWrapper._format_targets_for_yolo 在某个地方可以被调用
# 或者将类似的功能直接实现在这里或 ConditionalSR 中
# from models.detector import DetectorWrapper # 假设可以访问到


def compute_yolo_loss_from_predictions(
    yolo_raw_predictions: List[torch.Tensor], # List of 3 tensors [B, C, H, W] from ConditionalSR
    targets_coco_format: List[Dict],        # List of dicts (COCO format from DataLoader)
    sr_image_shape: Tuple[int, int, int, int], # (B, C, H_sr, W_sr) SR image shape
    device: torch.device,
    config: Dict,                           # Main config dictionary
    yolo_model_components: Dict,            # {'stride': Tensor, 'nc': int, 'reg_max': int, 'no': int, 'hyp': Dict}
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """
    计算YOLOv8的检测损失。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers: # pragma: no cover
            logging.basicConfig(level=logging.INFO)

    batch_size = sr_image_shape[0]
    img_h_sr, img_w_sr = sr_image_shape[-2:]

    # 从 yolo_model_components 和 config 获取参数
    stride = yolo_model_components['stride'].to(device)
    nc = yolo_model_components['nc']
    reg_max = yolo_model_components['reg_max']
    no = yolo_model_components['no'] # nc + reg_max * 4
    hyp = yolo_model_components['hyp']

    use_dfl = reg_max > 1
    proj_df = torch.arange(reg_max, dtype=torch.float, device=device)

    box_gain = hyp.get('box', 7.5)
    cls_gain = hyp.get('cls', 0.5)
    dfl_gain = hyp.get('dfl', 1.5)
    # label_smoothing_eps = hyp.get('label_smoothing', 0.0) # Not directly used here; TAL produces soft targets

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")

    tal_config = config.get('train', {}).get('yolo_assigner_params', {})
    assigner = TaskAlignedAssigner(
        topk=tal_config.get('topk', 10),
        num_classes=nc,
        alpha=tal_config.get('alpha', 0.5),
        beta=tal_config.get('beta', 6.0),
        eps=tal_config.get('eps', 1e-9), # Add eps to assigner if it uses it
        use_ciou=tal_config.get('use_ciou_for_tal_metric', False)
    )
    bbox_loss_calculator = BboxLoss(reg_max_val=reg_max, use_dfl=use_dfl).to(device)

    # 1. 解析预测 (feats -> pred_distri, pred_scores)
    pred_distri_list, pred_scores_list = [], []
    for feat_map_lvl in yolo_raw_predictions:
        bs_lvl, _, h_lvl, w_lvl = feat_map_lvl.shape
        pred_lvl = feat_map_lvl.view(bs_lvl, no, -1).permute(0, 2, 1)
        dist_lvl, score_lvl = pred_lvl.split((reg_max * 4, nc), dim=2)
        pred_distri_list.append(dist_lvl)
        pred_scores_list.append(score_lvl)
    
    pred_distri = torch.cat(pred_distri_list, dim=1) # [B, total_A, 4 * reg_max]
    pred_scores = torch.cat(pred_scores_list, dim=1)   # [B, total_A, nc] (raw logits)

    # 2. 生成锚点
    anchor_points, stride_tensor = make_anchors(yolo_raw_predictions, stride, grid_cell_offset=0.5)
    # anchor_points: [total_A, 2], stride_tensor: [total_A, 1] (feature map cell units)

    # 3. 预处理真实目标 (targets_coco_format -> yolo_formatted_targets -> lists for assigner)
    # This part is from the original losses.py, converting COCO to normalized xywh
    yolo_formatted_targets = []
    if not targets_coco_format: # handle case where targets_coco_format might be None
        logger.warning("No targets provided to compute_yolo_loss_from_predictions.")
        # Create empty targets to prevent downstream errors if assigner expects lists
        for _ in range(batch_size):
            yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
    else:
        for i, target_dict in enumerate(targets_coco_format):
            boxes_abs_coco = target_dict.get('boxes')
            labels = target_dict.get('labels')

            if boxes_abs_coco is None or labels is None or boxes_abs_coco.numel() == 0:
                yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
                continue
            
            boxes_abs_coco = boxes_abs_coco.float().to(device)
            labels = labels.to(device)

            boxes_xywh_abs = torch.zeros_like(boxes_abs_coco)
            boxes_xywh_abs[:, 0] = boxes_abs_coco[:, 0] + boxes_abs_coco[:, 2] / 2
            boxes_xywh_abs[:, 1] = boxes_abs_coco[:, 1] + boxes_abs_coco[:, 3] / 2
            boxes_xywh_abs[:, 2] = boxes_abs_coco[:, 2]
            boxes_xywh_abs[:, 3] = boxes_abs_coco[:, 3]

            boxes_xywh_norm = boxes_xywh_abs.clone()
            boxes_xywh_norm[:, [0, 2]] /= img_w_sr
            boxes_xywh_norm[:, [1, 3]] /= img_h_sr
            boxes_xywh_norm[:, 0:4] = torch.clamp(boxes_xywh_norm[:, 0:4], min=0.0, max=1.0)
            
            valid_indices = (boxes_xywh_norm[:, 2] > 1e-4) & (boxes_xywh_norm[:, 3] > 1e-4)
            if not valid_indices.all():
                boxes_xywh_norm = boxes_xywh_norm[valid_indices]
                labels_filtered = labels[valid_indices]
            else:
                labels_filtered = labels
            
            if boxes_xywh_norm.numel() == 0:
                yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
                continue
                
            yolo_target_for_image = torch.cat((labels_filtered.float().unsqueeze(1), boxes_xywh_norm), dim=1)
            yolo_formatted_targets.append(yolo_target_for_image)

    # Prepare lists for TaskAlignedAssigner
    gt_labels_list_assigner = []
    gt_bboxes_list_assigner = [] # xyxy, image scale
    mask_gt_list_assigner = []

    for i in range(batch_size):
        gt_info_img_i = yolo_formatted_targets[i] # shape [num_gt_for_image_i, 5] (cls, cx_n, cy_n, w_n, h_n)
        if gt_info_img_i.numel() == 0:
            gt_labels_list_assigner.append(torch.empty((0,1), dtype=torch.long, device=device))
            gt_bboxes_list_assigner.append(torch.empty((0,4), dtype=torch.float, device=device))
            mask_gt_list_assigner.append(torch.empty((0,1), dtype=torch.bool, device=device))
            continue

        gt_cls_img_i = gt_info_img_i[:, 0:1].long()
        gt_xywh_norm_img_i = gt_info_img_i[:, 1:]
        
        gt_xyxy_norm_img_i = xywh2xyxy(gt_xywh_norm_img_i) # Converts normalized xywh to normalized xyxy

        gt_xyxy_imgscale_img_i = gt_xyxy_norm_img_i.clone()
        gt_xyxy_imgscale_img_i[:, [0,2]] *= img_w_sr
        gt_xyxy_imgscale_img_i[:, [1,3]] *= img_h_sr
        
        gt_labels_list_assigner.append(gt_cls_img_i)
        gt_bboxes_list_assigner.append(gt_xyxy_imgscale_img_i)
        mask_gt_list_assigner.append(torch.ones_like(gt_cls_img_i, dtype=torch.bool))


    # 4. 解码预测边界框 (DFL分布 -> xyxy 特征图尺度)
    def _bbox_decode_local(anchor_pts_in, pred_dist_input, use_dfl_flag, proj_tensor, reg_max_val_in):
        # anchor_pts_in: [total_A, 2]
        # pred_dist_input: [B, total_A, 4 * reg_max_val_in]
        if use_dfl_flag:
            b_dec, a_dec, c_dec = pred_dist_input.shape
            pred_dist_out = pred_dist_input.view(b_dec, a_dec, 4, c_dec // 4).softmax(3).matmul(proj_tensor.to(pred_dist_input.dtype))
        else: # Should not happen if reg_max > 1
            pred_dist_out = pred_dist_input.view(b_dec, a_dec, 4, c_dec // 4).mean(3) # Placeholder if not DFL
        # anchor_pts_in needs to be broadcastable to (B, total_A, 2) for dist2bbox
        return dist2bbox(pred_dist_out, anchor_pts_in.unsqueeze(0), xywh=False)

    pred_bboxes_feat_scale = _bbox_decode_local(anchor_points, pred_distri, use_dfl, proj_df, reg_max)
    # pred_bboxes_feat_scale: [B, total_A, 4] (xyxy, feature map cell units)

    # 5. 执行目标分配
    # For assigner, pd_bboxes should be image scale
    pred_bboxes_img_scale_detached = (pred_bboxes_feat_scale.detach() * stride_tensor.unsqueeze(0).to(device))
    # anchor_points for assigner also needs to be image scale as per TAL logic for select_candidates_in_gts
    anchor_points_img_scale_for_assigner = anchor_points * stride_tensor.to(device)


    target_labels, target_bboxes_img_scale, target_scores, fg_mask, target_gt_idx = assigner(
        pred_scores.detach().sigmoid(),
        pred_bboxes_img_scale_detached.type_as(gt_bboxes_list_assigner[0] if any(g.numel() > 0 for g in gt_bboxes_list_assigner) else pred_bboxes_img_scale_detached),
        anchor_points_img_scale_for_assigner, # Assigner expects anc_points on image scale for select_candidates_in_gts
        gt_labels_list_assigner,
        gt_bboxes_list_assigner,
        mask_gt_list_assigner,
    )
    # fg_mask: [B, total_A] (boolean)
    # target_bboxes_img_scale: [B, total_A, 4] (xyxy image scale, for positive anchors)
    # target_scores: [B, total_A, nc] (soft targets for BCE)

    # 6. 计算损失
    target_scores_sum = max(target_scores.sum(), 1.0)

    loss_cls = torch.tensor(0.0, device=device)
    loss_iou = torch.tensor(0.0, device=device)
    loss_dfl = torch.tensor(0.0, device=device)
    
    # Classification Loss
    loss_cls_terms = bce_criterion(pred_scores, target_scores.to(pred_scores.dtype)) # [B, total_A, nc]
    loss_cls = loss_cls_terms.sum() / target_scores_sum

    num_fg_total = fg_mask.sum()
    if num_fg_total > 0:
        # Bbox Loss (CIoU + DFL)
        # For bbox_loss_calculator, target_bboxes need to be on feature map scale
        # target_bboxes_img_scale is [B, total_A, 4] (image scale, for positive regions defined by fg_mask)
        # pred_bboxes_feat_scale is [B, total_A, 4] (feature scale)
        # anchor_points is [total_A, 2] (feature scale)
        # pred_distri is [B, total_A, 4*reg_max]

        # Mask positive predictions and targets for BboxLoss
        masked_pred_dist = pred_distri[fg_mask]              # [N_fg, 4 * reg_max]
        masked_pred_bboxes_feat = pred_bboxes_feat_scale[fg_mask]  # [N_fg, 4]
        
        # Anchor points for positive predictions
        # anchor_points is [total_A, 2]. We need to select based on fg_mask.
        # fg_mask is [B, total_A]. Need to find which anchors are positive across batch.
        # This requires careful indexing if anchor_points is not [B, total_A, 2]
        # Assuming anchor_points can be broadcast or correctly indexed for positive ones.
        # Let's make anchor_points batched first:
        batched_anchor_points = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1) # [B, total_A, 2]
        masked_anchor_points_feat = batched_anchor_points[fg_mask]    # [N_fg, 2]

        # Target bboxes for positive predictions, scaled to feature map
        # target_bboxes_img_scale is [B, total_A, 4]
        # stride_tensor is [total_A, 1]
        batched_stride_tensor = stride_tensor.unsqueeze(0).repeat(batch_size, 1, 1) # [B, total_A, 1]
        masked_target_bboxes_img = target_bboxes_img_scale[fg_mask] # [N_fg, 4] (image scale)
        masked_strides = batched_stride_tensor[fg_mask]             # [N_fg, 1]
        masked_target_bboxes_feat = masked_target_bboxes_img / masked_strides # [N_fg, 4] (feat scale)

        # Target scores for positive predictions (for weighting in BboxLoss)
        masked_target_scores = target_scores[fg_mask] # [N_fg, nc]

        loss_iou_val, loss_dfl_val = bbox_loss_calculator(
            pred_dist=masked_pred_dist,
            pred_bboxes=masked_pred_bboxes_feat,
            anchor_points=masked_anchor_points_feat,
            target_bboxes=masked_target_bboxes_feat,
            target_scores=masked_target_scores, # For calculating weight inside BboxLoss
            target_scores_sum=target_scores_sum, # Global sum for normalization consistency
            fg_mask_from_assigner=None # Not needed as inputs are already masked
        )
        loss_iou = loss_iou_val
        loss_dfl = loss_dfl_val
    
    # Apply gains
    loss_iou *= box_gain
    loss_dfl *= dfl_gain
    loss_cls *= cls_gain

    total_detection_loss = (loss_iou + loss_dfl + loss_cls) * batch_size
    
    if torch.isnan(total_detection_loss): # pragma: no cover
        logger.warning("NaN detected in YOLO detection loss. Replacing with 0.0.")
        logger.debug(f"NaN details: loss_iou={loss_iou.item()}, loss_dfl={loss_dfl.item()}, loss_cls={loss_cls.item()}")
        # Potentially log more details about inputs if NaN occurs
        total_detection_loss = torch.tensor(0.0, device=device, requires_grad=True)


    # For debugging, log individual unweighted components if needed
    if logger and logger.level <= logging.DEBUG: # pragma: no cover
        # These are already normalized by target_scores_sum
        # To get per-positive-anchor average:
        # loss_iou_unweighted_avg = (loss_iou / box_gain / batch_size) if box_gain > 0 and batch_size > 0 and num_fg_total > 0 else 0
        # Similar for dfl and cls. The current ones are sums over batch.
        log_str_yolo = f"YOLO Loss Components (scaled by gains, sum over batch): IoU={loss_iou.item():.4f}, DFL={loss_dfl.item():.4f}, Cls={loss_cls.item():.4f}"
        logger.debug(log_str_yolo)
        if num_fg_total > 0:
            logger.debug(f"Num positive anchors (fg_mask.sum()): {num_fg_total.item()}")
            logger.debug(f"Target scores sum (normalization factor): {target_scores_sum.item()}")
        else:
            logger.debug("No positive anchors found in this batch for YOLO loss.")


    return total_detection_loss


def calculate_joint_loss(
    sr_images: torch.Tensor,
    mask_coarse: Optional[torch.Tensor],
    targets: Optional[List[Dict]],      # COCO 格式的 targets
    yolo_raw_predictions: Optional[Any], # ConditionalSR forward 方法中 'yolo_raw_predictions' 的值
    config: Dict,
    logger: Optional[logging.Logger] = None,
    # 新增: YOLO模型特定参数，由 stage3_finetune_joint.py 准备和传递
    yolo_model_components_for_loss: Optional[Dict] = None,
    precomputed_detection_loss: Optional[Union[torch.Tensor, Dict]] = None # No longer reliable
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    loss_dict: Dict[str, float] = {}
    device = sr_images.device

    # 1. 计算检测损失
    detection_weight = config['train']['loss_weights'].get('detection', 1.0)
    loss_detection = torch.tensor(0.0, device=device) 

    if detection_weight > 0 and yolo_raw_predictions is not None and targets is not None and yolo_model_components_for_loss is not None:
        try:
            loss_detection = compute_yolo_loss_from_predictions(
                yolo_raw_predictions,
                targets, # targets_coco_format
                sr_images.shape, # Pass SR image shape for normalization context
                device,
                config, # Main config
                yolo_model_components_for_loss, # Dict with stride, nc, reg_max, no, hyp
                logger
            )
        except Exception as e: # pragma: no cover
            if logger:
                logger.error(f"Error during YOLO loss computation: {e}", exc_info=True)
            loss_detection = torch.tensor(0.0, device=device) # Fallback on error
            
        if not (torch.is_tensor(loss_detection) and loss_detection.requires_grad): # pragma: no cover
            # If loss_detection became a non-tensor (e.g. float due to error) or lost grad
            if logger: logger.warning(f"Computed detection loss is not a tensor or does not require grad. Value: {loss_detection}. Resetting to 0.")
            loss_detection = torch.tensor(0.0, device=device, requires_grad=True) # Ensure it's a tensor needing grad
            # To give it a grad_fn if it's just a zero tensor and yolo_raw_predictions exist:
            if isinstance(yolo_raw_predictions, (list, tuple)) and len(yolo_raw_predictions) > 0 and torch.is_tensor(yolo_raw_predictions[0]):
                loss_detection = (loss_detection + yolo_raw_predictions[0].sum() * 0.0) # Make it depend on preds
            elif torch.is_tensor(yolo_raw_predictions):
                loss_detection = (loss_detection + yolo_raw_predictions.sum() * 0.0)


    elif detection_weight > 0: # pragma: no cover
        if logger: 
            if yolo_raw_predictions is None: logger.warning("Detection loss calculation skipped: yolo_raw_predictions was None.")
            if targets is None: logger.warning("Detection loss calculation skipped: targets were None.")
            if yolo_model_components_for_loss is None: logger.warning("Detection loss calculation skipped: yolo_model_components_for_loss was None.")
    
    loss_dict["loss_detection"] = loss_detection.item() if torch.is_tensor(loss_detection) else float(loss_detection) # Ensure float
    
    # 2. 计算稀疏度损失 (与目标比率的 MSE)
    sparsity_weight = config['train']['loss_weights'].get('sparsity', 0.0)
    loss_sparsity = torch.tensor(0.0, device=device)
    actual_sparsity_val = 0.0
    if sparsity_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        actual_sparsity_tensor = torch.mean(mask_coarse.float())
        actual_sparsity_val = actual_sparsity_tensor.item()
        target_sparsity_ratio = config['train'].get('target_sparsity_ratio', 0.0)
        target_sparsity_tensor = torch.tensor(target_sparsity_ratio, device=device, dtype=actual_sparsity_tensor.dtype)
        loss_sparsity = F.mse_loss(actual_sparsity_tensor, target_sparsity_tensor)
    loss_dict["loss_sparsity"] = loss_sparsity.item() if torch.is_tensor(loss_sparsity) else float(loss_sparsity)
    loss_dict["actual_sparsity"] = actual_sparsity_val

    # 3. 计算平滑度损失 (TV Loss)
    smoothness_weight = config['train']['loss_weights'].get('smoothness', 0.0)
    loss_smooth = torch.tensor(0.0, device=device)
    if smoothness_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        if mask_coarse.dim() == 4 and mask_coarse.shape[1] == 1: # B, 1, H, W
            dh = torch.abs(mask_coarse[:, :, :-1, :] - mask_coarse[:, :, 1:, :])
            dw = torch.abs(mask_coarse[:, :, :, :-1] - mask_coarse[:, :, :, 1:])
            loss_smooth = (torch.sum(dh) + torch.sum(dw)) / mask_coarse.size(0) 
        else: # pragma: no cover
            if logger: logger.warning(f"Smoothness loss expects mask_coarse with shape (B, 1, H, W), got {mask_coarse.shape}. Skipping loss.")
    loss_dict["loss_smooth"] = loss_smooth.item() if torch.is_tensor(loss_smooth) else float(loss_smooth)
    
    # 4. 加权求和
    # Ensure all participating losses are scalar tensors
    if not (torch.is_tensor(loss_detection) and loss_detection.numel() == 1): # pragma: no cover
        loss_detection = torch.tensor(loss_detection, device=device)
    if not (torch.is_tensor(loss_sparsity) and loss_sparsity.numel() == 1): # pragma: no cover
        loss_sparsity = torch.tensor(loss_sparsity, device=device)
    if not (torch.is_tensor(loss_smooth) and loss_smooth.numel() == 1): # pragma: no cover
        loss_smooth = torch.tensor(loss_smooth, device=device)
    
    total_loss = (
        detection_weight * loss_detection +
        sparsity_weight * loss_sparsity +
        smoothness_weight * loss_smooth
    )

    loss_dict["total_loss"] = total_loss.item()

    if logger and hasattr(logger, 'debug'): # pragma: no cover
        det_val = loss_dict.get('loss_detection', 0.0)
        spar_val = loss_dict.get('loss_sparsity', 0.0)
        act_spar_val = loss_dict.get('actual_sparsity', 0.0)
        smooth_val = loss_dict.get('loss_smooth', 0.0)
        total_val = loss_dict.get('total_loss', 0.0)

        log_str = f"Loss (unweighted per component avg/val): Det={det_val:.4f} Spar={spar_val:.4f} (Act={act_spar_val:.4f}) Smooth={smooth_val:.4f}"
        logger.debug(log_str)
        log_str_weighted = (f"Loss (weighted components for total): wDet={detection_weight * det_val:.4f} "
                            f"wSpar={sparsity_weight * spar_val:.4f} "
                            f"wSmooth={smoothness_weight * smooth_val:.4f} "
                            f"TOTAL={total_val:.4f}")
        logger.debug(log_str_weighted)

    return total_loss, loss_dict