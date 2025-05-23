# utils/losses.py
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
    dist2bbox,
    xywh2xyxy,
    # bbox_iou is used internally by TaskAlignedAssigner and BboxLoss
)
# 从新的工具模块导入目标格式化函数
from .yolo_target_utils import format_coco_targets_to_yolo


# 确保 DetectorWrapper._format_targets_for_yolo 在某个地方可以被调用 (现在不需要了)
# 或者将类似的功能直接实现在这里或 ConditionalSR 中 (已移至 yolo_target_utils)

def compute_yolo_loss_from_predictions(
    yolo_raw_predictions: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]], # YOLO模型原始输出
    targets_coco_format: List[Dict],        # COCO格式的标注列表 (每个元素是一个图像的标注字典)
    sr_image_shape: Tuple[int, int, int, int], # (B, C, H_sr, W_sr) SR 图像形状
    device: torch.device,
    config: Dict,                           # 主配置字典
    yolo_model_components: Dict,            # {'stride': Tensor, 'nc': int, 'reg_max': int, 'no': int, 'hyp': Dict}
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """
    计算YOLOv8的检测损失。
    此函数现在期望 yolo_raw_predictions 是检测头部的原始输出 (例如，特征图列表)。
    targets_coco_format 是 COCO 风格的标注列表。
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers(): # pragma: no cover
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    batch_size = sr_image_shape[0]
    img_h_sr, img_w_sr = sr_image_shape[-2:]

    # 从 yolo_model_components 和 config 获取参数
    stride_tensor_from_cfg = yolo_model_components['stride'].to(device) # 重命名以避免与下面的 stride 混淆
    nc = yolo_model_components['nc']
    reg_max = yolo_model_components['reg_max']
    # no: number of outputs = nc + reg_max * 4
    # yolo_model_components['no'] 应该等于 nc + reg_max * 4
    expected_no = nc + reg_max * 4
    if yolo_model_components.get('no') != expected_no:
        logger.warning(f"配置中的 yolo_model_components['no'] ({yolo_model_components.get('no')}) "
                       f"与计算值 nc + reg_max * 4 ({expected_no}) 不匹配。将使用计算值。")
    no = expected_no # 使用计算值确保一致性

    hyp = yolo_model_components['hyp'] # 超参数字典

    use_dfl = reg_max > 1
    # DFL的投影向量 (0, 1, ..., reg_max-1)
    proj_df = torch.arange(reg_max, dtype=torch.float, device=device)

    box_gain = hyp.get('box', 7.5)
    cls_gain = hyp.get('cls', 0.5)
    dfl_gain = hyp.get('dfl', 1.5)
    # label_smoothing_eps = hyp.get('label_smoothing', 0.0) # TAL 会产生软目标，BCEWithLogitsLoss 处理原始 logits

    bce_criterion = nn.BCEWithLogitsLoss(reduction="none") # 用于分类损失

    tal_config = config.get('train', {}).get('yolo_assigner_params', {})
    assigner = TaskAlignedAssigner(
        topk=tal_config.get('topk', 10),
        num_classes=nc,
        alpha=tal_config.get('alpha', 0.5),
        beta=tal_config.get('beta', 6.0),
        eps=tal_config.get('eps', 1e-9),
        use_ciou=tal_config.get('use_ciou_for_tal_metric', False) # 确保此参数在TaskAlignedAssigner中被使用
    )
    bbox_loss_calculator = BboxLoss(reg_max_val=reg_max, use_dfl=use_dfl).to(device)

    # --- 1. 解析预测 (yolo_raw_predictions -> pred_distri, pred_scores) ---
    # yolo_raw_predictions 是来自 YOLO Detect() 模块的输出，通常是一个列表或元组，
    # 每个元素对应一个检测层的输出，形状为 [Batch, Channels, Height, Width]
    # Channels 通常是 no = num_classes + 4 * reg_max
    if not isinstance(yolo_raw_predictions, (list, tuple)):
        # 如果不是列表/元组 (例如，某些模型可能直接返回拼接后的张量)
        # 或者如果是一个单一的预测张量，需要根据其结构进行适配
        # 此处假设它总是列表/元组形式的特征图输出
        logger.error(f"yolo_raw_predictions 期望是列表或元组，但得到: {type(yolo_raw_predictions)}")
        return torch.tensor(0.0, device=device, requires_grad=True) # 返回零损失或错误

    if not yolo_raw_predictions or not all(isinstance(p, torch.Tensor) for p in yolo_raw_predictions):
        logger.error(f"yolo_raw_predictions 列表为空或包含非张量元素。")
        return torch.tensor(0.0, device=device, requires_grad=True)


    pred_distri_list, pred_scores_list = [], []
    for i, feat_map_lvl in enumerate(yolo_raw_predictions):
        bs_lvl, ch_lvl, h_lvl, w_lvl = feat_map_lvl.shape
        if ch_lvl != no:
            logger.error(f"第 {i} 个特征图的通道数 ({ch_lvl}) 与期望的输出通道数 no ({no}) 不匹配。")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 从 [B, C, H, W] 转换为 [B, H*W, C]
        pred_lvl = feat_map_lvl.view(bs_lvl, no, -1).permute(0, 2, 1) # [B, H*W, no]
        # 分割分布预测和类别分数预测
        # reg_max * 4 是 DFL 部分的通道数
        dist_lvl, score_lvl = pred_lvl.split((reg_max * 4, nc), dim=2)
        pred_distri_list.append(dist_lvl)
        pred_scores_list.append(score_lvl) # 这些是原始 logits

    pred_distri = torch.cat(pred_distri_list, dim=1) # Shape: [B, total_A, 4 * reg_max]
    pred_scores = torch.cat(pred_scores_list, dim=1)   # Shape: [B, total_A, nc] (raw logits)

    # --- 2. 生成锚点 ---
    # make_anchors 需要原始的特征图列表来获取 H, W
    anchor_points, stride_tensor = make_anchors(
        yolo_raw_predictions,         # List of [B, C, H, W] tensors
        stride_tensor_from_cfg,       # Strides for each feature level
        grid_cell_offset=0.5
    )
    # anchor_points: [total_A, 2] (xy, 特征图单元尺度)
    # stride_tensor: [total_A, 1] (对应每个锚点的步长)

    # --- 3. 预处理真实目标 (targets_coco_format -> yolo_formatted_targets -> lists for assigner) ---
    # 使用新的工具函数进行格式化
    # 注意：targets_coco_format 是一个批次的标注列表，每个元素是一个字典
    if not targets_coco_format:
        logger.warning("计算YOLO损失时未提供真实目标 (targets_coco_format 为空或None)。")
        # 如果没有目标，理论上损失应为0，或者根据具体情况处理
        # TaskAlignedAssigner 可能无法处理空的 gt_labels_list，需要检查
        # 为简单起见，如果批次中没有任何GT，直接返回0损失
        # （更复杂的处理是允许部分图像无GT，部分有GT）
        # 查验 TaskAlignedAssigner 对完全无GT批次的处理
        num_gt_total = 0
        if targets_coco_format: # 即使是空列表，也要检查内容
             for t in targets_coco_format:
                 if t.get('boxes') is not None and t['boxes'].numel() > 0:
                     num_gt_total += t['boxes'].shape[0]
        if num_gt_total == 0:
            logger.info("批次中所有图像均无真实目标框。YOLO损失计为0。")
            # 仍然需要计算一个依赖于预测的0损失，以允许梯度回传
            loss_cls_placeholder = bce_criterion(pred_scores, torch.zeros_like(pred_scores)).sum() * 0.0
            return loss_cls_placeholder


    # yolo_formatted_targets: List[Tensor[N, 5]] (cls, cx_n, cy_n, w_n, h_n) 归一化
    # sr_image_shape (B, C, H_sr, W_sr) 用于提供归一化所需的图像尺寸
    try:
        yolo_formatted_targets = format_coco_targets_to_yolo(
            targets_coco_format, sr_image_shape, device
        )
    except Exception as e_fmt:
        logger.error(f"格式化 COCO 目标到 YOLO 格式时出错: {e_fmt}", exc_info=True)
        return torch.tensor(0.0, device=device, requires_grad=True)


    # 为 TaskAlignedAssigner 准备列表
    gt_labels_list_assigner = []  # List of Tensors [num_gt_img, 1]
    gt_bboxes_list_assigner = []  # List of Tensors [num_gt_img, 4] (xyxy, 图像尺度)
    mask_gt_list_assigner = []    # List of Tensors [num_gt_img, 1] (bool)

    for i in range(batch_size):
        # yolo_formatted_targets[i] 是 Tensor [num_gt_for_image_i, 5] (cls, cx_n, cy_n, w_n, h_n) 归一化
        gt_info_img_i = yolo_formatted_targets[i]
        if gt_info_img_i.numel() == 0:
            gt_labels_list_assigner.append(torch.empty((0,1), dtype=torch.long, device=device))
            gt_bboxes_list_assigner.append(torch.empty((0,4), dtype=torch.float, device=device))
            mask_gt_list_assigner.append(torch.empty((0,1), dtype=torch.bool, device=device))
            continue

        gt_cls_img_i = gt_info_img_i[:, 0:1].long()          # [N, 1]
        gt_xywh_norm_img_i = gt_info_img_i[:, 1:]            # [N, 4] (归一化 xywh)

        # 转换为 xyxy 格式 (仍然是归一化的)
        gt_xyxy_norm_img_i = xywh2xyxy(gt_xywh_norm_img_i) # [N, 4] (归一化 xyxy)

        # 转换为图像尺度 (绝对坐标)
        gt_xyxy_imgscale_img_i = gt_xyxy_norm_img_i.clone()
        gt_xyxy_imgscale_img_i[:, [0,2]] *= img_w_sr # 使用SR图像的宽高
        gt_xyxy_imgscale_img_i[:, [1,3]] *= img_h_sr

        gt_labels_list_assigner.append(gt_cls_img_i)
        gt_bboxes_list_assigner.append(gt_xyxy_imgscale_img_i)
        mask_gt_list_assigner.append(torch.ones_like(gt_cls_img_i, dtype=torch.bool)) # 假设所有GT都是有效的


    # --- 4. 解码预测边界框 (DFL分布 -> xyxy 特征图单元尺度) ---
    # pred_distri: [B, total_A, 4 * reg_max]
    # anchor_points: [total_A, 2] (xy, 特征图单元尺度)
    # proj_df: [reg_max] (0, 1, ..., reg_max-1)
    def _bbox_decode_local(anchor_pts_in, pred_dist_input, use_dfl_flag, proj_tensor_df, reg_max_val_in):
        # anchor_pts_in: [total_A, 2]
        # pred_dist_input: [B, total_A, 4 * reg_max_val_in]
        if use_dfl_flag:
            b_dec, a_dec, c_dec = pred_dist_input.shape # B, total_A, 4*reg_max
            # pred_dist_input.view: [B, total_A, 4, reg_max]
            # .softmax(3): 在 reg_max 维度上 softmax, 得到每个坐标的概率分布
            # .matmul(proj_tensor_df): 与 [reg_max] 的投影向量相乘, 计算期望值
            # 结果 pred_dist_out 形状: [B, total_A, 4] (ltrb 偏移量)
            pred_dist_out = pred_dist_input.view(b_dec, a_dec, 4, reg_max_val_in).softmax(3).matmul(proj_tensor_df.to(pred_dist_input.dtype))
        else: # reg_max = 1 (或0) 的情况，非 DFL
            # 如果不是 DFL，pred_dist_input 的解释可能不同，这里假设它直接是 ltrb 偏移
            # 或者需要一个不同的解码方式。YOLOv8 通常 reg_max > 1。
            # 为简单起见，如果真的走到这里，我们假设 pred_dist_input 已经是 ltrb 偏移量
            # 但通道数可能不匹配 4*reg_max (如果是4的话)。
            # 假设 pred_dist_input 已经是 [B, total_A, 4]
            if pred_dist_input.shape[-1] == 4 : #直接是 ltrb
                 pred_dist_out = pred_dist_input
            else: #如果还是 4*reg_max 结构但 use_dfl=False,取均值可能不合理
                 logger.warning("bbox_decode_local: use_dfl=False 但 pred_dist 通道数不是4，解码可能不准确。")
                 pred_dist_out = pred_dist_input.view(pred_dist_input.shape[0], pred_dist_input.shape[1], 4, reg_max_val_in).mean(3) # 粗略处理

        # dist2bbox: anchor_points [total_A, 2] 需要扩展为 [B, total_A, 2] 以匹配 pred_dist_out [B, total_A, 4]
        return dist2bbox(pred_dist_out, anchor_pts_in.unsqueeze(0).repeat(pred_dist_out.size(0), 1, 1), xywh=False) # 输出 xyxy

    pred_bboxes_feat_scale = _bbox_decode_local(anchor_points, pred_distri, use_dfl, proj_df, reg_max)
    # pred_bboxes_feat_scale: [B, total_A, 4] (xyxy, 特征图单元尺度)

    # --- 5. 执行目标分配 ---
    # 对于分配器 (assigner)，pd_bboxes (预测框) 需要是图像尺度
    # pred_bboxes_feat_scale 是特征图尺度，需要乘以步长
    # stride_tensor 是 [total_A, 1]，需要扩展为 [B, total_A, 1] 或 [B, total_A, 4]
    # .unsqueeze(0) -> [1, total_A, 1] then repeat or broadcast
    # .unsqueeze(-1) for stride_tensor would make it [total_A,1,1], then broadcasting might be tricky
    # stride_tensor.unsqueeze(0) -> [1, total_A, 1]
    # stride_tensor_expanded = stride_tensor.unsqueeze(0).expand(batch_size, -1, 1)
    pred_bboxes_img_scale_detached = (pred_bboxes_feat_scale.detach() * stride_tensor.unsqueeze(0).to(device))

    # TaskAlignedAssigner 的 anc_points 也期望是图像尺度 (根据其内部 select_candidates_in_gts 的逻辑)
    anchor_points_img_scale_for_assigner = anchor_points * stride_tensor.to(device) # [total_A, 2]

    # pred_scores 是原始 logits, assigner 期望 sigmoid后的分数
    target_labels, target_bboxes_img_scale, target_scores, fg_mask, target_gt_idx = assigner(
        pd_scores=pred_scores.detach().sigmoid(), # [B, total_A, nc]
        pd_bboxes=pred_bboxes_img_scale_detached, # [B, total_A, 4] (xyxy, 图像尺度)
        anc_points=anchor_points_img_scale_for_assigner, # [total_A, 2] (图像尺度)
        gt_labels_list=gt_labels_list_assigner,       # List of [N_gt, 1]
        gt_bboxes_list=gt_bboxes_list_assigner,       # List of [N_gt, 4] (xyxy, 图像尺度)
        mask_gt_list=mask_gt_list_assigner            # List of [N_gt, 1] (bool)
    )
    # fg_mask: [B, total_A] (bool), 标记前景锚点
    # target_bboxes_img_scale: [B, total_A, 4] (xyxy 图像尺度, 仅对 fg_mask 为 True 的位置有意义)
    # target_scores: [B, total_A, nc] (软目标分数, 用于BCE损失)
    # target_labels: [B, total_A] (整数类别索引，背景为 nc)
    # target_gt_idx: [B, total_A] (匹配的 GT 索引)

    # --- 6. 计算损失 ---
    # target_scores_sum 用于归一化分类损失和回归损失的权重 (在BboxLoss内部也可能用到)
    # 这个值应该是正样本对应的目标分数的总和
    target_scores_sum = max(target_scores.sum(), 1.0) # 避免除以零

    loss_cls = torch.tensor(0.0, device=device)
    loss_iou = torch.tensor(0.0, device=device)
    loss_dfl = torch.tensor(0.0, device=device)

    # 分类损失 (BCEWithLogitsLoss)
    # pred_scores 是原始 logits [B, total_A, nc]
    # target_scores 是软标签 [B, total_A, nc]
    loss_cls_terms = bce_criterion(pred_scores, target_scores.to(pred_scores.dtype))
    loss_cls = loss_cls_terms.sum() / target_scores_sum # 对所有项求和后归一化

    num_fg_total = fg_mask.sum() # 所有批次中前景锚点的总数
    if num_fg_total > 0:
        # --- 包围盒损失 (CIoU + DFL) ---
        # 需要为 BboxLoss 准备在特征图尺度上的输入

        # 1. 提取正样本的预测分布
        # pred_distri: [B, total_A, 4 * reg_max]
        # fg_mask: [B, total_A]
        masked_pred_dist = pred_distri[fg_mask] # [N_fg, 4 * reg_max]

        # 2. 提取正样本的预测框 (特征图尺度)
        # pred_bboxes_feat_scale: [B, total_A, 4]
        masked_pred_bboxes_feat = pred_bboxes_feat_scale[fg_mask] # [N_fg, 4]

        # 3. 提取正样本的锚点 (特征图尺度)
        # anchor_points: [total_A, 2]
        # 需要将 anchor_points 扩展到批次维度才能用 fg_mask 索引
        batched_anchor_points = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1) # [B, total_A, 2]
        masked_anchor_points_feat = batched_anchor_points[fg_mask] # [N_fg, 2]

        # 4. 提取正样本的目标框 (并转换为特征图尺度)
        # target_bboxes_img_scale: [B, total_A, 4] (图像尺度)
        masked_target_bboxes_img = target_bboxes_img_scale[fg_mask] # [N_fg, 4] (图像尺度)
        # 获取对应正样本的步长
        # stride_tensor: [total_A, 1]
        batched_stride_tensor = stride_tensor.unsqueeze(0).repeat(batch_size, 1, 1) # [B, total_A, 1]
        masked_strides = batched_stride_tensor[fg_mask] # [N_fg, 1]
        # 除以步长得到特征图尺度的目标框
        masked_target_bboxes_feat = masked_target_bboxes_img / masked_strides # [N_fg, 4] (特征图尺度)

        # 5. 提取正样本的目标分数 (用于 BboxLoss 内部的加权)
        # target_scores: [B, total_A, nc]
        masked_target_scores_for_bbox_loss = target_scores[fg_mask] # [N_fg, nc]

        loss_iou_val, loss_dfl_val = bbox_loss_calculator(
            pred_dist=masked_pred_dist,               # [N_fg, 4 * reg_max]
            pred_bboxes=masked_pred_bboxes_feat,      # [N_fg, 4] (xyxy, 特征图尺度)
            anchor_points=masked_anchor_points_feat,  # [N_fg, 2] (特征图尺度)
            target_bboxes=masked_target_bboxes_feat,  # [N_fg, 4] (xyxy, 特征图尺度)
            target_scores=masked_target_scores_for_bbox_loss, # [N_fg, nc], 用于内部权重
            target_scores_sum=target_scores_sum,      # 全局归一化因子
            fg_mask_from_assigner=None # 输入已是正样本，不需要此掩码
        )
        loss_iou = loss_iou_val
        loss_dfl = loss_dfl_val

    # 应用损失增益
    loss_iou *= box_gain
    loss_dfl *= dfl_gain
    loss_cls *= cls_gain

    # 总损失 = (IoU损失 + DFL损失 + 分类损失) * 批次大小
    # Ultralytics YOLOv8 的损失是每个批次的平均损失，然后乘以 batch_size
    # 这里的 loss_iou, loss_dfl, loss_cls 已经是通过 target_scores_sum 归一化的“平均”损失
    # 所以乘以 batch_size 得到整个批次的损失值，这与 Ultralytics 的做法一致
    total_detection_loss = (loss_iou + loss_dfl + loss_cls) * batch_size

    if torch.isnan(total_detection_loss): # pragma: no cover
        logger.warning("YOLO 检测损失中检测到 NaN。将替换为 0.0。")
        logger.debug(f"NaN 详情: loss_iou={loss_iou.item()}, loss_dfl={loss_dfl.item()}, loss_cls={loss_cls.item()}")
        # 如果发生 NaN，可以记录更多关于输入的详细信息
        total_detection_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # 确保它依赖于预测以进行梯度传播
        if pred_scores.requires_grad:
            total_detection_loss = total_detection_loss + pred_scores.sum() * 0.0


    if logger and logger.level <= logging.DEBUG: # pragma: no cover
        log_str_yolo = (f"YOLO 损失组成 (已乘以增益, 批次总和): "
                        f"IoU={loss_iou.item() * batch_size:.4f} ({loss_iou.item():.4f}/img_avg), "
                        f"DFL={loss_dfl.item() * batch_size:.4f} ({loss_dfl.item():.4f}/img_avg), "
                        f"Cls={loss_cls.item() * batch_size:.4f} ({loss_cls.item():.4f}/img_avg)")
        logger.debug(log_str_yolo)
        if num_fg_total > 0:
            logger.debug(f"正样本锚点数量 (fg_mask.sum()): {num_fg_total.item()}")
            logger.debug(f"目标分数总和 (归一化因子 target_scores_sum): {target_scores_sum.item()}")
        else:
            logger.debug("此批次中未找到用于 YOLO 损失的正样本锚点。")

    return total_detection_loss


def calculate_joint_loss(
    sr_images: torch.Tensor, # 来自 ConditionalSR 的超分图像
    mask_coarse: Optional[torch.Tensor], # 来自 ConditionalSR 的粗粒度掩码
    targets: Optional[List[Dict]],      # COCO 格式的标注列表
    yolo_raw_predictions: Optional[Any], # ConditionalSR.forward 中 'yolo_raw_predictions' 的值
    config: Dict,
    logger: Optional[logging.Logger] = None,
    # 新增: YOLO模型特定参数，由 stage3_finetune_joint.py 准备和传递
    yolo_model_components_for_loss: Optional[Dict] = None,
    precomputed_detection_loss: Optional[Union[torch.Tensor, Dict]] = None # 不再使用此参数
) -> Tuple[torch.Tensor, Dict[str, float]]:

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers(): # pragma: no cover
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    loss_dict: Dict[str, float] = {}
    device = sr_images.device

    # --- 1. 计算检测损失 ---
    detection_weight = config.get('train', {}).get('loss_weights', {}).get('detection', 1.0)
    loss_detection = torch.tensor(0.0, device=device)

    # precomputed_detection_loss 不再被依赖
    # detection_loss_from_wrapper (来自ConditionalSR的返回值) 也应该是 None

    if detection_weight > 0 and yolo_raw_predictions is not None and \
       targets is not None and yolo_model_components_for_loss is not None:
        try:
            # compute_yolo_loss_from_predictions 现在处理 COCO 格式的 targets
            loss_detection = compute_yolo_loss_from_predictions(
                yolo_raw_predictions=yolo_raw_predictions, # YOLO 模型的原始输出
                targets_coco_format=targets,             # COCO 格式的标注列表
                sr_image_shape=sr_images.shape,          # SR 图像的形状 (B,C,H,W)
                device=device,
                config=config,                           # 主配置
                yolo_model_components=yolo_model_components_for_loss, # 包含 stride, nc, reg_max, no, hyp
                logger=logger
            )
        except Exception as e: # pragma: no cover
            if logger:
                logger.error(f"计算 YOLO 损失时发生错误: {e}", exc_info=True)
            loss_detection = torch.tensor(0.0, device=device) # 发生错误时回退

        # 确保 loss_detection 是一个需要梯度的张量
        if not (torch.is_tensor(loss_detection) and loss_detection.requires_grad): # pragma: no cover
            current_val = loss_detection.item() if torch.is_tensor(loss_detection) else float(loss_detection)
            if logger:
                logger.warning(f"计算得到的检测损失不是张量或不需要梯度。值: {current_val}。重置为0。")
            loss_detection = torch.tensor(0.0, device=device, requires_grad=True)
            # 使其依赖于某个需要梯度的输入，以确保计算图的连接
            if isinstance(yolo_raw_predictions, (list, tuple)) and yolo_raw_predictions and \
               isinstance(yolo_raw_predictions[0], torch.Tensor) and yolo_raw_predictions[0].requires_grad:
                loss_detection = loss_detection + yolo_raw_predictions[0].sum() * 0.0
            elif isinstance(yolo_raw_predictions, torch.Tensor) and yolo_raw_predictions.requires_grad:
                 loss_detection = loss_detection + yolo_raw_predictions.sum() * 0.0


    elif detection_weight > 0: # pragma: no cover
        if logger:
            missing_components = []
            if yolo_raw_predictions is None: missing_components.append("yolo_raw_predictions")
            if targets is None: missing_components.append("targets")
            if yolo_model_components_for_loss is None: missing_components.append("yolo_model_components_for_loss")
            if missing_components:
                logger.warning(f"检测损失计算跳过，因为缺少以下组件: {', '.join(missing_components)}。")

    loss_dict["loss_detection"] = loss_detection.item() if torch.is_tensor(loss_detection) else float(loss_detection)

    # --- 2. 计算稀疏度损失 (与目标比率的 MSE) ---
    sparsity_weight = config.get('train', {}).get('loss_weights', {}).get('sparsity', 0.0)
    loss_sparsity = torch.tensor(0.0, device=device)
    actual_sparsity_val = 0.0
    if sparsity_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        # mask_coarse 期望是 (B, 1, H_mask, W_mask)，值在 [0,1] (训练时是软掩码)
        # 计算实际稀疏度 (1的数量比例，或软掩码的均值)
        actual_sparsity_tensor = torch.mean(mask_coarse.float()) # mask_coarse.float() 确保是浮点数
        actual_sparsity_val = actual_sparsity_tensor.item()
        target_sparsity_ratio = config.get('train', {}).get('target_sparsity_ratio', 0.0)
        target_sparsity_tensor = torch.tensor(target_sparsity_ratio, device=device, dtype=actual_sparsity_tensor.dtype)
        loss_sparsity = F.mse_loss(actual_sparsity_tensor, target_sparsity_tensor)
    loss_dict["loss_sparsity"] = loss_sparsity.item() if torch.is_tensor(loss_sparsity) else float(loss_sparsity)
    loss_dict["actual_sparsity"] = actual_sparsity_val

    # --- 3. 计算平滑度损失 (TV Loss on coarse mask) ---
    smoothness_weight = config.get('train', {}).get('loss_weights', {}).get('smoothness', 0.0)
    loss_smooth = torch.tensor(0.0, device=device)
    if smoothness_weight > 0 and mask_coarse is not None and mask_coarse.numel() > 0:
        if mask_coarse.dim() == 4 and mask_coarse.shape[1] == 1: # 期望形状 (B, 1, H_mask, W_mask)
            # 水平方向差分
            dh = torch.abs(mask_coarse[:, :, :-1, :] - mask_coarse[:, :, 1:, :])
            # 垂直方向差分
            dw = torch.abs(mask_coarse[:, :, :, :-1] - mask_coarse[:, :, :, 1:])
            # 对差分求和，然后按批次大小平均 (也可以是总像素数或批次内的像素数)
            # 除以 batch_size 使其与批次大小无关
            loss_smooth = (torch.sum(dh) + torch.sum(dw)) / mask_coarse.size(0)
        else: # pragma: no cover
            if logger:
                logger.warning(f"平滑度损失期望 mask_coarse 的形状为 (B, 1, H, W)，但得到 {mask_coarse.shape}。跳过损失计算。")
    loss_dict["loss_smooth"] = loss_smooth.item() if torch.is_tensor(loss_smooth) else float(loss_smooth)

    # --- 4. 加权求和 ---
    # 确保所有参与损失计算的项都是标量张量
    if not (torch.is_tensor(loss_detection) and loss_detection.numel() == 1): # pragma: no cover
        loss_detection = torch.tensor(loss_detection, device=device, dtype=torch.float32) # 确保类型
    if not (torch.is_tensor(loss_sparsity) and loss_sparsity.numel() == 1): # pragma: no cover
        loss_sparsity = torch.tensor(loss_sparsity, device=device, dtype=torch.float32)
    if not (torch.is_tensor(loss_smooth) and loss_smooth.numel() == 1): # pragma: no cover
        loss_smooth = torch.tensor(loss_smooth, device=device, dtype=torch.float32)

    total_loss = (
        detection_weight * loss_detection +
        sparsity_weight * loss_sparsity +
        smoothness_weight * loss_smooth
    )

    loss_dict["total_loss"] = total_loss.item()

    if logger and hasattr(logger, 'debug') and logger.level <= logging.DEBUG: # pragma: no cover
        det_val = loss_dict.get('loss_detection', 0.0)
        spar_val = loss_dict.get('loss_sparsity', 0.0)
        act_spar_val = loss_dict.get('actual_sparsity', 0.0)
        smooth_val = loss_dict.get('loss_smooth', 0.0)
        total_val = loss_dict.get('total_loss', 0.0)

        log_str = (f"损失 (各分量未加权平均/值): Det={det_val:.4f} Spar={spar_val:.4f} "
                   f"(实际稀疏度={act_spar_val:.4f}) Smooth={smooth_val:.4f}")
        logger.debug(log_str)
        log_str_weighted = (f"损失 (各加权分量之和): wDet={detection_weight * det_val:.4f} "
                            f"wSpar={sparsity_weight * spar_val:.4f} "
                            f"wSmooth={smoothness_weight * smooth_val:.4f} "
                            f"总计={total_val:.4f}")
        logger.debug(log_str_weighted)

    return total_loss, loss_dict