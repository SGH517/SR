import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision # For ops like nms, roi_align, if ever needed here. (Currently not directly by these core funcs)

# Version check for PyTorch (used in make_anchors)
TORCH_1_10 = torch.__version__ >= "1.10.0"

# --------------------------------------------------------------------------------------------------
# Helper Functions (typically from ultralytics.utils.ops or ultralytics.utils.tal)
# --------------------------------------------------------------------------------------------------

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    Args:
        x (torch.Tensor): Bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1 (N, 4) to box2 (M, 4).
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4).
        box2 (torch.Tensor): A tensor of shape (M, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. Default is True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Default is False.
        DIoU (bool, optional): If True, calculate Distance IoU. Default is False.
        CIoU (bool, optional): If True, calculate Complete IoU. Default is False.
        eps (float, optional): A small value to avoid division by zero. Default is 1e-7.
    Returns:
        (torch.Tensor): IoU values, shape (N, M).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1_b1, y1_b1, w1, h1), (x1_b2, y1_b2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1_b1 - w1_, x1_b1 + w1_, y1_b1 - h1_, y1_b1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x1_b2 - w2_, x1_b2 + w2_, y1_b2 - h2_, y1_b2 + h2_
    else:  # x1, y1, x2, y2 format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    # b1_x1 is (N,1), b2_x1.T is (1,M) -> result is (N,M)
    inter_x1 = torch.max(b1_x1, b2_x1.transpose(-1, -2))
    inter_y1 = torch.max(b1_y1, b2_y1.transpose(-1, -2))
    inter_x2 = torch.min(b1_x2, b2_x2.transpose(-1, -2))
    inter_y2 = torch.min(b1_y2, b2_y2.transpose(-1, -2))
    
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter = inter_w * inter_h # Shape (N, M)

    # Union Area
    area1 = w1 * h1 # Shape (N, 1)
    area2 = w2 * h2 # Shape (M, 1)
    union = area1 + area2.transpose(-1, -2) - inter + eps # Shape (N, M)

    # IoU
    iou = inter / union # Shape (N, M)

    if CIoU or DIoU or GIoU:
        # Minimum enclosing box
        cw = torch.max(b1_x2, b2_x2.transpose(-1, -2)) - torch.min(b1_x1, b2_x1.transpose(-1, -2))  # convex width (N,M)
        ch = torch.max(b1_y2, b2_y2.transpose(-1, -2)) - torch.min(b1_y1, b2_y1.transpose(-1, -2))  # convex height (N,M)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared (N,M)
            
            # Center distance squared
            # b1_center_x = (b1_x1 + b1_x2) / 2 (N,1)
            # b2_center_x = (b2_x1 + b2_x2) / 2 (M,1)
            # rho2_term_x = (b2_x1 + b2_x2).T - (b1_x1 + b1_x2) -> (1,M) - (N,1) -> (N,M)
            rho2_term_x = (b2_x1 + b2_x2).transpose(-1, -2) - (b1_x1 + b1_x2)
            rho2_term_y = (b2_y1 + b2_y2).transpose(-1, -2) - (b1_y1 + b1_y2)
            rho2 = (rho2_term_x**2 + rho2_term_y**2) / 4 # (N,M)
            
            if CIoU:
                # w1, h1 are (N,1); w2, h2 are (M,1)
                atan_w1_h1 = torch.atan(w1 / (h1 + eps)) # (N,1)
                atan_w2_h2_T = torch.atan(w2.transpose(-1, -2) / (h2.transpose(-1, -2) + eps)) # (1,M)
                v = (4 / math.pi**2) * torch.pow(atan_w1_h1 - atan_w2_h2_T, 2) # (N,M)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps)) # (N,M)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        
        c_area = cw * ch + eps  # convex area (N,M)
        return iou - (c_area - union) / c_area  # GIoU
        
    return iou  # IoU


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance(ltrb) to box(xywh or xyxy).
    Args:
        distance (Tensor): Distance matrix from anchor to bbox(ltrb). Shape: (N, 4) or (B, N, 4).
        anchor_points (Tensor): Anchor points. Shape: (N, 2) or (B, N, 2).
        xywh (bool): Convert to xywh format. Default is True.
        dim (int): Dimension to split distance. Default is -1.
    Returns:
        (Tensor): Box matrix. Shape: (N, 4) or (B, N, 4).
    """
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points_list, stride_tensor_list = [], []
    assert feats is not None, "Feature maps list cannot be None for anchor generation."
    if not isinstance(feats, (list, tuple)):
        raise TypeError(f"feats must be a list or tuple of tensors, got {type(feats)}")
    if not feats:
        raise ValueError("feats list cannot be empty")
        
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride_val in enumerate(strides):
        try:
            _, _, h, w = feats[i].shape
        except IndexError:
            raise ValueError(f"Feature map at index {i} is not a 4D tensor. Got shape: {feats[i].shape}")

        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        
        # Use indexing='ij' for torch.meshgrid if TORCH_1_10 is True, else 'xy' (older behavior)
        # However, 'ij' is generally preferred for consistency (Height, Width indexing)
        if TORCH_1_10:
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        else:
            # Fallback for older PyTorch, note that meshgrid's default changed.
            # For (H, W) output matching 'ij', if 'xy' is used, inputs might need to be sx, sy.
            # Ultralytics code uses "xy" for older versions, which might imply sx, sy order.
            # However, the standard way to get HxW grid is sy, sx with "ij".
            # Let's stick to "ij" logic and assume TORCH_1_10 or ensure users have it.
            # If strictly needing to match old ultralytics for older torch, sx, sy with "xy" might be needed.
            # For simplicity and modern PyTorch, using "ij" is better.
            sy, sx = torch.meshgrid(sy, sx, indexing="ij") # Sticking to 'ij'

        anchor_points_list.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor_list.append(torch.full((h * w, 1), stride_val.item(), dtype=dtype, device=device)) # Use .item() for stride_val if it's a 0-dim tensor
    return torch.cat(anchor_points_list), torch.cat(stride_tensor_list)


# --------------------------------------------------------------------------------------------------
# TaskAlignedAssigner and its helpers
# --------------------------------------------------------------------------------------------------

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """
    Select the positive anchor center in gt.
    Args:
        xy_centers (Tensor): shape(num_total_anchors, 2)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4) # Assumes gt_bboxes are xyxy
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    # Expand dims for broadcasting
    # gt_bboxes: (bs, n_boxes, 1, 4)
    # xy_centers: (1, 1, n_anchors, 2)
    lt, rb = gt_bboxes.view(bs, n_boxes, 1, 4).chunk(2, 3)  # left-top, right-bottom
    xy_centers_expanded = xy_centers.view(1, 1, n_anchors, 2)
    
    # bbox_deltas: (bs, n_boxes, n_anchors, 4) where 4 is (dx_lt, dy_lt, dx_rb, dy_rb)
    bbox_deltas = torch.cat((xy_centers_expanded - lt, rb - xy_centers_expanded), dim=3)
    return bbox_deltas.amin(3).gt_(eps) # Check if all 4 deltas are > eps


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """
    If an anchor box is assigned to multiple gts, the one with the highest iou will be selected.
    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors) # mask of anchors in GTs & topk
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors) # IoU between Gts and Preds (or anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors) # Index of the GT box assigned to each anchor
        fg_mask (Tensor): shape(bs, num_total_anchors) # Mask of foreground anchors
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors) # Updated mask_pos
    """
    # mask_pos: (bs, n_max_boxes, h*w)
    # overlaps: (bs, n_max_boxes, h*w)
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(1) # Number of GTs an anchor is assigned to
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gts
        # mask_multi_gts: (b, n_max_boxes, h*w)
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w) -> index of GT with max overlap for each anchor
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # Filter by max overlap if multiple GTs
        fg_mask = mask_pos.sum(1) # Re-calculate fg_mask
    # Find each anchor box's corresponding gt box index
    target_gt_idx = mask_pos.argmax(1)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9, use_ciou=False):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes # Background class index
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou_metric = use_ciou # Whether to use CIoU for metric calculation in TAL (RT-DETR specific)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels_list, gt_bboxes_list, mask_gt_list=None):
        """
        Compute assignment of predicted scores and bboxes to ground truth objects.
        Args:
            pd_scores (Tensor): Predicted class scores (post-sigmoid). Shape (bs, num_total_anchors, num_classes).
            pd_bboxes (Tensor): Predicted bboxes (xyxy image scale). Shape (bs, num_total_anchors, 4).
            anc_points (Tensor): Anchor points (xy, feature map scale). Shape (num_total_anchors, 2).
            gt_labels_list (List[Tensor]): List of GT labels per image. Each tensor shape (num_gt_img, 1).
            gt_bboxes_list (List[Tensor]): List of GT bboxes per image (xyxy image scale). Each tensor shape (num_gt_img, 4).
            mask_gt_list (List[Tensor], optional): List of masks indicating valid GTs. Each tensor shape (num_gt_img, 1).
        Returns:
            target_labels (Tensor): Assigned labels for all anchors. Shape (bs, num_total_anchors).
            target_bboxes (Tensor): Assigned bboxes for all anchors (xyxy image scale). Shape (bs, num_total_anchors, 4).
            target_scores (Tensor): Assigned scores for all anchors. Shape (bs, num_total_anchors, num_classes).
            fg_mask (Tensor): Foreground mask for all anchors. Shape (bs, num_total_anchors).
            target_gt_idx (Tensor): Index of matched GT for each positive anchor. Shape (bs, num_total_anchors).
        """
        self.bs = pd_scores.size(0)
        self.n_anchors = pd_scores.size(1) # num_total_anchors

        # Pad gt_labels and gt_bboxes to have the same n_max_boxes for batch processing
        if gt_labels_list is None or not any(len(gt) > 0 for gt in gt_labels_list) : # Check if any image has GTs
             max_num_obj = 0
        else:
            max_num_obj = max(len(gt) for gt in gt_labels_list if gt is not None and len(gt) > 0)

        if max_num_obj == 0: # No ground truth objects in the batch
            device = pd_scores.device
            return (
                torch.full((self.bs, self.n_anchors), self.bg_idx, dtype=torch.long, device=device),
                torch.zeros((self.bs, self.n_anchors, 4), dtype=torch.float, device=device),
                torch.zeros((self.bs, self.n_anchors, self.num_classes), dtype=torch.float, device=device),
                torch.zeros((self.bs, self.n_anchors), dtype=torch.bool, device=device),
                torch.zeros((self.bs, self.n_anchors), dtype=torch.long, device=device), # target_gt_idx
            )

        # Initialize padded tensors
        # These are (bs, max_num_obj, ...)
        padded_gt_labels = torch.full((self.bs, max_num_obj, 1), self.bg_idx, dtype=torch.long, device=pd_scores.device)
        padded_gt_bboxes = torch.zeros((self.bs, max_num_obj, 4), dtype=torch.float, device=pd_scores.device)
        # mask_gt indicates valid GTs after padding (bs, max_num_obj, 1)
        batched_mask_gt = torch.zeros((self.bs, max_num_obj, 1), dtype=torch.bool, device=pd_scores.device)


        for i in range(self.bs):
            num_gt_img = 0
            if gt_labels_list[i] is not None and gt_labels_list[i].numel() > 0:
                 num_gt_img = gt_labels_list[i].shape[0]
                 padded_gt_labels[i, :num_gt_img] = gt_labels_list[i]
                 padded_gt_bboxes[i, :num_gt_img] = gt_bboxes_list[i]
                 if mask_gt_list and mask_gt_list[i] is not None:
                     batched_mask_gt[i, :num_gt_img] = mask_gt_list[i]
                 else: # If no mask_gt_list provided, assume all gts in list are valid
                     batched_mask_gt[i, :num_gt_img] = True


        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, padded_gt_labels, padded_gt_bboxes, anc_points, batched_mask_gt
        ) # mask_pos, align_metric, overlaps are (bs, max_num_obj, n_anchors)

        target_gt_idx, fg_mask, _ = select_highest_overlaps(mask_pos, overlaps, max_num_obj)
        # target_gt_idx, fg_mask are (bs, n_anchors)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            padded_gt_labels, padded_gt_bboxes, target_gt_idx, fg_mask
        )
        # target_labels, target_bboxes, target_scores are (bs, n_anchors, ...)

        # Normalize target_scores by alignment metrics
        # This part is specific to how ultralytics' TAL assigns soft scores
        align_metric *= mask_pos # Apply mask_pos to zero out metrics for non-selected GTs per anchor
        pos_align_metrics = align_metric.amax(dim=1, keepdim=True)  # (bs, 1, n_anchors) -> max align_metric for each anchor
        pos_overlaps = (overlaps * mask_pos).amax(dim=1, keepdim=True)  # (bs, 1, n_anchors) -> max overlap for each anchor (among selected GTs)
        
        # Denominator for normalization: (max_align_metric_for_anchor * max_overlap_for_anchor) / (max_align_metric_for_anchor + eps)
        # This seems to be: (align_metric_for_assigned_GT * overlap_for_assigned_GT) / (max_align_metric_for_anchor + eps)
        # The norm_align_metric from ultralytics is:
        # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # which means, for each anchor, get the (align_metric_gt * pos_overlaps_anchor) / (pos_align_metrics_anchor + eps)
        # then take max over gt dimension. This should be a value per anchor.
        # Let's re-check the target_scores calculation in get_targets, it's one-hot.
        # The scaling seems to be: target_scores *= norm_align_metric
        # This means target_scores become soft based on the quality of alignment.

        # Let's get the alignment metric and overlap for the assigned GT for each positive anchor
        # target_gt_idx is (bs, n_anchors)
        # align_metric is (bs, max_num_obj, n_anchors)
        # overlaps is (bs, max_num_obj, n_anchors)
        
        # Create indices for gathering
        batch_idx_gather = torch.arange(self.bs, device=pd_scores.device).unsqueeze(1).repeat(1, self.n_anchors) # [B, A]
        # For anchors that are fg, their target_gt_idx is valid.
        # We want align_metric[b, target_gt_idx[b,a], a] and overlaps[b, target_gt_idx[b,a], a]
        
        assigned_align_metric = torch.zeros_like(fg_mask, dtype=torch.float) # [B, A]
        assigned_overlaps = torch.zeros_like(fg_mask, dtype=torch.float) # [B, A]

        # Convert fg_mask (counts) to a boolean mask.
        # fg_mask_bool will be True for anchors assigned to at least one GT.
        # Example: tensor([[False, True, True], [True, False, False]])
        fg_mask_bool = fg_mask > 0

        if fg_mask_bool.any():
            # Get the batch and anchor indices where fg_mask_bool is True.
            # positive_batches and positive_anchors will be 1D tensors listing the coordinates.
            # Example: if fg_mask_bool is tensor([[False, True], [True, False]])
            # positive_batches might be tensor([0, 1])
            # positive_anchors might be tensor([1, 0])
            positive_batches, positive_anchors = fg_mask_bool.nonzero(as_tuple=True)

            # target_gt_idx is [B, A]. We need the gt_idx for the positive anchors.
            # target_gt_idx[positive_batches, positive_anchors] will give a 1D tensor [N_fg]
            # containing the gt_idx for each of the N_fg positive anchors.
            gt_indices_for_align = target_gt_idx[positive_batches, positive_anchors]

            # align_metric is [B, max_num_obj, A]
            # overlaps is [B, max_num_obj, A]
            # We need to select from align_metric using:
            # - the batch index of the positive anchor (from positive_batches)
            # - the gt index assigned to that positive anchor (from gt_indices_for_align)
            # - the anchor index of the positive anchor (from positive_anchors)
            # This will result in a 1D tensor of [N_fg] values.
            align_values = align_metric[positive_batches, gt_indices_for_align, positive_anchors]
            overlap_values = overlaps[positive_batches, gt_indices_for_align, positive_anchors]

            # Assign these extracted values to the correct positions in
            # assigned_align_metric and assigned_overlaps using the boolean mask or the N_fg indices.
            assigned_align_metric[fg_mask_bool] = align_values
            assigned_overlaps[fg_mask_bool] = overlap_values

        # Normalize the scores of positive predictions
        # norm_align_metric_per_anchor should be [B, A, 1]
        norm_factor = (assigned_align_metric * assigned_overlaps / (assigned_align_metric.sum(dim=1, keepdim=True).clamp(min=self.eps) ) ).unsqueeze(-1) # Simplified norm based on idea
        # The original norm_align_metric: (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # pos_align_metrics is max over GT dim: (bs, 1, n_anchors)
        # pos_overlaps is max over GT dim: (bs, 1, n_anchors)
        # This means the denominator for normalization is anchor-specific.
        # The original target_scores are one-hot for the assigned class.
        # final_target_scores = target_scores * normalization_score_per_anchor
        
        # Replicating `target_scores = target_scores * norm_align_metric` from original TAL:
        # `align_metric` is (bs, max_num_obj, n_anchors)
        # `pos_overlaps` is effectively `overlaps.amax(dim=1, keepdim=True)` if we consider the best GT's overlap for an anchor
        # `pos_align_metrics` is `align_metric.amax(dim=1, keepdim=True)`
        # This normalization is applied to one-hot target_scores.
        
        # Use the simplified assigned_metrics as the scaling factor for one-hot targets
        # target_scores from get_targets is already one-hot [B, A, NC]
        # We need a scaling factor [B, A, 1]
        
        # Get the align_metric for the actual assigned GT for each anchor
        # This can be done by gathering from align_metric using target_gt_idx
        # align_metric_assigned_gt is [B, N_anchors]
        align_metric_assigned_gt_for_norm = torch.zeros_like(target_gt_idx, dtype=torch.float)
        if fg_mask.any():
             # For each anchor `a` in batch `b`, get `align_metric[b, target_gt_idx[b,a], a]`
            idx_b = torch.arange(self.bs).view(-1,1).repeat(1,self.n_anchors)
            align_metric_assigned_gt_for_norm = align_metric[idx_b, target_gt_idx, torch.arange(self.n_anchors).repeat(self.bs,1)]
        
        # Scale one-hot target_scores by this alignment metric
        target_scores = target_scores * align_metric_assigned_gt_for_norm.unsqueeze(-1) # [B,A,NC] * [B,A,1]

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get initial positive mask based on alignment metrics and top-k selection."""
        # pd_scores: (bs, n_anchors, nc) post-sigmoid
        # pd_bboxes: (bs, n_anchors, 4) xyxy image scale
        # gt_labels: (bs, max_num_obj, 1)
        # gt_bboxes: (bs, max_num_obj, 4) xyxy image scale
        # anc_points: (n_anchors, 2) xy feat scale
        # mask_gt: (bs, max_num_obj, 1) bool, indicates valid GTs

        # align_metric: (bs, n_anchors, max_num_obj) based on original TAL paper (pred vs GT)
        # overlaps: (bs, n_anchors, max_num_obj)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # Transpose to (bs, max_num_obj, n_anchors) for consistency with select_topk_candidates
        align_metric = align_metric.permute(0, 2, 1)
        overlaps = overlaps.permute(0, 2, 1)

        # Get anchors inside GT bboxes: mask_in_gts (bs, max_num_obj, n_anchors)
        # select_candidates_in_gts expects anc_points (feat scale) and gt_bboxes (image scale)
        # This function might need adjustment if anc_points are not scaled to image level first.
        # The provided select_candidates_in_gts expects xy_centers (anc_points) and gt_bboxes (image scale).
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes) # anc_points need to be image scale if gt_bboxes are.
                                                                    # Or gt_bboxes scaled to feature map for anc_points.
                                                                    # The provided select_candidates_in_gts uses anc_points as is, and gt_bboxes.
                                                                    # This implies anc_points should be image scale or gt_bboxes feature scale for this call.
                                                                    # Let's assume for now it handles scales correctly or input is pre-scaled.
                                                                    # The example in v8 loss scales pred_bboxes and anc_points by stride for assigner.
                                                                    # If anc_points are feat_scale, gt_bboxes should be feat_scale for select_candidates_in_gts.
                                                                    # Let's assume anc_points is scaled to image outside.

        # Select top-k candidates: mask_topk (bs, max_num_obj, n_anchors)
        # topk_mask for select_topk_candidates should be (bs, max_num_obj, topk) from mask_gt
        effective_topk_mask = mask_gt.repeat([1, 1, self.topk]) # if mask_gt is (bs,max_num_obj,1)
        
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=effective_topk_mask.bool())
        
        # Merge masks: mask_pos (bs, max_num_obj, n_anchors)
        mask_pos = mask_topk * mask_in_gts * mask_gt # Ensure mask_gt is broadcastable
        
        return mask_pos, align_metric, overlaps


# Inside class TaskAlignedAssigner:
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """
        Compute alignment metric and overlaps.
        pd_scores: (bs, n_anchors, nc) post-sigmoid
        pd_bboxes: (bs, n_anchors, 4) xyxy image scale
        gt_labels: (bs, max_num_obj_padded, 1) <--- This is the padded GT labels
        gt_bboxes: (bs, max_num_obj_padded, 4) xyxy image scale <--- This is the padded GT bboxes
        Returns:
            align_metric (Tensor): (bs, n_anchors, max_num_obj_padded)
            overlaps (Tensor): (bs, n_anchors, max_num_obj_padded)
        """
        # ---- 修改开始 ----
        # Determine current_max_num_obj from the shape of the input gt_labels (or gt_bboxes)
        current_max_num_obj = gt_labels.size(1) # This is the padded dimension for n_max_boxes
        # ---- 修改结束 ----

        overlaps_list = []
        for b in range(self.bs):
            # Slice gt_bboxes and gt_labels for the current batch item up to current_max_num_obj
            # This is technically not needed if current_max_num_obj is already the dimension of gt_bboxes[b]
            # but it's safer if gt_bboxes[b] might be larger for some reason (should not be the case here).
            overlaps_list.append(bbox_iou(pd_bboxes[b], gt_bboxes[b, :current_max_num_obj], xywh=False, CIoU=self.use_ciou_metric))
        overlaps = torch.stack(overlaps_list, dim=0) # (bs, n_anchors, current_max_num_obj)

        # gt_labels_squeezed will be (bs, current_max_num_obj)
        gt_labels_squeezed = gt_labels.squeeze(-1).long()
        # Expand gt_labels_squeezed for gathering: (bs, 1, current_max_num_obj) -> (bs, n_anchors, current_max_num_obj)
        # Use current_max_num_obj for expansion
        gt_cls_idx_expanded = gt_labels_squeezed.unsqueeze(1).expand(-1, self.n_anchors, current_max_num_obj)
        
        bbox_scores = torch.gather(pd_scores, 2, gt_cls_idx_expanded.clamp(0, self.num_classes - 1))
        
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the topk candidates based on the given metrics.
        Args:
            metrics (Tensor): (bs, max_num_obj, n_anchors)
            largest (bool): If True, select largest.
            topk_mask (Tensor, optional): (bs, max_num_obj, topk) boolean mask for valid topk entries.
        Returns:
            (Tensor): (bs, max_num_obj, n_anchors) boolean mask of selected candidates.
        """
        num_anchors = metrics.shape[-1] # n_anchors
        
        # (bs, max_num_obj, topk)
        actual_topk = min(self.topk, num_anchors) # Handle cases where n_anchors < self.topk
        if actual_topk == 0: # No anchors to select from
            return torch.zeros_like(metrics, dtype=torch.bool)

        topk_metrics, topk_idxs = torch.topk(metrics, actual_topk, dim=-1, largest=largest)

        if topk_mask is None:
            # Default: consider all topk metrics > eps as valid
            topk_mask = (topk_metrics.abs().max(-1, keepdim=True)[0] > self.eps).expand(-1, -1, actual_topk)
        else:
            topk_mask = topk_mask[..., :actual_topk] # Ensure topk_mask matches actual_topk

        # Zero out indices that are not valid based on topk_mask
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        
        # Create one-hot encoding for selected anchors
        # F.one_hot output is (bs, max_num_obj, topk, num_anchors)
        # Sum over topk dimension to get (bs, max_num_obj, num_anchors)
        mask_topk = F.one_hot(topk_idxs, num_classes=num_anchors).sum(-2)
        mask_topk = mask_topk.to(metrics.dtype).bool() # Ensure boolean output
        return mask_topk

    # Inside class TaskAlignedAssigner:
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, bboxes, and scores for assigned anchor points.
        Args:
            gt_labels (Tensor): Padded GT labels (bs, max_num_obj, 1).
            gt_bboxes (Tensor): Padded GT bboxes (bs, max_num_obj, 4) (xyxy image scale).
            target_gt_idx (Tensor): Index of assigned GT for each anchor (bs, n_anchors).
            fg_mask (Tensor): Foreground mask for anchors (bs, n_anchors).
                              This fg_mask contains counts from select_highest_overlaps.
        Returns:
            target_labels_assigned (Tensor): (bs, n_anchors)
            target_bboxes_assigned (Tensor): (bs, n_anchors, 4)
            target_scores_assigned (Tensor): (bs, n_anchors, num_classes) (one-hot like)
        """
        batch_ind = torch.arange(self.bs, dtype=torch.long, device=gt_labels.device).unsqueeze(1) # (bs, 1)
        
        target_labels_assigned = gt_labels[batch_ind, target_gt_idx.long()].squeeze(-1) # (bs, n_anchors)

        # ---- 修改开始 ----
        # Condition for background anchors should be where fg_mask (count of assigned GTs) is 0
        background_mask = (fg_mask == 0)

        target_labels_assigned[background_mask] = self.bg_idx # Set background anchors to bg_idx
        # ---- 修改结束 ----

        target_bboxes_assigned = gt_bboxes[batch_ind, target_gt_idx.long()] # (bs, n_anchors, 4)
        # ---- 修改开始 ----
        target_bboxes_assigned[background_mask] = 0 # Zero out bboxes for background anchors
        # ---- 修改结束 ----

        clamped_labels = target_labels_assigned.clamp(0, self.num_classes -1)
        target_scores_assigned = F.one_hot(clamped_labels, num_classes=self.num_classes).float()
        # ---- 修改开始 ----
        target_scores_assigned[background_mask] = 0 # Zero out scores for background
        # ---- 修改结束 ----

        return target_labels_assigned, target_bboxes_assigned, target_scores_assigned

# --------------------------------------------------------------------------------------------------
# BboxLoss Class
# --------------------------------------------------------------------------------------------------
class BboxLoss(nn.Module):
    def __init__(self, reg_max_val, use_dfl=False): # Renamed reg_max to reg_max_val to avoid conflict
        super().__init__()
        self.reg_max = reg_max_val # This should be the actual reg_max (e.g., 16)
                                 # The range of target distances is 0 to reg_max-1
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask_from_assigner):
        """
        Args:
            pred_dist (Tensor): Predicted DFL distribution for positive anchors [N_fg, 4 * reg_max]
                                OR full [B, total_A, 4 * reg_max] if fg_mask_from_assigner is also full.
                                Let's assume inputs are already masked for positive anchors.
                                So, pred_dist is [N_fg, 4 * reg_max]
            pred_bboxes (Tensor): Predicted bboxes for positive anchors [N_fg, 4] (xyxy, feature map scale)
            anchor_points (Tensor): Anchor points for positive anchors [N_fg, 2] (feature map scale)
            target_bboxes (Tensor): Target bboxes for positive anchors [N_fg, 4] (xyxy, feature map scale)
            target_scores (Tensor): Target scores for positive anchors [N_fg, nc] (for weight calculation)
            target_scores_sum (float): Normalization factor.
            fg_mask_from_assigner (Tensor): Not directly used here if inputs are pre-masked, but weight uses it.
                                          If inputs are not pre-masked, this would be [B, total_A]
        Returns:
            loss_iou (Tensor)
            loss_dfl (Tensor)
        """
        # Assume inputs are already for foreground anchors (N_fg elements)
        # Weight for IoU and DFL loss
        # target_scores is [N_fg, nc]. Sum over classes to get weight per anchor.
        weight = target_scores.sum(-1).unsqueeze(-1) # [N_fg, 1]
        num_fg = pred_bboxes.shape[0]
        if num_fg == 0:
            return torch.tensor(0.0, device=pred_dist.device), torch.tensor(0.0, device=pred_dist.device)

        # CIoU loss
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True) # [N_fg]
        loss_iou = ((1.0 - iou.unsqueeze(-1)) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            # target_ltrb: (N_fg, 4) - distances from anchor_points to target_bboxes edges
            # Clamped to [0, self.reg_max - 0.01]
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max) # Pass self.reg_max (e.g. 16)
                                                                                  # bbox2dist will clamp to reg_max-0.01

            # pred_dist is [N_fg, 4 * self.reg_max]
            # Reshape for df_loss: pred_dist_for_df : [N_fg * 4, self.reg_max]
            # target_ltrb_for_df: [N_fg * 4]
            pred_dist_for_df = pred_dist.view(-1, self.reg_max)
            target_ltrb_for_df = target_ltrb.view(-1)
            
            loss_dfl_terms = self._df_loss(pred_dist_for_df, target_ltrb_for_df) # [N_fg * 4, 1]
            # Reshape weight to match: [N_fg, 1] -> [N_fg, 4, 1] -> [N_fg * 4, 1]
            # Each of the 4 coordinates (l,t,r,b) for an anchor shares the same weight.
            weight_for_dfl = weight.repeat(1, 4).view(-1, 1) # [N_fg * 4, 1]

            loss_dfl = (loss_dfl_terms * weight_for_dfl).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    def bbox2dist(self, anchor_points, bbox, reg_max_param): # reg_max_param is e.g. 16
        """Transform bbox(xyxy) to dist(ltrb), clamped to [0, reg_max_param - 0.01]."""
        x1y1, x2y2 = bbox.chunk(2, -1) # bbox is [N, 4]
        # anchor_points is [N, 2]
        return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max_param - 0.01)

    def _df_loss(self, pred_dist, target): # pred_dist [N*4, R], target [N*4] where R=reg_max
        """Distribution Focal Loss (DFL)."""
        tl = target.long()  # target left index
        tr = tl + 1         # target right index
        wl = tr.float() - target  # weight left
        wr = 1.0 - wl             # weight right

        # Clamp indices to be valid for cross_entropy [0, R-1]
        # self.reg_max is, e.g., 16. So valid indices are 0-15.
        tl_clamped = tl.clamp(0, self.reg_max - 1)
        tr_clamped = tr.clamp(0, self.reg_max - 1)
        
        # pred_dist has shape (N_total_coords, self.reg_max)
        # tl/tr have shape (N_total_coords)
        loss_tl = F.cross_entropy(pred_dist, tl_clamped, reduction="none")
        loss_tr = F.cross_entropy(pred_dist, tr_clamped, reduction="none")
        
        return (loss_tl * wl + loss_tr * wr).mean(-1, keepdim=True) # mean over reg_max dim effectively if it was there