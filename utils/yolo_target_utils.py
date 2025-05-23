# utils/yolo_target_utils.py
import torch
from typing import List, Tuple, Dict, Optional

def format_coco_targets_to_yolo(coco_targets_batch: List[Dict],
                                image_shape: Tuple[int, ...],
                                device: torch.device) -> List[torch.Tensor]:
    """
    将一批 COCO 格式的标注转换为 YOLO 格式的标注。

    参数:
        coco_targets_batch (list): 一个字典列表，每个字典代表一个图像的 COCO 格式标注
                                  (键: 'boxes', 'labels')。
                                  'boxes' 是绝对坐标的 [x_min, y_min, width, height]。
        image_shape (tuple): 输入图像张量的形状 (B, C, H, W) 或单个图像的 (H, W)。
                              通常使用 sr_images.shape。
        device (torch.device): 目标设备。

    返回:
        list: 张量列表，每个张量形状为 [N, 5] (class_idx, x_center_norm, y_center_norm, w_norm, h_norm)，
              对应一个图像中的 N 个对象。
    """
    if not image_shape or len(image_shape) < 2:
        raise ValueError(f"image_shape 必须至少包含高度和宽度信息, 得到: {image_shape}")

    img_h, img_w = image_shape[-2:] # 从图像形状中获取高和宽
    yolo_formatted_targets = []

    if not isinstance(coco_targets_batch, list):
        # 如果输入不是列表，可能表示单张图片的标注或需要调整的格式
        # 为了函数的一致性，我们期望它是一个列表
        # 此处可以添加警告或错误处理
        # For now, assume it's an error or needs to be wrapped in a list by the caller
        raise TypeError(f"coco_targets_batch 期望是一个字典列表, 得到: {type(coco_targets_batch)}")


    for target_dict in coco_targets_batch:
        if not isinstance(target_dict, dict):
            # 如果列表中的元素不是字典，也需要处理
            yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
            # 可以添加日志警告
            # logger.warning(f"coco_targets_batch 中的元素不是字典: {target_dict}")
            continue

        boxes_abs_coco = target_dict.get('boxes')
        labels = target_dict.get('labels')

        if boxes_abs_coco is None or labels is None or \
           not isinstance(boxes_abs_coco, torch.Tensor) or not isinstance(labels, torch.Tensor) or \
           boxes_abs_coco.numel() == 0:
            yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
            continue

        # 确保 boxes 和 labels 在正确的设备上
        boxes_abs_coco = boxes_abs_coco.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.int64)


        # 将 COCO 的 [x_min, y_min, w, h] 转换为 [x_center, y_center, w, h]
        boxes_xywh_abs = torch.zeros_like(boxes_abs_coco)
        boxes_xywh_abs[:, 0] = boxes_abs_coco[:, 0] + boxes_abs_coco[:, 2] / 2
        boxes_xywh_abs[:, 1] = boxes_abs_coco[:, 1] + boxes_abs_coco[:, 3] / 2
        boxes_xywh_abs[:, 2] = boxes_abs_coco[:, 2]
        boxes_xywh_abs[:, 3] = boxes_abs_coco[:, 3]

        # 归一化
        boxes_xywh_norm = boxes_xywh_abs.clone()
        if img_w > 0:
            boxes_xywh_norm[:, [0, 2]] /= img_w
        else:
            boxes_xywh_norm[:, [0, 2]] = 0 # 避免除以零
        if img_h > 0:
            boxes_xywh_norm[:, [1, 3]] /= img_h
        else:
            boxes_xywh_norm[:, [1, 3]] = 0 # 避免除以零

        boxes_xywh_norm[:, 0:4] = torch.clamp(boxes_xywh_norm[:, 0:4], min=0.0, max=1.0) # 限制在[0,1]范围

        # 过滤掉过小的边框
        # 阈值可以根据需要调整，或者从配置中读取
        min_box_size_normalized = 1e-4
        valid_indices = (boxes_xywh_norm[:, 2] > min_box_size_normalized) & \
                        (boxes_xywh_norm[:, 3] > min_box_size_normalized)

        if not valid_indices.all():
            boxes_xywh_norm = boxes_xywh_norm[valid_indices]
            labels_filtered = labels[valid_indices]
        else:
            labels_filtered = labels

        if boxes_xywh_norm.numel() == 0:
            yolo_formatted_targets.append(torch.empty((0, 5), device=device, dtype=torch.float32))
            continue

        # YOLO 格式: [class_idx, x_center_norm, y_center_norm, w_norm, h_norm]
        yolo_target_for_image = torch.cat((labels_filtered.float().unsqueeze(1), boxes_xywh_norm), dim=1)
        yolo_formatted_targets.append(yolo_target_for_image)

    return yolo_formatted_targets