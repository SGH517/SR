# utils/evaluation_utils.py
import os
import json
import torch
from torch.utils.data import DataLoader # 确保导入 DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, Tuple, Optional, List, Union, Any # Any 用于模型输出
import logging
import math # 用于 isnan 和 isinf 检查
import inspect # 用于模型 forward 签名检查

# 确保 logger 已设置，如果此文件独立运行或早期导入
logger_eval = logging.getLogger(__name__) # 使用特定名称避免与全局logger冲突
if not logger_eval.hasHandlers(): # 检查是否有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 该模块提供评估相关的工具函数，例如运行COCO评估。

def run_coco_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader, # 类型提示
    device: torch.device,
    annotation_file: str, # COCO GT 标注文件路径
    output_dir: str,      # 保存检测结果JSON文件的目录
    step_or_epoch: Union[int, str], # 用于文件名
    logger: Optional[logging.Logger] = None, # 允许传递外部 logger
    use_hard_mask: bool = True, # 推理时是否使用硬掩码
    prefix: str = "eval" # 结果文件的前缀
) -> Tuple[Dict[str, float], float]:
    """
    运行基于 COCO 的目标检测评估。
    确保所有用于 JSON 序列化的数值数据都是 Python 原生类型。
    """
    eval_logger = logger if logger else logger_eval # 使用传入的logger或此模块的logger

    model.eval() # 设置模型为评估模式
    detection_results_for_coco = []
    total_mask_sparsity_accumulator = 0.0
    num_images_processed_for_sparsity = 0

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 为当前评估创建一个唯一的JSON文件名
    output_json_path = os.path.join(output_dir, f"{prefix}_detections_step_{step_or_epoch}.json")

    with torch.no_grad():
        for batch_idx, batch_content in enumerate(tqdm(dataloader, desc=f"评估步骤 {step_or_epoch}")):
            lr_images_batch: Optional[torch.Tensor] = None
            targets_in_batch: Optional[List[Dict]] = None # 原始targets，主要用于获取 image_id

            # 从 DataLoader 中解包数据
            # DetectionDataset 的 collate_fn 返回 (images_tensor, targets_list)
            if isinstance(batch_content, (list, tuple)) and len(batch_content) == 2:
                lr_images_batch, targets_in_batch = batch_content
            elif torch.is_tensor(batch_content): # 仅包含图像的情况 (不常见于COCO评估)
                lr_images_batch = batch_content
                eval_logger.warning(f"批次 {batch_idx}: 评估批次仅包含图像。将缺少 image_id 用于COCO评估。")
            else:
                eval_logger.warning(f"批次 {batch_idx}: 未预期的评估批次类型: {type(batch_content)}。跳过此批次。")
                continue

            if lr_images_batch is None or lr_images_batch.numel() == 0:
                eval_logger.warning(f"批次 {batch_idx}: lr_images_batch 为 None 或为空。跳过此批次。")
                continue

            # 获取当前批次的 image_ids
            # targets_in_batch 是一个字典列表，每个字典包含 'image_id' 等
            image_ids_for_current_batch: List[int] = []
            if targets_in_batch and isinstance(targets_in_batch, list):
                for target_item in targets_in_batch:
                    if isinstance(target_item, dict) and 'image_id' in target_item:
                        img_id_tensor_or_val = target_item['image_id']
                        if torch.is_tensor(img_id_tensor_or_val):
                            image_ids_for_current_batch.append(img_id_tensor_or_val.item())
                        else: # 假设已经是 int 或可以转换为 int
                            try:
                                image_ids_for_current_batch.append(int(img_id_tensor_or_val))
                            except ValueError:
                                eval_logger.warning(f"批次 {batch_idx}: 无法将 image_id '{img_id_tensor_or_val}' 转换为整数。")
                    else:
                        eval_logger.warning(f"批次 {batch_idx}: 目标项格式不正确或缺少 'image_id'。")
            else: # 如果 targets_in_batch 不可用，我们无法准确获取 image_id
                 eval_logger.warning(f"批次 {batch_idx}: 无法从 targets_in_batch 获取 image_id。COCO评估可能不准确。")
                 # 可以尝试生成基于索引的伪ID，但这对于COCO评估没有意义
                 # image_ids_for_current_batch = [ some_fallback_id_logic ]


            lr_images_batch = lr_images_batch.to(device)

            # 模型前向传播
            model_outputs: Dict[str, Any]
            try:
                # 检查 ConditionalSR.forward 是否接受 hard_mask_inference 参数
                sig = inspect.signature(model.forward)
                if 'hard_mask_inference' in sig.parameters:
                    model_outputs = model(lr_images_batch, hard_mask_inference=use_hard_mask)
                else: # 旧模型可能没有此参数
                    model_outputs = model(lr_images_batch)
            except Exception as e:
                eval_logger.error(f"批次 {batch_idx}: 模型前向传播时发生错误: {e}", exc_info=True)
                continue # 跳过此批次

            # 从模型输出中获取所需信息
            # sr_images_output = model_outputs.get("sr_image") # 超分图像，检测器已在其上运行
            mask_fused_output = model_outputs.get("mask_fused") # 上采样后的融合掩码 (B,1,H_sr,W_sr)
            # yolo_raw_predictions 在推理模式下现在是格式化后的检测结果列表
            detections_list_from_model = model_outputs.get("yolo_raw_predictions")

            # 计算并累加掩码稀疏度 (1 - 掩码均值，表示选择 SR_Fast 的比例)
            if mask_fused_output is not None and mask_fused_output.numel() > 0:
                try:
                    # mask_fused_output 的值在 [0,1]，0 表示 SR_Fast, 1 表示 SR_Quality
                    # 稀疏度 = 1 的比例 (Quality路径)
                    # 或者，如果想衡量 "节约" (Fast路径)，可以用 1 - mean
                    batch_sparsity_quality_path = torch.mean(mask_fused_output.float()).item()
                    # 我们通常关心 "节约了多少"，即 SR_Fast 的使用比例
                    # 但如果 "sparsity" 指的是 "mask中1的数量少"，则直接用 mean(mask)
                    # 这里我们假设 "sparsity" 指的是 "mask中0的数量多"，即 (1-mean(mask))
                    # 或者更直接地，记录 "quality_path_usage_ratio" = mean(mask)
                    total_mask_sparsity_accumulator += batch_sparsity_quality_path * lr_images_batch.size(0) # 加权平均
                    num_images_processed_for_sparsity += lr_images_batch.size(0)
                except Exception as e_sparsity:
                    eval_logger.warning(f"批次 {batch_idx}: 计算稀疏度时出错: {e_sparsity}。"
                                       f"掩码形状: {mask_fused_output.shape if mask_fused_output is not None else 'None'}")

            # 处理检测结果
            if detections_list_from_model is None:
                eval_logger.warning(f"批次 {batch_idx}: 模型输出中 'yolo_raw_predictions' (检测结果) 为 None。")
                # 即使没有检测结果，也需要为该批次中的图像添加空条目吗？
                # COCOeval 通常能处理图片有GT但无预测的情况。
                # 为确保 image_id 对应，如果 image_ids_for_current_batch 非空，可以考虑添加空预测
                # 但如果 detections_list_from_model 为 None，意味着检测器完全没运行或失败
                continue # 跳过此批次的检测结果处理


            if not isinstance(detections_list_from_model, list):
                eval_logger.warning(f"批次 {batch_idx}: 'yolo_raw_predictions' 期望是列表，但得到 {type(detections_list_from_model)}。")
                continue

            # detections_list_from_model 是一个列表，每个元素对应批次中的一张图像
            # 每个元素是一个字典 {'boxes': tensor, 'scores': tensor, 'labels': tensor}
            if len(detections_list_from_model) != lr_images_batch.size(0):
                eval_logger.warning(f"批次 {batch_idx}: 检测结果数量 ({len(detections_list_from_model)}) "
                                   f"与批次中的图像数量 ({lr_images_batch.size(0)}) 不匹配。")
                # 尝试处理，但可能会出错
                # continue

            for i, dets_per_image_dict in enumerate(detections_list_from_model):
                current_image_id_val: Optional[int] = None
                if i < len(image_ids_for_current_batch):
                    current_image_id_val = image_ids_for_current_batch[i]
                else:
                    eval_logger.warning(f"批次 {batch_idx}, 图像索引 {i}: 无法获取对应的 image_id。"
                                       "此图像的检测结果将无法用于 COCO 评估。")
                    # 如果没有 image_id，则无法将其添加到 COCO 结果中
                    continue

                if not isinstance(dets_per_image_dict, dict) or \
                   not all(k in dets_per_image_dict for k in ['boxes', 'scores', 'labels']):
                    eval_logger.warning(f"批次 {batch_idx}, 图像ID {current_image_id_val}: "
                                       f"检测结果字典格式无效: {type(dets_per_image_dict)}。跳过此图像的检测。")
                    continue

                # 获取单张图像的检测结果 (已经是CPU上的张量)
                boxes_xyxy_cpu = dets_per_image_dict['boxes']   # [N, 4] xyxy
                scores_cpu = dets_per_image_dict['scores'] # [N]
                labels_cpu = dets_per_image_dict['labels'] # [N]

                # 转换为 COCO bbox 格式 [x_min, y_min, width, height]
                # 并确保数据类型正确以进行 JSON 序列化
                for detection_idx in range(boxes_xyxy_cpu.shape[0]):
                    box_xyxy = boxes_xyxy_cpu[detection_idx].tolist() # tolist() 转换为 python list of floats
                    score = scores_cpu[detection_idx].item()    # .item() 转换为 python float
                    label = labels_cpu[detection_idx].item()    # .item() 转换为 python int

                    x_min, y_min, x_max, y_max = box_xyxy
                    width = x_max - x_min
                    height = y_max - y_min

                    # 进行基本的有效性检查
                    if width <= 0 or height <= 0 or score < 0 or score > 1:
                        eval_logger.debug(f"批次 {batch_idx}, 图像ID {current_image_id_val}, 检测索引 {detection_idx}: "
                                         f"无效的边框尺寸 (w={width:.2f}, h={height:.2f}) 或分数 (s={score:.3f})。跳过此检测。")
                        continue
                    if math.isnan(x_min) or math.isnan(y_min) or math.isnan(width) or math.isnan(height) or math.isnan(score):
                        eval_logger.debug(f"批次 {batch_idx}, 图像ID {current_image_id_val}, 检测索引 {detection_idx}: "
                                         f"检测结果中存在 NaN 值。跳过此检测。")
                        continue


                    coco_bbox = [
                        round(x_min, 3),
                        round(y_min, 3),
                        round(width, 3),
                        round(height, 3)
                    ]

                    detection_results_for_coco.append({
                        "image_id": int(current_image_id_val), # 确保是 int
                        "category_id": int(label),          # 确保是 int
                        "bbox": coco_bbox,
                        "score": round(score, 5)            # 分数保留多位小数
                    })

    # --- COCO 评估 ---
    map_results_dict = {"map": 0.0, "map_50": 0.0, "map_75": 0.0} # 初始化默认值

    if not detection_results_for_coco:
        eval_logger.warning("没有生成有效的检测结果，无法进行 COCO 评估。")
    elif not os.path.exists(annotation_file):
         eval_logger.error(f"COCO GT 标注文件在 {annotation_file} 未找到。无法进行 COCO 评估。")
    else:
        try:
            eval_logger.info(f"正在将 {len(detection_results_for_coco)} 条检测结果保存到 {output_json_path}...")
            with open(output_json_path, 'w') as f:
                json.dump(detection_results_for_coco, f, indent=2) # 使用 indent=2 减小文件大小

            coco_gt = COCO(annotation_file) # 加载真实标注
            if not coco_gt.dataset.get('images'):
                eval_logger.error(f"COCO GT 文件 {annotation_file} 已加载，但不包含 'images'。评估无法进行。")
            else:
                # 加载预测结果
                # 如果 detection_results_for_coco 为空，coco_gt.loadRes([]) 是安全的
                coco_dt = coco_gt.loadRes(detection_results_for_coco if detection_results_for_coco else [])

                if not coco_dt.dataset.get('annotations') and detection_results_for_coco:
                    eval_logger.warning(f"从内存加载检测结果后，COCOeval 的注解为空，即使原始预测非空。mAP 可能为0。")

                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox') # 'bbox' 表示进行边界框检测评估
                coco_eval.evaluate()    # 执行评估 (计算每个类别、每个IoU阈值下的TP, FP, FN)
                coco_eval.accumulate()  # 累积评估结果 (计算PR曲线等)
                coco_eval.summarize()   # 打印标准的COCO评估摘要 (mAP, AP50, AP75 等)

                map_stats = coco_eval.stats # stats 是一个包含12个指标的numpy数组
                if map_stats is not None and len(map_stats) >= 2: # 通常有12个指标
                    map_results_dict = {
                        "map": round(float(map_stats[0]), 5),      # AP @ IoU=0.50:0.95 area=all maxDets=100
                        "map_50": round(float(map_stats[1]), 5),   # AP @ IoU=0.50  area=all maxDets=100
                        "map_75": round(float(map_stats[2]), 5)    # AP @ IoU=0.75  area=all maxDets=100
                        # 可以按需添加更多指标:
                        # "map_small": float(map_stats[3]),
                        # "map_medium": float(map_stats[4]),
                        # "map_large": float(map_stats[5]),
                    }
                else:
                    eval_logger.error("COCOeval.stats 为 None 或长度不足。mAP 结果将为0。")
        except FileNotFoundError: # 再次检查，尽管上面已检查
            eval_logger.error(f"COCO GT 标注文件在 {annotation_file} 未找到 (在 COCOeval 设置期间)。")
        except Exception as e_coco:
            eval_logger.error(f"COCO 评估过程中发生错误: {e_coco}", exc_info=True)

    # 计算平均稀疏度 (这里是 quality path usage ratio)
    average_quality_path_usage = total_mask_sparsity_accumulator / num_images_processed_for_sparsity \
                                 if num_images_processed_for_sparsity > 0 else 0.0

    eval_logger.info(f"评估步骤 {step_or_epoch} 完成: "
                     f"mAP@.5:.95={map_results_dict.get('map', 0.0):.4f}, "
                     f"mAP@.5={map_results_dict.get('map_50', 0.0):.4f}, "
                     f"Quality路径使用率 (1-稀疏度)={average_quality_path_usage:.4f}")

    model.train() # 恢复模型到训练模式 (如果后续还有操作)
    return map_results_dict, average_quality_path_usage