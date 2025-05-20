import os
import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, Tuple, Optional, List, Union
import logging

def run_coco_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    annotation_file: str,
    output_dir: str,
    step_or_epoch: Union[int, str],
    logger: Optional[logging.Logger] = None,
    use_hard_mask: bool = True,
    prefix: str = "eval"
) -> Tuple[Dict[str, float], float]:
    """
    运行基于 COCO 的目标检测评估。

    Args:
        model: 要评估的模型 (例如 ConditionalSR)。
        dataloader: 评估数据集的 DataLoader。
        device: 运行设备。
        annotation_file: COCO 格式的 GT 标注文件路径。
        output_dir: 保存检测结果 JSON 文件的目录。
        step_or_epoch: 当前的训练步数或轮数 (用于文件名)。
        logger: 日志记录器。
        use_hard_mask: 推理时是否使用硬掩码 (如果模型支持)。
        prefix: 输出文件名的前缀 (例如 'eval' 或 'final_eval')。

    Returns:
        Tuple containing:
            - map_results (Dict[str, float]): 包含 'map' 和 'map_50' 的字典。
            - average_sparsity (float): 平均掩码稀疏度。
    """
    if logger is None:
        logger = logging.getLogger(__name__) # Fallback logger

    model.eval()
    detection_results = []
    total_sparsity = 0.0
    num_batches = 0
    output_json_path = os.path.join(output_dir, f"{prefix}_results_step_{step_or_epoch}.json")
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Step {step_or_epoch}"):
            # --- Batch Handling (获取图像和 image_id) ---
            lr_images = None
            image_ids = []
            if isinstance(batch, (list, tuple)) and len(batch) >= 2: # Assume (images, targets)
                lr_images, targets = batch[:2]
                if isinstance(targets, list) and all(isinstance(t, dict) and 'image_id' in t for t in targets):
                     image_ids = [t['image_id'].item() if torch.is_tensor(t['image_id']) else t['image_id'] for t in targets]
                else:
                     logger.warning("Targets format incorrect or missing 'image_id'. Using batch index.")
                     image_ids = list(range(num_batches * dataloader.batch_size, (num_batches + 1) * dataloader.batch_size))[:lr_images.size(0)]

            elif torch.is_tensor(batch): # Assume only images
                lr_images = batch
                logger.warning("Eval batch only contains images. Using batch index for image_id.")
                image_ids = list(range(num_batches * dataloader.batch_size, (num_batches + 1) * dataloader.batch_size))[:lr_images.size(0)]
            else:
                logger.warning(f"Unexpected eval batch type: {type(batch)}. Skipping.")
                continue

            if lr_images is None: continue

            lr_images = lr_images.to(device)

            # --- Model Inference ---
            try:
                # Check if model forward accepts hard_mask_inference argument
                import inspect
                sig = inspect.signature(model.forward)
                if 'hard_mask_inference' in sig.parameters:
                    outputs = model(lr_images, hard_mask_inference=use_hard_mask)
                else:
                    outputs = model(lr_images) # Assume model handles eval mode internally if needed
            except Exception as e:
                logger.error(f"Error during model forward pass in evaluation: {e}", exc_info=True)
                continue

            sr_images = outputs.get("sr_image")
            mask_fused = outputs.get("mask_fused") # Use fused mask for sparsity calculation

            # --- Sparsity Calculation ---
            if mask_fused is not None:
                batch_sparsity = torch.mean(mask_fused.float()).item()
                total_sparsity += batch_sparsity * lr_images.size(0) # Weighted by batch size
            else:
                batch_sparsity = 0.0 # Assume fast path if mask is None

            # --- Detection Result Formatting ---
            # Try getting results from 'detection_results' first (ConditionalSR format)
            detections_list = outputs.get("detection_results")

            # If not found, try calling detector directly (if model has one)
            if detections_list is None and hasattr(model, 'detector') and model.detector is not None and sr_images is not None:
                 try:
                     detections_list = model.detector(sr_images) # Assume detector returns list of dicts
                 except Exception as e:
                     logger.error(f"Error calling model.detector during evaluation: {e}", exc_info=True)
                     detections_list = None

            if isinstance(detections_list, list):
                for i, dets in enumerate(detections_list):
                    if not isinstance(dets, dict) or not all(k in dets for k in ['boxes', 'scores', 'labels']):
                        logger.warning(f"Skipping invalid detection result format: {type(dets)}")
                        continue
                    image_id = image_ids[i] if i < len(image_ids) else num_batches * dataloader.batch_size + i # Fallback image_id
                    boxes = dets['boxes'].cpu().numpy()
                    scores = dets['scores'].cpu().numpy()
                    labels = dets['labels'].cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        # COCO format: [x_min, y_min, width, height]
                        coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        detection_results.append({
                            "image_id": int(image_id),
                            "category_id": int(label), # Ensure category_id is int
                            "bbox": [round(coord, 2) for coord in coco_box],
                            "score": round(float(score), 3)
                        })
            elif detections_list is not None:
                 logger.warning(f"Unexpected format for detections_list: {type(detections_list)}")


            num_batches += 1

    # --- COCO Evaluation ---
    map_results = {"map": 0.0, "map_50": 0.0}
    if not detection_results:
        logger.warning("No detection results generated during evaluation.")
    elif not os.path.exists(annotation_file):
         logger.error(f"Annotation file not found at {annotation_file}. Cannot perform COCO evaluation.")
    else:
        try:
            logger.info(f"Saving detection results to {output_json_path}...")
            with open(output_json_path, 'w') as f:
                json.dump(detection_results, f)

            coco_gt = COCO(annotation_file)
            coco_dt = coco_gt.loadRes(output_json_path)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            map_stats = coco_eval.stats
            map_results = {"map": map_stats[0], "map_50": map_stats[1]}
        except FileNotFoundError: # Should be caught above, but double-check
            logger.error(f"Annotation file not found at {annotation_file} for evaluation.")
        except Exception as e:
            logger.error(f"Error during COCO evaluation: {e}", exc_info=True)

    total_images = num_batches * dataloader.batch_size # Approximate, might be less in last batch
    average_sparsity = total_sparsity / total_images if total_images > 0 else 0.0
    logger.info(f"Evaluation Step {step_or_epoch}: mAP50={map_results.get('map_50', 0.0):.4f}, mAP={map_results.get('map', 0.0):.4f}, Sparsity={average_sparsity:.4f}")

    model.train() # Set model back to training mode
    return map_results, average_sparsity