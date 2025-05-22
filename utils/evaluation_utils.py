import os
import json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, Tuple, Optional, List, Union
import logging
import math # Added for isnan and isinf checks
import inspect # Added for model.forward signature check

# Ensure logger is set up if this file is run standalone or imported early
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# 该模块提供评估相关的工具函数，例如运行COCO评估。

def run_coco_evaluation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    annotation_file: str,
    output_dir: str,
    step_or_epoch: Union[int, str],
    logger: Optional[logging.Logger] = None, # Allow passing an external logger
    use_hard_mask: bool = True,
    prefix: str = "eval"
) -> Tuple[Dict[str, float], float]:
    """
    运行基于 COCO 的目标检测评估。
    Ensures all numerical data stored for JSON serialization are Python native types.
    """
    if logger is None:
        logger = logging.getLogger(__name__) # Use local logger if none provided

    model.eval()
    detection_results = []
    total_sparsity = 0.0
    num_batches_processed = 0 # More accurate naming
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{prefix}_results_step_{step_or_epoch}.json")

    with torch.no_grad():
        for batch_idx, batch_content in enumerate(tqdm(dataloader, desc=f"Evaluating Step {step_or_epoch}")):
            lr_images = None
            image_ids_for_batch = []
            
            if isinstance(batch_content, (list, tuple)) and len(batch_content) >= 2:
                lr_images, targets_in_batch = batch_content[:2]
                if isinstance(targets_in_batch, list) and all(isinstance(t, dict) and 'image_id' in t for t in targets_in_batch):
                    image_ids_for_batch = [
                        t['image_id'].item() if torch.is_tensor(t['image_id']) else t['image_id'] 
                        for t in targets_in_batch
                    ]
                else:
                    logger.warning(f"Batch {batch_idx}: Targets format incorrect or missing 'image_id'. Using fallback indexing.")
                    if lr_images is not None and hasattr(lr_images, 'size') and lr_images.size(0) > 0:
                         image_ids_for_batch = list(range(num_batches_processed * dataloader.batch_size, (num_batches_processed + 1) * dataloader.batch_size))[:lr_images.size(0)]
                    else:
                         image_ids_for_batch = []


            elif torch.is_tensor(batch_content):
                lr_images = batch_content
                logger.warning(f"Batch {batch_idx}: Eval batch only contains images. Using fallback indexing for image_id.")
                image_ids_for_batch = list(range(num_batches_processed * dataloader.batch_size, (num_batches_processed + 1) * dataloader.batch_size))[:lr_images.size(0)]
            else:
                logger.warning(f"Batch {batch_idx}: Unexpected eval batch type: {type(batch_content)}. Skipping.")
                continue

            if lr_images is None or lr_images.numel() == 0:
                logger.warning(f"Batch {batch_idx}: lr_images is None or empty. Skipping batch.")
                continue
            
            if not image_ids_for_batch and lr_images.size(0) > 0 : 
                 logger.warning(f"Batch {batch_idx}: image_ids_for_batch is empty despite images present. Using fallback batch indexing.")
                 image_ids_for_batch = list(range(num_batches_processed * dataloader.batch_size, (num_batches_processed + 1) * dataloader.batch_size))[:lr_images.size(0)]


            lr_images = lr_images.to(device)

            try:
                sig = inspect.signature(model.forward)
                if 'hard_mask_inference' in sig.parameters:
                    outputs = model(lr_images, hard_mask_inference=use_hard_mask)
                else:
                    outputs = model(lr_images)
            except Exception as e:
                logger.error(f"Batch {batch_idx}: Error during model forward pass: {e}", exc_info=True)
                continue

            sr_images_output = outputs.get("sr_image")
            mask_fused = outputs.get("mask_fused")

            if mask_fused is not None and mask_fused.numel() > 0:
                try:
                    batch_sparsity = torch.mean(mask_fused.float()).item()
                    total_sparsity += batch_sparsity * lr_images.size(0)
                except Exception as e:
                    logger.warning(f"Batch {batch_idx}: Error calculating sparsity: {e}. Mask shape: {mask_fused.shape if mask_fused is not None else 'None'}")
            
            detections_list = outputs.get("yolo_raw_predictions")

            if detections_list is None and hasattr(model, 'detector') and model.detector is not None and sr_images_output is not None:
                logger.info(f"Batch {batch_idx}: yolo_raw_predictions was None. Calling model.detector directly.")
                try:
                    raw_detector_output, _ = model.detector(sr_images_output)
                    detections_list = raw_detector_output
                except Exception as e:
                    logger.error(f"Batch {batch_idx}: Error calling model.detector: {e}", exc_info=True)
                    detections_list = None

            if isinstance(detections_list, list):
                for i, dets_per_image in enumerate(detections_list): 
                    if not isinstance(dets_per_image, dict) or not all(k in dets_per_image for k in ['boxes', 'scores', 'labels']):
                        logger.warning(f"Batch {batch_idx}, Image index {i}: Invalid detection result format: {type(dets_per_image)}. Skipping.")
                        continue
                    
                    current_image_id_val = 0
                    if i < len(image_ids_for_batch):
                        current_image_id_val = image_ids_for_batch[i]
                    else:
                        logger.warning(f"Batch {batch_idx}, Image index {i}: Ran out of image_ids. Using fallback. This might indicate an issue.")
                        current_image_id_val = num_batches_processed * dataloader.batch_size + i
                    
                    np_boxes = dets_per_image['boxes'].cpu().numpy()
                    np_scores = dets_per_image['scores'].cpu().numpy()
                    np_labels = dets_per_image['labels'].cpu().numpy()

                    for detection_idx in range(np_boxes.shape[0]):
                        box_np_single = np_boxes[detection_idx]
                        score_np_single = np_scores[detection_idx]
                        label_np_single = np_labels[detection_idx]

                        if box_np_single.shape != (4,):
                            logger.warning(f"Batch {batch_idx}, ImgID {current_image_id_val}, Det {detection_idx}: Invalid box shape {box_np_single.shape}. Skipping.")
                            continue

                        b0_py = box_np_single[0].item()
                        b1_py = box_np_single[1].item()
                        b2_py = box_np_single[2].item()
                        b3_py = box_np_single[3].item()

                        if any(math.isnan(c) or math.isinf(c) for c in [b0_py, b1_py, b2_py, b3_py]):
                            logger.warning(f"Batch {batch_idx}, ImgID {current_image_id_val}, Det {detection_idx}: NaN/Inf in raw bbox coords {box_np_single}. Skipping.")
                            continue
                        
                        width_py = b2_py - b0_py
                        height_py = b3_py - b1_py

                        if math.isnan(width_py) or math.isinf(width_py) or \
                           math.isnan(height_py) or math.isinf(height_py) or \
                           width_py <= 0 or height_py <= 0: # Width/height should be positive
                            logger.warning(f"Batch {batch_idx}, ImgID {current_image_id_val}, Det {detection_idx}: Invalid W/H ({width_py:.2f},{height_py:.2f}) from box {box_np_single}. Skipping.")
                            continue
                        
                        coco_bbox_final = [
                            round(b0_py, 2),    # x_min
                            round(b1_py, 2),    # y_min
                            round(width_py, 2), # width
                            round(height_py, 2) # height
                        ]
                        
                        score_py_item = score_np_single.item()
                        if math.isnan(score_py_item) or math.isinf(score_py_item):
                            logger.warning(f"Batch {batch_idx}, ImgID {current_image_id_val}, Det {detection_idx}: NaN/Inf score {score_np_single}. Skipping.")
                            continue
                        score_final = round(score_py_item, 3)
                        
                        label_final = label_np_single.item()

                        detection_results.append({
                            "image_id": int(current_image_id_val), 
                            "category_id": int(label_final), 
                            "bbox": coco_bbox_final,    
                            "score": score_final        
                        })
            elif detections_list is not None:
                 logger.warning(f"Batch {batch_idx}: Unexpected format for detections_list: {type(detections_list)}")

            num_batches_processed += 1

    map_results = {"map": 0.0, "map_50": 0.0}
    if not detection_results:
        logger.warning("No valid detection results were generated to save to JSON or evaluate.")
    elif not os.path.exists(annotation_file):
         logger.error(f"Annotation file not found at {annotation_file}. Cannot perform COCO evaluation.")
    else:
        try:
            logger.info(f"Saving {len(detection_results)} detection results to {output_json_path}...")
            with open(output_json_path, 'w') as f:
                json.dump(detection_results, f, indent=4)

            coco_gt = COCO(annotation_file)
            if not coco_gt.dataset.get('images'): # Check if GT images list is empty
                logger.error(f"COCO ground truth file {annotation_file} loaded, but contains no 'images'. Evaluation cannot proceed.")
            else:
                if not detection_results: # If detection_results is empty, loadRes might fail or behave unexpectedly
                    logger.warning(f"No detections to load into coco_dt from {output_json_path}. mAP will be 0.")
                    coco_dt = coco_gt.loadRes([]) # Load empty results
                else:
                    coco_dt = coco_gt.loadRes(output_json_path)

                if not coco_dt.dataset.get('annotations') and detection_results: # If predictions were non-empty but loadRes resulted in empty
                    logger.warning(f"Loaded detection results from {output_json_path}, but it resulted in zero annotations for COCOeval. mAP will be 0.")
                
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize() 
                map_stats = coco_eval.stats
                if map_stats is not None and len(map_stats) >= 2:
                    map_results = {"map": float(map_stats[0]), "map_50": float(map_stats[1])}
                else:
                    logger.error("COCOeval stats were None or too short. mAP results will be 0.")
        except FileNotFoundError: 
            logger.error(f"Annotation file not found at {annotation_file} during COCOeval setup.")
        except Exception as e:
            logger.error(f"Error during COCO evaluation process: {e}", exc_info=True)

    total_images_in_dataset = 0
    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
        total_images_in_dataset = len(dataloader.dataset)
    elif num_batches_processed > 0 and hasattr(dataloader, 'batch_size') and dataloader.batch_size is not None : 
        total_images_in_dataset = num_batches_processed * dataloader.batch_size 
        logger.warning(f"Dataset has no __len__ or dataloader issues. Approximating total images as {total_images_in_dataset}.")
    else:
        logger.warning("Could not determine total number of images for sparsity calculation if num_batches_processed is 0.")


    average_sparsity = total_sparsity / total_images_in_dataset if total_images_in_dataset > 0 else 0.0
    logger.info(f"Evaluation Step {step_or_epoch}: mAP50={map_results.get('map_50', 0.0):.4f}, mAP={map_results.get('map', 0.0):.4f}, Sparsity={average_sparsity:.4f}")

    model.train() 
    return map_results, average_sparsity