# models/detector.py
import torch
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple, Optional, Any
import os

class DetectorWrapper(torch.nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(DetectorWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[YOLO] = None
        self.yolo_model_module: Optional[torch.nn.Module] = None

        try:
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"YOLO model file not found at {model_path}")
            
            self.model = YOLO(model_path) 

            if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
                self.yolo_model_module = self.model.model
                self.yolo_model_module.to(self.device) 
            else:
                print(f"Warning: Could not directly access underlying nn.Module of YOLO model. Train/eval state might not be managed correctly by DetectorWrapper.")
                if self.model is not None:
                    try:
                        # This might not be sufficient if self.model itself is not the nn.Module part
                        self.model.to(self.device)
                    except Exception as e_to:
                        print(f"Warning: Failed to move self.model to device {self.device}: {e_to}")
            
            # 将YOLO对象本身也移到指定设备，以防其内部有其他需要设备同步的操作
            if self.model is not None:
                self.model.to(self.device)


            print(f"YOLO model loaded successfully from {model_path}. Task: {self.model.task if self.model else 'unknown'}")

        except Exception as e:
            print(f"Error loading YOLO model from {model_path}: {e}")
            self.model = None
            self.yolo_model_module = None

    def train(self, mode: bool = True):
        # 1. 手动设置 DetectorWrapper 自身的 training 属性
        self.training = mode
        
        # 2. 只对底层的 yolo_model_module 设置模式
        if self.yolo_model_module:
            self.yolo_model_module.train(mode)
        # 不要调用 super().train(mode)，以避免它递归调用 self.model.train(mode)
        return self

    def eval(self):
        # 1. 手动设置 DetectorWrapper 自身的 training 属性为 False
        self.training = False
        
        # 2. 只对底层的 yolo_model_module 设置模式
        if self.yolo_model_module:
            self.yolo_model_module.eval()
        # 不要调用 super().eval()
        return self

    def _format_targets_for_yolo(self, images_tensor: torch.Tensor, coco_targets: List[Dict]) -> Optional[List[torch.Tensor]]:
        # ... (此方法内容与上一版本相同，确保设备正确性) ...
        if not coco_targets:
            return None

        yolo_formatted_targets = []
        for i, target_dict in enumerate(coco_targets):
            current_image_tensor = images_tensor[i].to(self.device) if images_tensor.device != self.device else images_tensor[i]
            img_h, img_w = current_image_tensor.shape[-2:]
            
            boxes_abs_coco = target_dict.get('boxes')
            labels = target_dict.get('labels')

            if boxes_abs_coco is None or labels is None or boxes_abs_coco.numel() == 0:
                yolo_formatted_targets.append(torch.empty((0, 5), device=self.device, dtype=torch.float32))
                continue

            boxes_abs_coco = torch.as_tensor(boxes_abs_coco, dtype=torch.float32, device=self.device)
            labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device) # 确保label也是tensor且在device上

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
                yolo_formatted_targets.append(torch.empty((0, 5), device=self.device, dtype=torch.float32))
                continue
            
            yolo_target_for_image = torch.cat((labels_filtered.float().unsqueeze(1), boxes_xywh_norm), dim=1)
            yolo_formatted_targets.append(yolo_target_for_image)
        
        return yolo_formatted_targets

    def forward(self,
                images: torch.Tensor,
                targets: Optional[List[Dict]] = None
                ) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        if self.yolo_model_module is None:
            raise RuntimeError("YOLO model's underlying nn.Module (self.yolo_model_module) was not loaded correctly or is unavailable.")

        images = images.to(self.device)

        # forward 方法内不需要再调用 self.yolo_model_module.train() / .eval()
        # 因为外部的 conditional_sr.train() / .eval() 已经通过 DetectorWrapper 的 train/eval 方法设置了模式
        if self.training:
            predictions = self.yolo_model_module(images) 
            loss_value = None 
            return predictions, loss_value
        else: 
            with torch.no_grad():
                 raw_results_from_yolo_object = self.model(images, verbose=False)

            detections = []
            if isinstance(raw_results_from_yolo_object, list): 
                for res_obj in raw_results_from_yolo_object:
                    detections.append({
                        "boxes": res_obj.boxes.xyxy.cpu(),
                        "scores": res_obj.boxes.conf.cpu(),
                        "labels": res_obj.boxes.cls.cpu()
                    })
            elif hasattr(raw_results_from_yolo_object, 'pred'):
                 for pred_tensor in raw_results_from_yolo_object.pred:
                     detections.append({
                         "boxes": pred_tensor[:, :4].cpu(),
                         "scores": pred_tensor[:, 4].cpu(),
                         "labels": pred_tensor[:, 5].cpu()
                     })
            else:
                print(f"Warning: Unexpected format for YOLO inference results: {type(raw_results_from_yolo_object)}. Returning as is.")
                return raw_results_from_yolo_object, None
            
            return detections, None