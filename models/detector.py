# models/detector.py
import torch
import torch.nn as nn # 确保 nn 被导入
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple, Optional, Any
import os
import logging # 添加 logging

# 从新的工具模块导入
from utils.yolo_target_utils import format_coco_targets_to_yolo

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# 该模块封装了YOLO检测器，用于目标检测任务。
class DetectorWrapper(nn.Module): # 继承自 nn.Module
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(DetectorWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[YOLO] = None
        self.yolo_model_module: Optional[nn.Module] = None # 底层的 nn.Module

        try:
            if not model_path: # 检查 model_path 是否为空
                logger.warning("DetectorWrapper 初始化：未提供 model_path，将不加载YOLO模型。")
                # 保持 self.model 和 self.yolo_model_module 为 None
                return # 提前返回，不尝试加载

            if not os.path.exists(model_path):
                 logger.error(f"YOLO 模型文件在路径 {model_path} 未找到。")
                 # 保持 self.model 和 self.yolo_model_module 为 None
                 return # 提前返回

            self.model = YOLO(model_path)

            # 尝试访问并存储底层的 torch.nn.Module
            # YOLOv8的 nn.Module 通常在 self.model.model
            if hasattr(self.model, 'model') and isinstance(self.model.model, nn.Module):
                self.yolo_model_module = self.model.model
                self.yolo_model_module.to(self.device)
                logger.info(f"已成功访问并移动 YOLO 模型的底层 nn.Module 到设备 {self.device}。")
            else:
                logger.warning(f"无法直接访问 YOLO 模型的底层 nn.Module。 "
                               f"DetectorWrapper 可能无法正确管理其训练/评估状态。")
                # 尝试将整个YOLO对象移动到设备，但这可能不足够
                if self.model is not None:
                    try:
                        self.model.to(self.device)
                        logger.info(f"已尝试将 YOLO 对象本身移动到设备 {self.device}。")
                    except Exception as e_to:
                        logger.warning(f"将 YOLO 对象移动到设备 {self.device} 失败: {e_to}")

            logger.info(f"YOLO 模型已成功从 {model_path} 加载。任务: {self.model.task if self.model else '未知'}")

        except FileNotFoundError: # 特别处理 FileNotFoundError
            logger.error(f"YOLO 模型文件在路径 {model_path} 未找到 (FileNotFoundError)。DetectorWrapper 将不包含有效模型。")
            self.model = None
            self.yolo_model_module = None
        except Exception as e:
            logger.error(f"从 {model_path} 加载 YOLO 模型时发生错误: {e}", exc_info=True)
            self.model = None
            self.yolo_model_module = None

    def train(self, mode: bool = True):
        # 1. 设置 DetectorWrapper 自身的 training 属性
        super().train(mode) # 调用 nn.Module 的 train 方法，它会设置 self.training

        # 2. 只对底层的 yolo_model_module 设置模式
        if self.yolo_model_module:
            self.yolo_model_module.train(mode)
        elif self.model: # 如果没有 yolo_model_module，但有 model 对象，尝试调用其 train
            try:
                self.model.train(mode) # YOLO 对象本身可能有 train 模式
            except AttributeError:
                logger.warning("YOLO 对象没有 train(mode) 方法，且底层模块不可用。")
            except Exception as e:
                logger.warning(f"调用 self.model.train({mode}) 时出错: {e}")
        return self

    def eval(self):
        # 1. 设置 DetectorWrapper 自身的 training 属性为 False
        super().eval() # 调用 nn.Module 的 eval 方法

        # 2. 只对底层的 yolo_model_module 设置模式
        if self.yolo_model_module:
            self.yolo_model_module.eval()
        elif self.model: # 如果没有 yolo_model_module，但有 model 对象，尝试调用其 eval
            try:
                self.model.eval()
            except AttributeError:
                logger.warning("YOLO 对象没有 eval() 方法，且底层模块不可用。")
            except Exception as e:
                logger.warning(f"调用 self.model.eval() 时出错: {e}")
        return self

    def _format_targets_for_yolo(self,
                                 images_tensor: torch.Tensor,
                                 coco_targets_batch: List[Dict]
                                 ) -> Optional[List[torch.Tensor]]:
        """
        使用共享的工具函数格式化COCO目标为YOLO格式。
        images_tensor 用于获取图像尺寸。
        coco_targets_batch 是一个批次的COCO风格标注列表。
        """
        if not coco_targets_batch:
            return None
        if not isinstance(images_tensor, torch.Tensor) or images_tensor.ndim != 4:
            logger.error(f"_format_targets_for_yolo 需要一个4D的图像张量 (B,C,H,W)，但得到: {type(images_tensor)} shape {images_tensor.shape if isinstance(images_tensor, torch.Tensor) else 'N/A'}")
            # 可能需要返回一个空列表或引发错误，取决于调用者的期望
            return [torch.empty((0,5), device=self.device, dtype=torch.float32)] * len(coco_targets_batch) if coco_targets_batch else []


        # 调用新的工具函数
        return format_coco_targets_to_yolo(coco_targets_batch, images_tensor.shape, self.device)


    def forward(self,
                images: torch.Tensor,
                targets: Optional[List[Dict]] = None
                ) -> Tuple[Optional[Any], Optional[torch.Tensor]]:
        """
        DetectorWrapper 的前向传播。

        在训练模式下:
            - 如果 self.yolo_model_module 可用，直接调用它进行预测。
            - 损失的计算被移到外部的 YOLO 损失函数中。此方法返回原始预测。
        在评估模式下:
            - 使用 self.model(images) 进行预测，并格式化输出。

        参数:
            images (torch.Tensor): 输入图像张量 (B, C, H, W)。
            targets (Optional[List[Dict]]): COCO 格式的真实标注列表，仅在训练时可能用到（如果模型内部需要）。
                                            但在此设计中，主要由外部损失函数使用。

        返回:
            Tuple[Optional[Any], Optional[torch.Tensor]]:
                - 第一个元素:
                    - 训练模式: YOLO模型头部的原始输出 (通常是一个包含多层特征图的列表/元组)。
                    - 评估模式: 一个检测结果列表，每个元素是一个字典，包含 'boxes', 'scores', 'labels'。
                - 第二个元素: 始终为 None (损失计算已移出)。
        """
        if not self.model and not self.yolo_model_module:
            logger.error("DetectorWrapper.forward: YOLO 模型未正确加载或不可用。")
            return None, None # 或者引发 RuntimeError

        images = images.to(self.device)

        # forward 方法内不需要再调用 self.yolo_model_module.train() / .eval()
        # 因为外部的 conditional_sr.train() / .eval() 已经通过 DetectorWrapper 的 train/eval 方法设置了模式

        if self.training:
            if self.yolo_model_module:
                # 在训练时，我们期望 self.yolo_model_module 返回的是进入损失函数前的原始预测
                # 例如，对于YOLOv8，这可能是 Detect()模块的输出，是一个包含3个级别特征图的列表
                raw_predictions = self.yolo_model_module(images)
                return raw_predictions, None # 损失将在外部计算
            else:
                logger.warning("DetectorWrapper 处于训练模式，但 yolo_model_module 不可用。无法获取原始预测。")
                # 尝试使用 self.model 进行预测，但这可能不是原始的头部输出
                try:
                    # YOLO 对象直接调用可能不等同于 model.model()
                    # 这里需要确认 ultralytics YOLO 对象在训练模式下如何返回原始预测
                    # 通常，ultralytics 的 train() 方法会处理整个训练循环
                    # 如果我们只是想获取预测，可能需要一个不同的调用方式或确保 yolo_model_module 可用
                    # 为了简单起见，如果 yolo_model_module 不可用，我们返回 None
                    logger.warning("尝试调用 self.model(images) 进行训练模式预测，结果可能非预期。")
                    # results = self.model(images) # 这在训练时可能行为不确定或不返回原始 logits
                    return None, None # 最好是确保 yolo_model_module 可用
                except Exception as e:
                    logger.error(f"尝试在训练模式下使用 self.model(images) 获取预测时出错: {e}", exc_info=True)
                    return None, None
        else: # 推理 (self.training is False)
            if not self.model: # 确保 self.model 在推理时可用
                logger.error("DetectorWrapper 处于评估模式，但 self.model 不可用。")
                return None, None

            with torch.no_grad():
                 # self.model() 在推理时返回 Results 对象列表
                 results_from_yolo_object = self.model(images, verbose=False) # verbose=False 避免打印

            detections = []
            if isinstance(results_from_yolo_object, list): # YOLOv8 返回 Results 对象列表
                for res_obj in results_from_yolo_object:
                    # 确保 res_obj.boxes 是存在的
                    if hasattr(res_obj, 'boxes') and res_obj.boxes is not None:
                        detections.append({
                            "boxes": res_obj.boxes.xyxy.cpu(),  # [N, 4]
                            "scores": res_obj.boxes.conf.cpu(), # [N]
                            "labels": res_obj.boxes.cls.cpu()   # [N]
                        })
                    else:
                        # 处理没有检测结果的图像
                        detections.append({
                            "boxes": torch.empty((0, 4), dtype=torch.float32),
                            "scores": torch.empty((0), dtype=torch.float32),
                            "labels": torch.empty((0), dtype=torch.int64)
                        })
            # 有些旧版本或不同配置的YOLO可能返回不同结构
            elif hasattr(results_from_yolo_object, 'pred') and isinstance(results_from_yolo_object.pred, list):
                 # 这是YOLOv5风格的输出 (List of [N, 6] tensors: x1, y1, x2, y2, conf, class)
                 for pred_tensor in results_from_yolo_object.pred: # pred_tensor for each image in batch
                     if pred_tensor is not None and pred_tensor.numel() > 0:
                         detections.append({
                             "boxes": pred_tensor[:, :4].cpu(),
                             "scores": pred_tensor[:, 4].cpu(),
                             "labels": pred_tensor[:, 5].cpu()
                         })
                     else:
                        detections.append({
                            "boxes": torch.empty((0, 4), dtype=torch.float32),
                            "scores": torch.empty((0), dtype=torch.float32),
                            "labels": torch.empty((0), dtype=torch.int64)
                        })
            else:
                logger.warning(f"YOLO 推理结果的格式未知或意外: {type(results_from_yolo_object)}。将原样返回。")
                return results_from_yolo_object, None # 或者返回空列表

            return detections, None # 推理模式下不计算损失