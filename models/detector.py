# models/detector.py
import torch
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple, Optional, Any
import os # 确保 os 被导入

class DetectorWrapper(torch.nn.Module):
    def __init__(self, model_path: str, device: str = 'cuda'):
        super(DetectorWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[YOLO] = None
        self.yolo_model_module: Optional[torch.nn.Module] = None # 用于存储实际的 nn.Module

        try:
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"YOLO model file not found at {model_path}")
            
            self.model = YOLO(model_path) # 加载 YOLO 模型对象
            self.model.to(self.device)
            
            # 获取底层的 torch.nn.Module
            if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
                self.yolo_model_module = self.model.model
            else:
                # 如果直接的 .model 属性不是 nn.Module, 可能需要其他方式获取
                # 或者接受 self.model 本身可能是一个 nn.Module 的子类 (需要查阅 Ultralytics 文档)
                # 为了安全，如果找不到，我们先假设它可能无法直接设置 train/eval
                print(f"Warning: Could not directly access underlying nn.Module of YOLO model. Train/eval state might not be managed correctly by DetectorWrapper.")

            print(f"YOLO model loaded successfully from {model_path} to {self.device}")
            # 不在这里显式调用 self.model.train() 或 self.yolo_model_module.train()
            # 将由外部的 ConditionalSR 控制

        except Exception as e:
            print(f"Error loading YOLO model from {model_path}: {e}")
            self.model = None
            self.yolo_model_module = None

    def train(self, mode: bool = True):
        # 重写 train 方法以正确设置底层模型的模式
        super().train(mode) # 设置 DetectorWrapper 自身的 training 属性
        if self.yolo_model_module:
            self.yolo_model_module.train(mode)
        elif self.model: # 如果 yolo_model_module 未找到，尝试直接在 self.model 上设置
            self.model.train(mode) 
            # 注意: Ultralytics YOLO对象的 .train(mode) 可能不仅仅是 nn.Module.train(mode)
            # 它可能会尝试启动完整的训练流程，如果这样，这个方式仍有问题。
            # 最理想的是只控制 torch.nn.Module 部分的模式。
        return self

    def eval(self):
        # 重写 eval 方法
        super().eval()
        if self.yolo_model_module:
            self.yolo_model_module.eval()
        elif self.model:
            self.model.eval()
        return self

    def _format_targets_for_yolo(self, images_tensor: torch.Tensor, coco_targets: List[Dict]) -> Optional[List[torch.Tensor]]:
        """
        将 COCO 风格的 targets (List[Dict]) 转换为 YOLO 期望的格式。
        YOLO 通常期望一个列表的张量，每个张量对应批次中的一张图像，形状为 [N_objects, 5]，
        内容为 [class_idx, x_center_norm, y_center_norm, width_norm, height_norm]。
        """
        if not coco_targets:
            return None

        yolo_formatted_targets = []
        for i, target_dict in enumerate(coco_targets):
            img_h, img_w = images_tensor[i].shape[-2:] # 获取当前图像的 H, W (来自 SR 图像)
            
            boxes_abs_coco = target_dict.get('boxes') # COCO: [x_min, y_min, width, height]
            labels = target_dict.get('labels')

            if boxes_abs_coco is None or labels is None or boxes_abs_coco.numel() == 0:
                yolo_formatted_targets.append(torch.empty((0, 5), device=self.device, dtype=torch.float32))
                continue

            # 确保 boxes_abs_coco 是浮点数张量
            boxes_abs_coco = boxes_abs_coco.float()

            # 转换为中心点坐标和宽高 [x_center, y_center, width, height] (绝对像素值)
            boxes_xywh_abs = torch.zeros_like(boxes_abs_coco)
            boxes_xywh_abs[:, 0] = boxes_abs_coco[:, 0] + boxes_abs_coco[:, 2] / 2  # x_center
            boxes_xywh_abs[:, 1] = boxes_abs_coco[:, 1] + boxes_abs_coco[:, 3] / 2  # y_center
            boxes_xywh_abs[:, 2] = boxes_abs_coco[:, 2]                            # width
            boxes_xywh_abs[:, 3] = boxes_abs_coco[:, 3]                            # height

            # 归一化
            boxes_xywh_norm = boxes_xywh_abs.clone()
            boxes_xywh_norm[:, [0, 2]] /= img_w
            boxes_xywh_norm[:, [1, 3]] /= img_h
            
            # 确保坐标在 [0, 1] 范围内，并且宽高为正
            boxes_xywh_norm[:, 0:4] = torch.clamp(boxes_xywh_norm[:, 0:4], min=0.0, max=1.0)
            # 过滤掉宽高为0的框 (如果需要)
            valid_indices = (boxes_xywh_norm[:, 2] > 1e-4) & (boxes_xywh_norm[:, 3] > 1e-4)
            if not valid_indices.all():
                 boxes_xywh_norm = boxes_xywh_norm[valid_indices]
                 labels_filtered = labels[valid_indices]
            else:
                 labels_filtered = labels
            
            if boxes_xywh_norm.numel() == 0:
                yolo_formatted_targets.append(torch.empty((0, 5), device=self.device, dtype=torch.float32))
                continue

            # YOLO 格式: [class_idx, x_center_norm, y_center_norm, width_norm, height_norm]
            # 确保 labels 是浮点数以便拼接
            yolo_target_for_image = torch.cat((labels_filtered.float().unsqueeze(1), boxes_xywh_norm), dim=1)
            yolo_formatted_targets.append(yolo_target_for_image)
        
        return yolo_formatted_targets

    def forward(self,
                images: torch.Tensor, # 这些应该是 SR 处理后的图像
                targets: Optional[List[Dict]] = None # COCO 风格的 targets
                ) -> Union[Tuple[Optional[Any], Optional[Union[torch.Tensor, Dict]]], List[Dict]]:
        if self.model is None:
            # 之前这里会引发 RuntimeError，这应该保留
            raise RuntimeError("YOLO model was not loaded correctly or is unavailable.")

        images = images.to(self.device)

        if self.training and targets is not None:
            yolo_targets_for_loss = self._format_targets_for_yolo(images, targets)
            
            # 调用 YOLO 模型进行前向传播和损失计算
            # Ultralytics YOLO 在 __call__ 中如果提供了 targets，通常会计算损失
            # verbose=False 减少不必要的日志输出
            # 我们期望 results 是一个包含损失的对象/元组，或者直接就是损失
            try:
                # 重要: Ultralytics 的 model(images, targets=...) 的行为可能高度依赖版本
                # 和 targets 的确切格式。这里的 yolo_targets_for_loss 是尝试的格式。
                yolo_results = self.model(images, targets=yolo_targets_for_loss, verbose=False)
            except Exception as e:
                print(f"Error during YOLO model forward call with targets: {e}")
                import traceback
                traceback.print_exc()
                return None, None # 表示错误

            # 从 yolo_results 中提取损失
            # Ultralytics YOLOv8 的 train() 返回值通常不直接是 loss，
            # loss 通常存储在 trainer 的属性中或通过回调访问。
            # 当直接调用 model(images, targets=targets) 时，其行为可能不同。
            # 需要检查 yolo_results 的类型和内容来确定如何获取损失。
            # 这个提取逻辑需要基于 Ultralytics 的具体实现。

            loss_value = None
            predictions = None # 通常是检测头的原始输出

            if isinstance(yolo_results, tuple) and len(yolo_results) > 0:
                # 常见的一种情况是返回 (predictions, loss_components_dict_or_tensor)
                # 或者 YOLOv8 的 results 对象本身可能包含 loss (例如 results.loss)
                # 对于YOLOv8, 调用 model(batch) 返回一个列表的 Results 对象 (每个图像一个)
                # 如果提供了 targets, 这些 Results 对象可能不直接包含组合损失，
                # 或者 model() 的返回值结构会改变。
                # 如果 yolo_results 是一个包含损失的字典：
                if isinstance(yolo_results, dict) and 'loss' in yolo_results: # 假设1: 返回 dict {'loss': tensor, ...}
                    loss_value = yolo_results['loss']
                    predictions = yolo_results.get('predictions') # 假设的键
                # 如果 yolo_results 是 (preds, loss_tensor)
                elif isinstance(yolo_results, tuple) and len(yolo_results) == 2 and torch.is_tensor(yolo_results[1]):
                    predictions = yolo_results[0]
                    loss_value = yolo_results[1]
                # 如果 yolo_results 是 Ultralytics 的 Results 对象列表 (通常用于推理)
                # 在训练模式下直接调用 __call__ 可能不同。
                # Ultralytics YOLOv8 `trainer.criterion` 计算损失。
                # 直接调用 `model(images, targets=targets)` 可能不直接返回一个易于解析的单一损失张量。
                # 它可能返回原始的检测头输出，然后外部代码（如 trainer）用它和targets计算损失。

                # 这是一个棘手的部分，因为我们试图在外部使用YOLO的损失。
                # Ultralytics 的内部 `loss` 属性在 `Validator` 类中计算并返回。
                # 在 `trainer.py` 中，`model(batch)` 后，`self.loss, self.loss_items = self.criterion(preds, batch)`
                # 这意味着 `model(batch)` 返回的是 `preds`。
                
                # 重新思考: `self.model(images, targets=yolo_targets_for_loss)` 可能只返回 `preds`.
                # 然后我们需要使用这些 `preds` 和 `yolo_targets_for_loss` 来计算损失。
                # 这需要访问YOLO模型的损失函数 `self.model.criterion` (如果暴露) 或重新实现。

                # 简单起见，我们先假设 `yolo_results` 直接是 `preds` (常见情况)
                # 并且 `DetectorWrapper` 的目标是提供这些 `preds`，让 `calculate_joint_loss`
                # 中的 `precomputed_detection_loss` 参数接收一个 *可以计算成损失的结构*。
                # Ultralytics YOLO 的 `model(source, stream)` 通常返回 `Results` 对象或生成器。
                # `model.predict(source)` 也是如此。
                # `model.train(args)` 启动训练。
                # `model.val()` 启动验证。

                # 最可靠的做法可能是让 DetectorWrapper 返回 YOLO 的原始预测，
                # 并在 ConditionalSR 或 calculate_joint_loss 中，使用这些预测和目标，
                # 调用一个标准的检测损失函数 (如 Focal Loss, CIoU Loss 等)，
                # 而不是试图从 YOLO 内部提取一个黑盒的 "loss"。
                # 但目前的脚本结构是期望 DetectorWrapper 返回一个 "detection_loss"。

                # 鉴于日志 "engine\trainer: task=detect, mode=train, model=..., data=..."
                # 表明 `self.model(...)` 正在尝试运行其内部的训练器逻辑。
                # 这通常是因为 `targets` 参数的存在。
                # 如果我们只想用它进行推理式的前向传播，应该不传 `targets` 给 `self.model()`。
                # 但我们需要损失。

                # **一个关键的调整：**
                # `DetectorWrapper` 的 `forward` 在训练模式下，不应该直接返回从 `self.model()` 中提取的损失。
                # 而是应该返回 `self.model(images)` 的预测结果。
                # 然后 `ConditionalSR` 将这些预测结果和 `targets` 一起传递给 `calculate_joint_loss`。
                # `calculate_joint_loss` 则需要使用这些预测和 `targets` 来计算检测损失
                # (可能需要重新实现一个简单的检测损失，或者看看YOLO模型是否暴露了其criterion)。

                # 为了最小化改动并尝试解决当前问题：
                # 假设 `self.model(images, targets=yolo_targets_for_loss)` 返回的 `yolo_results`
                # 是一个特殊的对象，我们可以从中尝试提取损失。
                # 如果提取不到，`loss_value` 会是 `None`。
                # `predictions` 可以是 `yolo_results` 本身或其一部分。

                # 尝试从 Ultralytics YOLO 的输出中获取 loss 和 preds
                # 这部分非常依赖于YOLO库的具体版本和内部实现
                if hasattr(yolo_results, 'loss_items') and yolo_results.loss_items is not None: # YOLOv8 val()返回的Results对象有这个
                    loss_value = yolo_results.loss # 这是总损失
                    # preds可能需要从yolo_results的其他属性获取，或者yolo_results就是preds
                    predictions = yolo_results # 假设
                elif isinstance(yolo_results, list) and len(yolo_results) > 0 and hasattr(yolo_results[0], 'loss_items'): # 如果返回列表的Results
                    # 通常，训练时调用 model(batch) 返回的是原始预测张量，而不是Results列表
                    # 这里假设一个场景，具体需要看 Ultralytics 的文档或源码
                    # 对于批处理，损失通常是整个批次的平均损失
                    try:
                        # 这个逻辑可能不正确，因为model()在训练时的确切返回值需要确认
                        # 通常，model(batch)返回preds，然后 preds, batch -> criterion -> loss
                        # 我们这里没有直接调用criterion
                        # 我们假设如果 model(images, targets=...) 成功计算了损失，它会以某种方式返回
                        # 如果上面的扫描 data_stage1.yaml 的行为消失了，说明 targets 被接受了
                        # 这时 yolo_results 里应该有损失，或者 yolo_results 本身是 preds
                        pass # 需要进一步确认如何从 yolo_results 提取 loss 和 preds
                    except Exception as extraction_e:
                        print(f"Could not determine loss/preds structure from YOLO output: {extraction_e}")
                
                if loss_value is None:
                    print("Warning: Could not extract 'loss' from YOLO model output in training mode via known attributes. Detection loss will be None.")
                
                # 暂时将 yolo_results 作为 preds 返回
                return yolo_results, loss_value
            
            # except Exception as e:
            #     print(f"Error during YOLO forward/loss calculation in training: {e}")
            #     import traceback
            #     print(traceback.format_exc())
            #     return None, None # 表示错误

        else: # 推理模式 (self.training is False or targets is None)
            # 确保底层模型也处于评估模式
            if self.yolo_model_module: self.yolo_model_module.eval()
            elif self.model: self.model.eval()

            with torch.no_grad():
                 # 在推理模式下，不传递 targets
                 raw_results = self.model(images, verbose=False) # 这通常返回 Results 对象的列表

            # 提取检测结果 (通常是 List[Dict] 格式，每个字典包含 boxes, scores, labels)
            detections = []
            if isinstance(raw_results, list): # YOLOv8 通常返回列表的 Results 对象
                for res_obj in raw_results:
                    detections.append({
                        "boxes": res_obj.boxes.xyxy.cpu(),  # [N, 4]
                        "scores": res_obj.boxes.conf.cpu(), # [N]
                        "labels": res_obj.boxes.cls.cpu()   # [N]
                    })
            elif hasattr(raw_results, 'pred'): # 兼容旧版 YOLOv5 可能的返回
                 # raw_results.pred 是一个列表，每个元素是 [N, 6] (x1, y1, x2, y2, conf, cls)
                 for pred_tensor in raw_results.pred:
                     detections.append({
                         "boxes": pred_tensor[:, :4].cpu(),
                         "scores": pred_tensor[:, 4].cpu(),
                         "labels": pred_tensor[:, 5].cpu()
                     })
            else:
                print(f"Warning: Unexpected format for YOLO inference results: {type(raw_results)}")
                # 返回原始结果，让调用者处理
                return raw_results, None 

            return detections, None # 推理模式不返回损失