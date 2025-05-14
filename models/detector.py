import torch
from ultralytics import YOLO
from typing import List, Dict, Union, Tuple, Optional, Any

class DetectorWrapper(torch.nn.Module):
    """
    YOLO 检测器包装类，用于封装 YOLO 模型并提供标准化接口。
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化 YOLO 检测器包装类。

        参数:
            model_path (str): 预训练 YOLO 模型的路径 (.pt file)。
            device (str): 运行设备 ('cuda' 或 'cpu')。
        """
        super(DetectorWrapper, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[YOLO] = None # Initialize as None
        try:
            # Check if model_path is valid before loading
            import os
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"YOLO model file not found at {model_path}")
            self.model = YOLO(model_path) # Load model first
            self.model.to(self.device) # Then move to device
            print(f"YOLO model loaded successfully from {model_path} to {self.device}")
            # Set to training mode by default, assuming it might be fine-tuned
            # or used for loss calculation. Can be changed via .eval() externally.
            self.model.train()
        except Exception as e:
            print(f"Error loading YOLO model from {model_path}: {e}")
            self.model = None # Ensure model is None if loading failed

    def forward(self,
                images: torch.Tensor,
                targets: Optional[List[Dict]] = None
                ) -> Union[Tuple[Optional[Any], Optional[Union[torch.Tensor, Dict]]], List[Dict]]:
        """
        前向传播，执行目标检测或计算损失。

        参数:
            images (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
                                   Ultralytics YOLO 通常期望 RGB 格式。
            targets (Optional[List[Dict]]): 目标检测标注 (仅在训练时用于计算损失)。
                                            格式应与 YOLO 模型兼容。

        返回:
            如果在训练模式 (self.training is True) 且提供了 targets:
                返回 (preds, loss)，其中 preds 是模型的原始输出，loss 是计算得到的损失 (tensor 或 dict)。
            如果在评估模式 (self.training is False) 或未提供 targets:
                返回检测结果列表 (List[Dict])，每个字典包含 'boxes', 'scores', 'labels'。
            如果模型加载失败，会引发 RuntimeError。
        """
        if self.model is None:
            raise RuntimeError("YOLO model was not loaded correctly or is unavailable.")

        # Ensure images are on the correct device
        images = images.to(self.device)

        if self.training and targets is not None:
            # --- 训练模式 ---
            # 假设 ultralytics YOLO 模型在训练时，调用 forward (即 model(...))
            # 传入 images 和 targets 会直接返回包含损失的结果。
            # targets 需要是 YOLO 模型期望的格式。
            # 通常，YOLO 的 .train() 方法会处理数据加载和 targets 格式化，
            # 但在这里我们直接调用 forward，需要确保 targets 格式正确。
            try:
                # --- 修改点：直接将 targets 传递给 model ---
                # Ultralytics YOLO v8 在训练模式下调用 __call__ 时，
                # 如果提供了 'targets' 参数，通常会返回一个包含损失的对象或元组。
                # 我们需要从返回结果中提取损失。
                # 注意：YOLO 可能需要 targets 是特定格式的 Tensor，而不是 list of dicts。
                # 可能需要在此处添加 targets 格式转换的代码。
                # 假设 targets 已经是模型所需的格式。
                results = self.model(images, targets=targets, verbose=False)

                # --- 提取损失 ---
                # Ultralytics YOLOv8 的 train() 返回值通常不直接是 loss，
                # loss 通常存储在 trainer 的属性中或通过回调访问。
                # 当直接调用 model(images, targets=targets) 时，其行为可能不同，
                # 可能不直接返回 loss，或者返回包含 loss 的复杂对象。
                #
                # **更稳妥的方式（推荐）：**
                # 在 ConditionalSR 中，不直接计算 detection_loss，而是让
                # ConditionalSR 返回 YOLO 的原始预测 (bounding boxes)。
                # 然后在 calculate_joint_loss 函数中，使用这些预测和 targets，
                # 调用 YOLO 模型提供的独立损失计算函数（如果存在）或
                # 使用标准的检测损失（如 IoU loss, classification loss）来计算。
                #
                # **如果坚持在此处获取损失（需要验证）：**
                # 检查 `results` 的类型和内容，看是否包含损失信息。
                # 这取决于具体的 ultralytics 版本。
                # 示例：假设 loss 存储在 results 的某个属性中 (需要确认!)
                loss = None
                if hasattr(results, 'loss'): # 示例属性名，需要确认
                    loss = results.loss
                elif isinstance(results, (tuple, list)) and len(results) > 1 and torch.is_tensor(results[1]): # 假设返回 (preds, loss_tensor)
                    loss = results[1]
                elif isinstance(results, dict) and 'loss' in results: # 假设返回 dict 包含 'loss'
                    loss = results['loss']
                else:
                    # 如果无法直接从 model() 调用中获取损失，
                    # 可能需要调用模型的特定损失函数（如果暴露出来）。
                    # 或者如上所述，在外部计算损失。
                    print("Warning: Could not directly extract loss from YOLO model output in training mode.")
                    # 返回模型的原始预测结果，损失设为 None
                    # preds = results # 假设 results 是预测结果
                    # return preds, None
                    # 暂时返回 None, None 表示无法计算
                    return None, None


                # 假设 preds 是模型的主要预测输出 (可能就是 results 本身或其一部分)
                preds = results # 或者 results[0] 等，取决于模型返回格式

                return preds, loss

            except Exception as e:
                print(f"Error during YOLO forward/loss calculation in training: {e}")
                import traceback
                print(traceback.format_exc())
                return None, None # Indicate error

        else:
            # --- 推理模式 ---
            self.model.eval() # 确保模型处于评估模式
            with torch.no_grad(): # 禁用梯度计算
                 results = self.model(images, verbose=False)
            # 如果之前是训练模式，调用者应负责将其设置回去
            # self.model.train() # 通常由外部管理

            # 提取检测结果
            detections = []
            # results 通常是 ultralytics Results 对象列表
            for result in results:
                detections.append({
                    "boxes": result.boxes.xyxy.cpu(), # [N, 4]
                    "scores": result.boxes.conf.cpu(), # [N]
                    "labels": result.boxes.cls.cpu()   # [N]
                })
            return detections # 返回检测结果列表
