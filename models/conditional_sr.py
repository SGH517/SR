# models/conditional_sr.py
import torch
import torch.nn as nn
import torch.nn.functional as F # 确保 F 被导入
import os
from typing import Dict, Optional, Any, Tuple, List, Union
from collections import OrderedDict
import logging

from models.detector import DetectorWrapper
from models.masker import Masker # 确保 Masker 被导入
from models.sr_fast import SRFast # 确保 SRFast 被导入
from models.sr_quality import SRQuality # 确保 SRQuality 被导入
from utils.gumbel import gumbel_softmax
# 从新的工具模块导入权重加载函数
from utils.model_utils import load_model_weights

# 设置一个 logger 实例
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class ConditionalSR(nn.Module):
    def __init__(self,
                 sr_fast: SRFast, # 类型提示
                 sr_quality: SRQuality, # 类型提示
                 masker: Masker, # 类型提示
                 detector_weights: Optional[str], # detector_weights 可以是 None
                 sr_fast_weights: Optional[str],
                 sr_quality_weights: Optional[str],
                 masker_weights: Optional[str] = None,
                 device: str = 'cuda',
                 config: Optional[Dict] = None):
        """
        初始化 ConditionalSR 模块。

        参数:
            sr_fast: SRFast 网络实例。
            sr_quality: SRQuality 网络实例。
            masker: Masker 网络实例。
            detector_weights: YOLO 检测器的预训练权重路径 (如果为 None 或空字符串，则不加载)。
            sr_fast_weights: SRFast 网络的预训练权重路径 (如果为 None，则不加载)。
            sr_quality_weights: SRQuality 网络的预训练权重路径 (如果为 None，则不加载)。
            masker_weights: Masker 网络的预训练权重路径 (可选, 如果为 None，则不加载)。
            device: 运行设备 ('cuda' 或 'cpu')。
            config: 配置字典 (可选, 但推荐传入以获取参数如 threshold)。
        """
        super(ConditionalSR, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"ConditionalSR 将在设备: {self.device} 上运行。")

        self.sr_fast = sr_fast.to(self.device)
        self.sr_quality = sr_quality.to(self.device)
        self.masker = masker.to(self.device)
        self.config = config if config is not None else {} # 确保 config 是一个字典

        # 初始化检测器
        self.detector: Optional[DetectorWrapper] = None
        if detector_weights and isinstance(detector_weights, str) and os.path.exists(detector_weights):
            logger.info(f"正在从 {detector_weights} 为 ConditionalSR 初始化 DetectorWrapper...")
            self.detector = DetectorWrapper(model_path=detector_weights, device=str(self.device))
            if self.detector.model is None and self.detector.yolo_model_module is None:
                logger.warning(f"ConditionalSR 中的 DetectorWrapper 未能成功加载 YOLO 模型。检测功能将不可用。")
                self.detector = None # 如果加载失败，则将其设为 None
            else:
                logger.info(f"ConditionalSR 中的 DetectorWrapper 已成功初始化。")
        elif detector_weights: # 如果提供了路径但不存在
            logger.warning(f"DetectorWrapper 的权重路径 {detector_weights} 未找到。检测器将不可用。")
        else:
            logger.info("未提供检测器权重路径。ConditionalSR 中的检测器将不可用。")


        # 使用新的工具函数加载预训练权重
        load_model_weights(self.sr_fast, sr_fast_weights, self.device, "SR_Fast", logger, strict=False)
        load_model_weights(self.sr_quality, sr_quality_weights, self.device, "SR_Quality", logger, strict=False)
        load_model_weights(self.masker, masker_weights, self.device, "Masker", logger, strict=False)

        # 在初始化结束时校验配置
        try:
            self._validate_config()
        except ValueError as e:
            logger.error(f"ConditionalSR 配置校验失败: {e}", exc_info=True)
            # 根据需要决定如何处理无效配置：引发错误，使用默认值等。
            # 当前只是记录错误。

    def _validate_config(self):
        """
        验证配置字典的完整性。
        """
        if not self.config:
            # 允许 config 为空，但记录警告，因为某些操作可能依赖它
            logger.warning("ConditionalSR 的配置字典 (self.config) 为空。"
                           " 某些功能 (如掩码阈值) 可能使用默认值或无法工作。")
            self.config = {} # 确保它是一个字典

        model_cfg = self.config.get('model', {})
        masker_cfg = model_cfg.get('masker', {})

        if 'threshold' not in masker_cfg:
            logger.warning("在 config['model']['masker'] 中未找到 'threshold'。"
                           " 推理时硬掩码的默认阈值可能为 0.5。")
            # 可以选择在这里设置一个默认值到 self.config 中
            if 'model' not in self.config: self.config['model'] = {}
            if 'masker' not in self.config['model']: self.config['model']['masker'] = {}
            self.config['model']['masker']['threshold'] = 0.5 # 设置默认值

        # 训练相关的配置校验可以放在训练脚本中，这里主要校验 ConditionalSR 自身运行所需的参数
        train_cfg = self.config.get('train', {})
        # if 'target_sparsity_ratio' not in train_cfg:
        #     logger.warning("在 config['train'] 中未找到 'target_sparsity_ratio'。")

        # 可以添加对 SR_Fast, SR_Quality 配置参数的校验，确保与传入的模型实例匹配
        # 例如，检查 scale_factor 是否一致等，但这通常在更高层级的配置校验中完成。

    def forward(self,
                lr_image: torch.Tensor,
                targets: Optional[List[Dict]] = None, # COCO 格式的标注列表
                temperature: float = 1.0,
                hard_mask_inference: bool = False
                ) -> Dict[str, Optional[Any]]:
        """
        ConditionalSR 的前向传播。

        参数:
            lr_image (torch.Tensor): 低分辨率输入图像张量 (B, C, H_lr, W_lr)。
            targets (Optional[List[Dict]]): 真实标注列表 (COCO格式)，
                                             主要在训练时传递给 DetectorWrapper，然后进一步传递给损失函数。
            temperature (float): Gumbel-Softmax 的温度参数，仅在训练时使用。
            hard_mask_inference (bool): 是否在推理时使用硬掩码 (基于阈值)。

        返回:
            Dict[str, Optional[Any]]: 包含以下键的字典：
                - "sr_image": 超分辨率图像 (B, C, H_sr, W_sr)。
                - "mask_coarse": Masker 输出的粗粒度掩码 (B, 1, H_mask, W_mask)。
                - "mask_fused": 上采样后用于融合 SR 结果的掩码 (B, 1, H_sr, W_sr)。
                - "yolo_raw_predictions":
                    - 训练模式: YOLO 检测头部的原始输出 (通常是列表/元组的特征图)。
                    - 推理模式: 格式化后的检测结果列表 (每个元素是一个包含 'boxes', 'scores', 'labels' 的字典)。
                - "detection_loss_from_wrapper": 始终为 None (损失计算已移至外部)。
        """
        lr_image = lr_image.to(self.device)

        # --- 1. Mask Generation ---
        mask_logits_coarse = self.masker(lr_image) # (B, 1, H_mask, W_mask)
        mask_coarse_output: Optional[torch.Tensor] = None # 用于返回的粗掩码
        mask_to_upsample: Optional[torch.Tensor] = None   # 用于上采样和融合的掩码

        # 从配置中获取阈值，如果配置不存在或键不存在，则使用默认值
        mask_threshold_cfg = self.config.get('model', {}).get('masker', {}).get('threshold', 0.5)

        if self.training:
            # Gumbel-Softmax for differentiable sampling of the mask
            # 输入 logits 应该是 [B, num_choices * C_out_masker, H_coarse, W_coarse]
            # 这里我们有2个选择 (SR_Quality vs SR_Fast)，C_out_masker 通常是 1
            # 所以我们构造一个 [B, 2, H_mask, W_mask] 的输入
            # mask_logits_coarse 是 "选择 SR_Quality" 的 logits
            # 0 是 "选择 SR_Fast" 的 logits (可以认为是 -mask_logits_coarse，或简单地是0)
            gumbel_input_logits = torch.cat(
                [mask_logits_coarse, torch.zeros_like(mask_logits_coarse)],
                dim=1
            ) # Shape: [B, 2, H_mask, W_mask]

            mask_gumbel_output = gumbel_softmax(
                gumbel_input_logits,
                tau=temperature,
                hard=False, # 在训练时通常使用软掩码以保证梯度流
                dim=1       # 在 "选择" 维度上应用 softmax
            )
            # mask_gumbel_output[:, 0, :, :] 是选择 SR_Quality 的概率
            # mask_gumbel_output[:, 1, :, :] 是选择 SR_Fast 的概率
            mask_soft_coarse = mask_gumbel_output[:, 0:1, :, :] # 取选择 SR_Quality 的概率作为掩码
            mask_coarse_output = mask_soft_coarse
            mask_to_upsample = mask_soft_coarse
        else: # 推理模式
            mask_prob_coarse = torch.sigmoid(mask_logits_coarse) # 将 logits 转换为概率
            if hard_mask_inference:
                mask_hard_coarse = (mask_prob_coarse > mask_threshold_cfg).float()
                mask_coarse_output = mask_hard_coarse
                mask_to_upsample = mask_hard_coarse
            else:
                # 如果不是硬掩码，可以使用概率图或经过阈值处理的软掩码
                # 为了与训练时的软掩码行为（某种程度上）保持一致，这里可以使用概率值
                mask_soft_coarse = mask_prob_coarse
                mask_coarse_output = mask_soft_coarse
                mask_to_upsample = mask_soft_coarse

        # --- 2. Super-Resolution ---
        # 即使相应的路径可能不被完全使用，也先计算两个SR结果
        # 优化：如果掩码完全是0或1，可以考虑只计算一个路径，但这会使计算图动态化，增加复杂性
        sr_fast_output = self.sr_fast(lr_image)     # (B, C, H_sr, W_sr)
        sr_quality_output = self.sr_quality(lr_image) # (B, C, H_sr, W_sr)

        # --- 3. Fusion based on Mask ---
        mask_for_fusion_resized: Optional[torch.Tensor] = None
        if mask_to_upsample is not None:
            target_sr_size = sr_fast_output.shape[-2:] # H_sr, W_sr
            mask_for_fusion_resized = F.interpolate( # 使用 torch.nn.functional.interpolate
                mask_to_upsample.float(),       # 确保是 float 类型
                size=target_sr_size,
                mode='bilinear',                # 双线性插值通常效果较好
                align_corners=False             # 通常设为 False
            )
            # 确保插值后的掩码值在 [0, 1] 范围内 (特别是当 hard=False 时)
            mask_for_fusion_resized = torch.clamp(mask_for_fusion_resized, 0.0, 1.0)

            # 融合：M * HQ + (1-M) * Fast
            sr_image = mask_for_fusion_resized * sr_quality_output + \
                       (1.0 - mask_for_fusion_resized) * sr_fast_output
        else:
            # 理论上 mask_to_upsample 不应该为 None，除非 Masker 完全失效或未初始化
            logger.warning("mask_to_upsample 为 None。SR 图像将默认使用 sr_fast_output。")
            sr_image = sr_fast_output # Fallback

        # 确保SR图像的像素值在有效范围内 (例如 [0,1] 如果输入是归一化的)
        sr_image = torch.clamp(sr_image, 0.0, 1.0) # 假设图像数据范围是 [0,1]

        # --- 4. Detection ---
        yolo_raw_predictions: Optional[Any] = None
        detection_loss_value: Optional[torch.Tensor] = None # 来自 DetectorWrapper 的损失（现在应为 None）

        if self.detector is not None:
            # 设置 DetectorWrapper 的模式 (训练或评估)
            # nn.Module 的 train() 方法会递归设置子模块的模式
            # DetectorWrapper 的 train/eval 方法会正确处理其内部的 yolo_model_module
            self.detector.train(self.training)

            if self.training:
                if targets is None:
                     logger.warning("ConditionalSR 处于训练模式，但未提供目标标注给检测器。")
                # DetectorWrapper.forward 现在返回 (raw_predictions, None)
                yolo_raw_predictions, detection_loss_value = self.detector(sr_image, targets=targets)
                # detection_loss_value 应该为 None, 损失计算已移至外部
            else: # 推理模式
                # DetectorWrapper.forward 返回 (detections_list, None)
                yolo_raw_predictions, _ = self.detector(sr_image, targets=None)
        else:
            # logger.info("ConditionalSR 中的检测器不可用。跳过检测步骤。") # 可能过于频繁
            pass


        return {
            "sr_image": sr_image,
            "mask_coarse": mask_coarse_output,       # Masker 的直接输出 (B, 1, H_mask, W_mask)
            "mask_fused": mask_for_fusion_resized,   # 上采样后用于融合的掩码 (B, 1, H_sr, W_sr)
            "yolo_raw_predictions": yolo_raw_predictions, # 训练: 原始输出; 推理: 格式化结果
            "detection_loss_from_wrapper": detection_loss_value, # 应为 None
        }