import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.detector import DetectorWrapper
from utils.gumbel import gumbel_softmax
from typing import Dict, Optional, Any, Tuple, List, Union
from collections import OrderedDict
from models.masker import Masker # Import Masker to access its parameters
import logging # Import logging

# Setup a logger for this module
logger = logging.getLogger(__name__)
# Ensure logger has handlers if not configured globally
if not logger.handlers:
    logging.basicConfig(level=logging.INFO) # Basic config if no handlers exist

class ConditionalSR(nn.Module):
    def __init__(self,
                 sr_fast: nn.Module,
                 sr_quality: nn.Module,
                 masker: nn.Module, # Masker instance is passed
                 detector_weights: str,
                 sr_fast_weights: str,
                 sr_quality_weights: str,
                 masker_weights: Optional[str] = None, # Masker weights can be loaded here
                 device: str = 'cuda',
                 config: Optional[Dict] = None):
        """
        初始化 ConditionalSR 模块。

        参数:
            sr_fast: SRFast 网络实例。
            sr_quality: SRQuality 网络实例。
            masker: Masker 网络实例。
            detector_weights: YOLO 检测器的预训练权重路径。
            sr_fast_weights: SRFast 网络的预训练权重路径。
            sr_quality_weights: SRQuality 网络的预训练权重路径。
            masker_weights: Masker 网络的预训练权重路径（可选）。
            device: 运行设备 ('cuda' 或 'cpu')。
            config: 配置字典（可选）。
        """
        super(ConditionalSR, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sr_fast = sr_fast.to(self.device)
        self.sr_quality = sr_quality.to(self.device)
        self.masker = masker.to(self.device) # Use the passed instance
        self.config = config if config is not None else {} # Ensure config is a dict

        # Initialize detector as None initially
        self.detector: Optional[DetectorWrapper] = None
        if detector_weights and isinstance(detector_weights, str):
            # Check path existence before initializing DetectorWrapper
            if os.path.exists(detector_weights):
                # Pass config to DetectorWrapper if it needs it (e.g., for class names)
                self.detector = DetectorWrapper(detector_weights, device=device)
                # Move detector's internal model to the correct device (handled in DetectorWrapper init)
            else:
                logger.warning(f"Detector weights path not found: {detector_weights}. Detector will be unavailable.")
        else:
            logger.info("No valid detector weights path provided. Detector will be unavailable.")

        # Load pre-trained weights
        self._load_weights_from_checkpoint(self.sr_fast, sr_fast_weights, "SR_Fast")
        self._load_weights_from_checkpoint(self.sr_quality, sr_quality_weights, "SR_Quality")
        self._load_weights_from_checkpoint(self.masker, masker_weights, "Masker") # Load Masker weights if provided

        # Validate config after initialization
        try:
            self._validate_config()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            # Decide how to handle invalid config: raise error, use defaults, etc.
            # For now, just print the warning.

    def _load_weights_from_checkpoint(self, model: nn.Module, weights_path: Optional[str], model_name: str):
        """ Helper function to load weights from checkpoint files with logging for strict=False. """
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = None
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.info(f"Loading {model_name} weights from 'model_state_dict' key in {weights_path}")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Common alternative key
                    state_dict = checkpoint['state_dict']
                    logger.info(f"Loading {model_name} weights from 'state_dict' key in {weights_path}")
                elif isinstance(checkpoint, dict): # Check if the checkpoint itself is the state_dict
                    # Basic check: are keys typical layer names?
                    if all(isinstance(k, str) for k in checkpoint.keys()):
                         state_dict = checkpoint
                         logger.info(f"Loading {model_name} weights directly from checkpoint dictionary: {weights_path}")
                elif isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict): # Direct state_dict save
                     state_dict = checkpoint
                     logger.info(f"Loading {model_name} weights directly from state_dict file: {weights_path}")

                if state_dict:
                    # Remove 'module.' prefix if present (from DataParallel/DDP)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v

                    # Load with strict=False and log mismatches
                    model_state_dict = model.state_dict()
                    # Filter out keys that don't exist in the model
                    pretrained_keys = set(new_state_dict.keys())
                    model_keys = set(model_state_dict.keys())

                    missing_in_pretrained = list(model_keys - pretrained_keys)
                    unexpected_in_pretrained = list(pretrained_keys - model_keys)

                    # Attempt to load
                    load_info = model.load_state_dict(new_state_dict, strict=False)

                    if load_info.missing_keys or load_info.unexpected_keys:
                         logger.warning(f"Loaded {model_name} weights from {weights_path} with strict=False.")
                         if load_info.missing_keys:
                             logger.warning(f"  Missing keys in checkpoint: {load_info.missing_keys}")
                         if load_info.unexpected_keys:
                             logger.warning(f"  Unexpected keys in checkpoint: {load_info.unexpected_keys}")
                    else:
                         logger.info(f"Successfully loaded {model_name} weights.")

                else:
                    logger.warning(f"Could not find a valid state_dict in {weights_path} for {model_name}.")

            except Exception as e:
                logger.error(f"Error loading {model_name} weights from {weights_path}: {e}")
        elif weights_path:
             logger.warning(f"{model_name} weights path not found: {weights_path}")
        else:
             logger.info(f"No weights path provided for {model_name}. Using initialized weights.")

    def _validate_config(self):
        """
        验证配置字典的完整性。
        """
        if not self.config:
            raise ValueError("Configuration dictionary (self.config) is missing or empty.")
        # Example required keys - adjust as needed
        required_keys_model = ['masker']
        required_keys_train = ['loss_weights'] # Example

        if 'model' not in self.config:
             raise ValueError("Missing 'model' section in configuration.")
        for key in required_keys_model:
            if key not in self.config['model']:
                raise ValueError(f"Missing required configuration key in 'model': {key}")

        # Only validate train keys if in training mode potentially?
        # Or assume config should always be complete.
        if 'train' not in self.config:
             raise ValueError("Missing 'train' section in configuration.")
        for key in required_keys_train:
             if key not in self.config['train']:
                 raise ValueError(f"Missing required configuration key in 'train': {key}")
        # Check for specific sub-keys if necessary
        if 'threshold' not in self.config['model']['masker']:
             logger.warning("'threshold' not found in config['model']['masker']. Defaulting might occur.")
        if 'target_sparsity_ratio' not in self.config['train']:
             logger.warning("'target_sparsity_ratio' not found in config['train']. Defaulting might occur.")


    def forward(self,
                lr_image: torch.Tensor,
                targets: Optional[List[Dict]] = None,
                temperature: float = 1.0,
                hard_mask_inference: bool = False
                ) -> Dict[str, Optional[Any]]:
        lr_image = lr_image.to(self.device)

        # --- Mask Generation ---
        mask_logits_coarse = self.masker(lr_image)
        mask_threshold = self.config.get('model', {}).get('masker', {}).get('threshold', 0.5)
        mask_coarse_out: Optional[torch.Tensor] = None
        mask_for_fusion_resized: Optional[torch.Tensor] = None

        if self.training:
            # ... (Gumbel-Softmax 逻辑不变) ...
            gumbel_input_logits = torch.cat([mask_logits_coarse, torch.zeros_like(mask_logits_coarse)], dim=1) # [B, 2*C_out_masker, H_coarse, W_coarse]
            mask_gumbel_output = gumbel_softmax(gumbel_input_logits, tau=temperature, hard=False, dim=1) # Apply Gumbel to the "channel" dim
            mask_soft_coarse = mask_gumbel_output[:, 0:1, :, :] # Select the "chosen" part
            mask_coarse_out = mask_soft_coarse
            mask_to_upsample = mask_soft_coarse
        else:
            mask_prob_coarse = torch.sigmoid(mask_logits_coarse)
            if hard_mask_inference:
                mask_hard_coarse = (mask_prob_coarse > mask_threshold).float()
                mask_coarse_out = mask_hard_coarse
                mask_to_upsample = mask_hard_coarse
            else:
                # 在推理时，如果不是 hard_mask，通常也期望是概率图或者经过阈值处理的软掩码
                # 为了与训练统一，这里可以是 mask_prob_coarse，由调用者决定如何使用
                mask_soft_coarse = mask_prob_coarse # 使用概率值
                mask_coarse_out = mask_soft_coarse
                mask_to_upsample = mask_soft_coarse
        
        sr_fast_output = self.sr_fast(lr_image)
        sr_quality_output = self.sr_quality(lr_image)

        if mask_to_upsample is not None:
            target_size = sr_fast_output.shape[-2:]
            mask_for_fusion_resized = torch.nn.functional.interpolate( # 使用 F.interpolate
                mask_to_upsample.float(), #确保是 float
                size=target_size,
                mode='bilinear', # 'bilinear' 通常效果更好，如果掩码比较粗糙
                align_corners=False # 通常设为 False
            )
            # 确保掩码在 [0,1] 范围内，尤其是 hard_mask_inference=False 时 sigmoid 输出后直接用
            mask_for_fusion_resized = torch.clamp(mask_for_fusion_resized, 0.0, 1.0)

            sr_image = mask_for_fusion_resized * sr_quality_output + (1 - mask_for_fusion_resized) * sr_fast_output
        else:
            logger.warning("mask_to_upsample is None. Defaulting SR image to sr_fast_output.")
            sr_image = sr_fast_output # Fallback
        
        # --- Detection ---
        yolo_raw_predictions: Optional[Any] = None
        # detection_loss_from_wrapper 将会是 None (来自 DetectorWrapper 的新行为)
        detection_loss_from_wrapper: Optional[torch.Tensor] = None 

        if self.detector is not None:
            self.detector.train(self.training) # 设置 DetectorWrapper 的模式

            if self.training:
                if targets is None:
                     logger.warning("ConditionalSR is in training mode, but no targets were provided for detection.")
                
                # DetectorWrapper.forward 现在返回 (raw_predictions, None)
                yolo_raw_predictions, detection_loss_from_wrapper = self.detector(sr_image, targets=targets)
                # `detection_loss_from_wrapper` 应该是 None, 我们不再依赖它来传递损失

            else: # 推理模式
                # DetectorWrapper.forward 现在返回 (detections_list, None)
                # detections_list 是处理好的检测结果列表
                yolo_raw_predictions, _ = self.detector(sr_image, targets=None) 
        else:
            logger.info("Detector is not available. Skipping detection.")

        # 返回给 calculate_joint_loss 的将是 yolo_raw_predictions
        # 和原始的 targets (calculate_joint_loss 将需要自己格式化 targets)
        # detection_loss_from_wrapper 现在是 None，所以 calculate_joint_loss 中的 precomputed_detection_loss 会是 None
        return {
            "sr_image": sr_image,
            "mask_coarse": mask_coarse_out, # 这是 Masker 的直接输出 (B, 1, H_coarse, W_coarse)
            "mask_fused": mask_for_fusion_resized, # 这是上采样后用于融合的掩码 (B, 1, H_sr, W_sr)
            "yolo_raw_predictions": yolo_raw_predictions, # 在训练时是YOLO的原始预测，推理时是格式化的检测结果
            "detection_loss_from_wrapper": detection_loss_from_wrapper, # 在训练时这里应该是 None
        }