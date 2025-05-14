import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.detector import DetectorWrapper
from utils.gumbel import gumbel_softmax
from typing import Dict, Optional, Any, Tuple, List
from collections import OrderedDict

class ConditionalSR(nn.Module):
    def __init__(self,
                 sr_fast: nn.Module,
                 sr_quality: nn.Module,
                 masker: nn.Module,
                 detector_weights: str,
                 sr_fast_weights: str,
                 sr_quality_weights: str,
                 masker_weights: Optional[str] = None,
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
        self.masker = masker.to(self.device)
        self.config = config if config is not None else {} # Ensure config is a dict

        # Initialize detector as None initially
        self.detector: Optional[DetectorWrapper] = None
        if detector_weights and isinstance(detector_weights, str):
            # Check path existence before initializing DetectorWrapper
            if os.path.exists(detector_weights):
                self.detector = DetectorWrapper(detector_weights, device=device)
                # Move detector's internal model to the correct device (handled in DetectorWrapper init)
            else:
                print(f"Warning: Detector weights path not found: {detector_weights}. Detector will be unavailable.")
        else:
            print("Warning: No valid detector weights path provided. Detector will be unavailable.")

        # Load pre-trained weights
        self._load_weights_from_checkpoint(self.sr_fast, sr_fast_weights, "SR_Fast")
        self._load_weights_from_checkpoint(self.sr_quality, sr_quality_weights, "SR_Quality")
        self._load_weights_from_checkpoint(self.masker, masker_weights, "Masker")

        # Validate config after initialization
        try:
            self._validate_config()
        except ValueError as e:
            print(f"Configuration validation failed: {e}")
            # Decide how to handle invalid config: raise error, use defaults, etc.
            # For now, just print the warning.

    def _load_weights_from_checkpoint(self, model: nn.Module, weights_path: Optional[str], model_name: str):
        """ Helper function to load weights from checkpoint files. """
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                state_dict = None
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"Loading {model_name} weights from 'model_state_dict' key in {weights_path}")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Common alternative key
                    state_dict = checkpoint['state_dict']
                    print(f"Loading {model_name} weights from 'state_dict' key in {weights_path}")
                elif isinstance(checkpoint, dict): # Check if the checkpoint itself is the state_dict
                    # Basic check: are keys typical layer names?
                    if all(isinstance(k, str) for k in checkpoint.keys()):
                         state_dict = checkpoint
                         print(f"Loading {model_name} weights directly from checkpoint dictionary: {weights_path}")
                elif isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict): # Direct state_dict save
                     state_dict = checkpoint
                     print(f"Loading {model_name} weights directly from state_dict file: {weights_path}")

                if state_dict:
                    # Remove 'module.' prefix if present (from DataParallel/DDP)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    # Load with strict=False to be more tolerant to minor mismatches
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"Successfully loaded {model_name} weights.")
                else:
                    print(f"Warning: Could not find a valid state_dict in {weights_path} for {model_name}.")

            except Exception as e:
                print(f"Error loading {model_name} weights from {weights_path}: {e}")
        elif weights_path:
             print(f"Warning: {model_name} weights path not found: {weights_path}")
        else:
             print(f"Info: No weights path provided for {model_name}. Using initialized weights.")

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
             print("Warning: 'threshold' not found in config['model']['masker']. Defaulting might occur.")
        if 'target_sparsity_ratio' not in self.config['train']:
             print("Warning: 'target_sparsity_ratio' not found in config['train']. Defaulting might occur.")


    def forward(self,
                lr_image: torch.Tensor,
                targets: Optional[List[Dict]] = None,
                temperature: float = 1.0,
                hard_mask_inference: bool = False
                ) -> Dict[str, Optional[Any]]:
        """
        前向传播。

        参数:
            lr_image (torch.Tensor): 输入低分辨率图像 (B, C, H, W)。
            targets (list, optional): 目标检测标注 (仅在训练时用于计算损失)。
            temperature (float, optional): Gumbel-Softmax 温度。
            hard_mask_inference(bool): 推理时是否使用硬掩码。

        返回:
            dict: 包含超分图像、粗糙掩码 (用于损失)、检测器原始输出。
        """
        # Ensure input is on the correct device
        lr_image = lr_image.to(self.device)

        # --- Mask Generation ---
        mask_logits = self.masker(lr_image)  # Masker 输出 logits (B, 1, H, W)

        # Determine mask threshold from config, with a default
        mask_threshold = self.config.get('model', {}).get('masker', {}).get('threshold', 0.5)

        mask_coarse_out: Optional[torch.Tensor] = None
        mask_for_fusion: Optional[torch.Tensor] = None

        if self.training:
            # 训练时使用 Gumbel-Softmax 生成软掩码
            # Create 2 channels for Gumbel: [Logit_Quality, Logit_Fast]
            # Assuming Logit_Fast is 0 (or -Logit_Quality if symmetric)
            gumbel_input_logits = torch.cat([mask_logits, torch.zeros_like(mask_logits)], dim=1) # (B, 2, H, W)
            mask_gumbel_output = gumbel_softmax(gumbel_input_logits, tau=temperature, hard=False, dim=1) # Apply Gumbel along channel dim
            mask_soft = mask_gumbel_output[:, 0:1, :, :]  # Take the first channel (probability of Quality path)
            mask_for_fusion = mask_soft
            mask_coarse_out = mask_soft # Use soft mask for loss calculation during training
        else:
            # 推理时
            mask_prob = torch.sigmoid(mask_logits) # Convert logits to probabilities (0 to 1)
            if hard_mask_inference:
                mask_hard = (mask_prob > mask_threshold).float() # Apply threshold for hard mask
                mask_for_fusion = mask_hard
                mask_coarse_out = mask_hard # Output hard mask if used
            else:
                mask_soft = mask_prob # Use probabilities as soft mask
                mask_for_fusion = mask_soft
                mask_coarse_out = mask_soft # Output soft mask

        # --- Super-Resolution ---
        # Ensure sub-models are on the correct device (should be handled in __init__)
        sr_fast_output = self.sr_fast(lr_image)
        sr_quality_output = self.sr_quality(lr_image)

        # --- Mask Upsampling & Fusion ---
        if mask_for_fusion is not None:
            target_size = sr_fast_output.shape[-2:] # Get H, W from SR output
            # Ensure mask_for_fusion is float for interpolation
            mask_for_fusion_resized = F.interpolate(
                mask_for_fusion.float(),
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            sr_image = mask_for_fusion_resized * sr_quality_output + (1 - mask_for_fusion_resized) * sr_fast_output
        else:
            # Handle case where mask generation failed or wasn't performed
            print("Warning: mask_for_fusion is None. Defaulting SR image (e.g., to sr_fast_output).")
            # Choose a default behavior, e.g., use the fast SR output
            sr_image = sr_fast_output
            # Or raise an error if mask is essential
            # raise RuntimeError("Mask generation failed, cannot proceed with fusion.")


        # --- Detection ---
        detection_results: Optional[Union[List[Dict], Tuple]] = None
        detection_loss: Optional[Union[torch.Tensor, Dict]] = None

        if self.detector is not None and self.detector.model is not None:
            # Set detector mode based on ConditionalSR mode
            self.detector.train(self.training)

            if self.training:
                if targets is None:
                     print("Warning: ConditionalSR is in training mode, but no targets were provided for detection loss calculation.")
                     # Proceed with detection inference if needed, but loss will be None
                     detection_results = self.detector(sr_image, targets=None)
                else:
                     # Pass targets for loss calculation
                     detection_results, detection_loss = self.detector(sr_image, targets=targets)
            else:
                # Inference mode for detector
                detection_results = self.detector(sr_image, targets=None)
        else:
            print("Warning: Detector is not available. Skipping detection.")


        return {
            "sr_image": sr_image,
            "mask_coarse": mask_coarse_out, # Mask used for loss (soft/hard)
            "mask_fused": mask_for_fusion_resized if mask_for_fusion is not None else None, # Resized mask used for fusion
            "detection_results": detection_results, # Raw results from detector (preds in train, detections in eval)
            "detection_loss": detection_loss, # Loss tensor/dict in train, None in eval
        }