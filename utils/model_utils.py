# utils/model_utils.py
import torch
import torch.nn as nn
import os
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any

# 获取一个 logger 实例
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # 确保 logger 有处理器
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def load_model_weights(model: nn.Module,
                       weights_path: Optional[str],
                       device: torch.device,
                       model_name: str = "Model",
                       logger_instance: Optional[logging.Logger] = None,
                       strict: bool = False) -> bool:
    """
    从检查点文件加载权重到模型中的辅助函数。

    参数:
        model (nn.Module): 需要加载权重的模型实例。
        weights_path (Optional[str]): 权重文件的路径。如果为 None 或路径不存在，则不加载。
        device (torch.device): 权重加载的目标设备。
        model_name (str): 用于日志记录的模型名称。
        logger_instance (Optional[logging.Logger]): 用于记录日志的 logger 实例。
        strict (bool): 是否严格加载权重 (参考 torch.nn.Module.load_state_dict 的 strict 参数)。

    返回:
        bool: 如果成功加载权重则为 True，否则为 False。
    """
    effective_logger = logger_instance if logger_instance else logger

    if not weights_path:
        effective_logger.info(f"未提供 {model_name} 的权重路径。将使用初始化权重。")
        return False # 不能算成功加载

    if not os.path.exists(weights_path):
        effective_logger.warning(f"{model_name} 权重路径未找到: {weights_path}。将使用初始化权重。")
        return False

    try:
        checkpoint: Any = torch.load(weights_path, map_location=device)
        state_dict: Optional[OrderedDict] = None

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                effective_logger.info(f"从 {weights_path} 的 'model_state_dict' 键加载 {model_name} 权重。")
            elif 'state_dict' in checkpoint: # 常见备用键
                state_dict = checkpoint['state_dict']
                effective_logger.info(f"从 {weights_path} 的 'state_dict' 键加载 {model_name} 权重。")
            elif all(isinstance(k, str) for k in checkpoint.keys()): # 检查点本身可能就是 state_dict
                # 这是一个基本检查，更复杂的检查可能需要查看键名是否符合层名模式
                is_likely_state_dict = True
                # 可以添加更复杂的检查，比如是否有 optimizer_state_dict 等来区分
                if 'optimizer_state_dict' in checkpoint or 'epoch' in checkpoint:
                    is_likely_state_dict = False

                if is_likely_state_dict:
                    state_dict = checkpoint
                    effective_logger.info(f"直接从检查点字典加载 {model_name} 权重: {weights_path}")
                else:
                    effective_logger.warning(f"检查点是字典但未找到 'model_state_dict' 或 'state_dict', "
                                           f"且不像直接的状态字典: {weights_path} for {model_name}."
                                           f" 包含的键: {list(checkpoint.keys())}")

        elif isinstance(checkpoint, OrderedDict): # 直接保存的 state_dict
             state_dict = checkpoint
             effective_logger.info(f"直接从 OrderedDict 类型的 state_dict 文件加载 {model_name} 权重: {weights_path}")
        else:
            effective_logger.warning(f"{model_name} 的检查点文件格式无法识别 (期望是 dict 或 OrderedDict): {weights_path}，类型为: {type(checkpoint)}")


        if not state_dict:
            effective_logger.warning(f"在 {weights_path} 中找不到 {model_name} 的有效 state_dict。")
            return False

        # 移除 'module.' 前缀 (通常来自 DataParallel 或 DDP 保存的模型)
        new_state_dict = OrderedDict()
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

        for k, v in state_dict.items():
            name = k
            if has_module_prefix and k.startswith('module.'):
                name = k[7:] # 移除 `module.`
            elif not has_module_prefix and any(model_k.startswith('module.') for model_k in model.state_dict().keys()) and not name.startswith('module.'):
                # 如果模型期望 'module.' 前缀但检查点没有，可能需要添加（较少见）
                # effective_logger.warning(f"模型 {model_name} 期望 'module.' 前缀，但检查点权重没有。尝试加载...")
                pass # 通常是移除检查点的前缀以匹配无前缀的模型
            new_state_dict[name] = v

        load_info = model.load_state_dict(new_state_dict, strict=strict)

        if (load_info.missing_keys or load_info.unexpected_keys):
            if strict:
                effective_logger.error(f"严格加载 {model_name} 权重失败 (strict=True)。")
            else: # strict=False
                effective_logger.warning(f"加载 {model_name} 权重 (strict=False) 时存在不匹配的键:")
            if load_info.missing_keys:
                log_func = effective_logger.error if strict else effective_logger.warning
                log_func(f"  模型中存在但检查点中缺失的键: {load_info.missing_keys}")
            if load_info.unexpected_keys:
                log_func = effective_logger.error if strict else effective_logger.warning
                log_func(f"  检查点中存在但模型中意外的键: {load_info.unexpected_keys}")
            if strict: return False # 严格加载失败
        else:
            effective_logger.info(f"成功加载 {model_name} 权重从 {weights_path} (strict={strict})。")
        return True

    except Exception as e:
        effective_logger.error(f"从 {weights_path} 加载 {model_name} 权重时发生错误: {e}", exc_info=True)
        return False

def load_full_checkpoint(path: str,
                         device: torch.device,
                         logger_instance: Optional[logging.Logger] = None
                         ) -> Optional[Dict[str, Any]]:
    """
    加载完整的训练检查点文件。

    参数:
        path (str): 检查点文件的路径。
        device (torch.device): 加载检查点到的设备。
        logger_instance (Optional[logging.Logger]): 用于记录日志的 logger 实例。

    返回:
        Optional[Dict[str, Any]]: 加载的检查点字典，如果失败则为 None。
    """
    effective_logger = logger_instance if logger_instance else logger
    if not os.path.exists(path):
        effective_logger.warning(f"检查点文件在 {path} 未找到。无法恢复。")
        return None
    try:
        checkpoint: Dict[str, Any] = torch.load(path, map_location=device)
        effective_logger.info(f"检查点已从 {path} 加载。包含键: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        effective_logger.error(f"从 {path} 加载完整检查点时出错: {e}", exc_info=True)
        return None