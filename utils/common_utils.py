# utils/common_utils.py
import torch
import logging
from typing import Optional

def get_device(use_gpu_arg: bool, logger_instance: Optional[logging.Logger] = None) -> torch.device:
    """
    根据参数和可用性确定并返回适当的 torch.device。

    参数:
        use_gpu_arg (bool): 是否尝试使用 GPU 的标志 (通常来自命令行参数)。
        logger_instance (Optional[logging.Logger]): 用于记录日志的 logger 实例。
                                                   如果为 None，则使用此模块的默认 logger。

    返回:
        torch.device: 选择的设备 (cuda 或 cpu)。
    """
    # 使用传入的 logger 或获取此模块的默认 logger
    effective_logger = logger_instance if logger_instance else logging.getLogger(__name__)
    if not effective_logger.hasHandlers(): # 确保 logger 有处理器
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    device: torch.device
    if use_gpu_arg:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            effective_logger.info(f"请求使用 GPU 且 GPU 可用。当前使用设备: {device}")

            # 可选: 记录更多 GPU 详细信息
            num_gpus = torch.cuda.device_count()
            effective_logger.info(f"可用 GPU 数量: {num_gpus}")
            # 记录每个 GPU 的名称
            # for i in range(num_gpus):
            #     effective_logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            effective_logger.info(f"当前 CUDA 默认设备索引: {torch.cuda.current_device()}")
            effective_logger.info(f"当前 CUDA 默认设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            device = torch.device("cpu")
            effective_logger.warning("请求使用 GPU 但 GPU 不可用。将回退到 CPU。")
    else:
        device = torch.device("cpu")
        effective_logger.info("未请求使用 GPU 或 GPU 不可用。当前使用 CPU。")

    return device