import torch

try:
    from thop import profile
except ImportError:
    raise ImportError("请先安装 thop 库: pip install thop")

def count_model_flops(model: torch.nn.Module, input_size: tuple):
    """
    计算模型的 FLOPs 和参数数量。

    参数:
        model (torch.nn.Module): 待评估的模型。
        input_size (tuple): 输入张量的尺寸，例如 (1, 3, 64, 64)。
    
    返回:
        flops (float): 模型前向传播时的 FLOPs 数量。
        params (float): 模型参数总数。
    """
    dummy_input = torch.randn(input_size)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params

if __name__ == "__main__":
    # 示例：计算 ConditionalSR 模块的 FLOPs 和参数数量
    from models.sr_fast import SRFast
    from models.sr_quality import SRQuality
    from models.masker import Masker
    from models.conditional_sr import ConditionalSR

    # 实例化各子模块（使用默认参数）
    sr_fast = SRFast(scale_factor=4)
    sr_quality = SRQuality(scale_factor=4)
    masker = Masker(in_channels=3, base_channels=32, num_blocks=4)
    # detector_weights 可传占位空字符串
    from ultralytics import YOLO
    # 为避免使用实际 detector，传入空模型路径，并忽略 detector 部分计算
    conditional_sr = ConditionalSR(
        sr_fast=sr_fast,
        sr_quality=sr_quality,
        masker=masker,
        detector_weights="",
        sr_fast_weights="",
        sr_quality_weights="",
        masker_weights=None,
        device="cpu"
    ).eval()

    input_size = (1, 3, 64, 64)
    flops, params = count_model_flops(conditional_sr, input_size)
    print(f"FLOPs: {flops:.2f}, Params: {params:.2f}")