import torch
import torch.nn as nn

class Masker(nn.Module):
    """
    Masker 网络，用于生成粗粒度掩码 logits。
    """
    def __init__(self, in_channels=3, base_channels=32, num_blocks=4, output_channels=1, output_patch_size=16):
        """
        Args:
            in_channels (int): 输入图像通道数。
            base_channels (int): 基础卷积通道数。
            num_blocks (int): 中间卷积块数量。
            output_channels (int): 输出通道数 (通常为 1)。
            output_patch_size (int): 粗粒度掩码相对于输入 LR 图像的下采样因子。
                                     例如，输入 64x64，output_patch_size=16，则输出掩码尺寸为 4x4。
        """
        super(Masker, self).__init__()
        self.output_patch_size = output_patch_size

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)]
        )
        # 输出卷积层，输出 logits
        self.output_conv = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

        # 添加 AdaptiveAvgPool2d 实现粗粒度输出
        # 输出尺寸将是 (H_in / output_patch_size, W_in / output_patch_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((
            None, # Keep original ratio, or calculate target size
            None
        ))


    def forward(self, x):
        # Calculate target output size based on input size and patch size
        h_in, w_in = x.shape[-2:]
        h_out = max(1, h_in // self.output_patch_size)
        w_out = max(1, w_in // self.output_patch_size)
        self.adaptive_pool.output_size = (h_out, w_out) # Set target size dynamically

        x = self.relu(self.input_conv(x))
        x = self.blocks(x)
        logits = self.output_conv(x) # (B, 1, H_in, W_in)

        # 应用自适应平均池化到目标粗粒度尺寸
        coarse_logits = self.adaptive_pool(logits) # (B, 1, H_in/P, W_in/P)

        return coarse_logits

# 示例用法
if __name__ == "__main__":
    model = Masker(in_channels=3, base_channels=32, num_blocks=4, output_channels=1, output_patch_size=16) # 输出 1 通道 logits
    input_tensor = torch.randn(1, 3, 64, 64) # LR input
    output_logits = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}, Output logits shape: {output_logits.shape}") # Expected output: (1, 1, 4, 4)
    input_tensor_large = torch.randn(1, 3, 128, 128) # Larger LR input
    output_logits_large = model(input_tensor_large)
    print(f"Input shape: {input_tensor_large.shape}, Output logits shape: {output_logits_large.shape}") # Expected output: (1, 1, 8, 8)
# 该模块实现了掩码生成器，用于动态选择SR路径。
# 通过学习图像的不同区域的重要性，掩码生成器能够为每个上采样块提供自适应的处理策略。
# 这种方法允许网络在保持计算效率的同时，专注于图像中最相关的部分。