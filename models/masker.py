import torch
import torch.nn as nn

class Masker(nn.Module):
    """
    Masker 网络，用于生成掩码 logits。
    """
    def __init__(self, in_channels=3, base_channels=32, num_blocks=4, output_channels=1):
        super(Masker, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True) # 添加 ReLU 激活
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)]
        )
        # 输出卷积层，输出 logits
        self.output_conv = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.input_conv(x)) # 在输入卷积后添加激活
        x = self.blocks(x)
        logits = self.output_conv(x) # 直接返回 logits (B, 1, H, W)
        return logits

# 示例用法
if __name__ == "__main__":
    model = Masker(in_channels=3, base_channels=32, num_blocks=4, output_channels=1) # 输出 1 通道 logits
    input_tensor = torch.randn(1, 3, 64, 64)
    output_logits = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}, Output logits shape: {output_logits.shape}")