import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    残差块，用于构建 EDSR 的主体部分。
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + identity

class SRQuality(nn.Module):
    """
    高质量超分辨率网络，仿照 EDSR。
    """
    def __init__(self, scale_factor=4, in_channels=3, num_channels=64, num_blocks=16):
        super(SRQuality, self).__init__()
        self.scale_factor = scale_factor

        # 特征提取层
        self.head = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)

        # 残差块
        self.body = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)],
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        )

        # 上采样层
        self.upsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

        self.tail = nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        # 遵循 EDSR 的典型残差结构，将 head 的输出作为 body 的残差输入
        identity = x
        x = self.body(x)
        x = x + identity # 应用残差
        x = self.upsample(x)
        x = self.tail(x)
        return x

# 示例用法
if __name__ == "__main__":
    model = SRQuality(scale_factor=4)
    print(model)
    dummy_input = torch.randn(1, 3, 64, 64) # Batch, Channels, Height, Width
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # 期望输出通道数为3