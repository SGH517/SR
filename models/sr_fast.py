import torch
import torch.nn as nn

class SRFast(nn.Module):
    """
    该模块实现了快速超分辨率网络（SR_Fast）。
    """
    def __init__(self, scale_factor=4, in_channels=3, d=56, s=12, m=4):
        super(SRFast, self).__init__()
        self.scale_factor = scale_factor

        # 特征提取层
        self.feature_extraction = nn.Conv2d(in_channels, d, kernel_size=5, padding=2)

        # 下采样层
        self.shrinking = nn.Conv2d(d, s, kernel_size=1)

        # 非线性映射层
        self.mapping = nn.Sequential(
            *[nn.Conv2d(s, s, kernel_size=3, padding=1) for _ in range(m)]
        )

        # 扩展层
        self.expanding = nn.Conv2d(s, d, kernel_size=1)

        # 反卷积上采样层
        self.deconvolution = nn.ConvTranspose2d(d, in_channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.shrinking(x))
        x = self.mapping(x)
        x = self.relu(self.expanding(x))
        x = self.deconvolution(x)
        return x

# 示例用法
if __name__ == "__main__":
    model = SRFast(scale_factor=4)
    print(model)