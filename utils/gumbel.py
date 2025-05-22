import torch
import torch.nn.functional as F

def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> torch.Tensor:
    """
    实现 Gumbel-Softmax 函数。

    Args:
        logits (torch.Tensor): 输入的 logits 张量。
        tau (float): 温度参数，控制分布的平滑程度。
        hard (bool): 是否返回 one-hot 编码的离散值。
        dim (int): 应用 softmax 的维度。

    Returns:
        torch.Tensor: Gumbel-Softmax 采样结果。
    """
    # 生成与 logits 形状相同的 Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    # 添加 Gumbel 噪声到 logits
    y = logits + gumbel_noise
    # 应用 softmax
    y_soft = F.softmax(y / tau, dim=dim)

    if hard:
        # 将 softmax 输出转换为 one-hot 编码
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        # 在前向传播中返回离散值，但在反向传播中保留梯度
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft
# 该模块实现了Gumbel采样相关的工具函数。