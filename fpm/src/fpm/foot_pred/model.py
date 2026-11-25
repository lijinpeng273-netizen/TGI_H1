# 定义落脚点回归网络，以及将训练集输入均值和标准差一起保存到模型中的推理封装。

import torch
import torch.nn as nn


# 用带层归一化和激活函数的全连接层提取状态特征，最后用线性层回归坐标。
def foot_predictor_mlp(input_size: int, output_size: int, hidden_dims: tuple[int, ...] = (128, 128)) -> nn.Sequential:
    """The FPM network: Linear-LayerNorm-LeakyReLU blocks and a linear output (PyTorch default initialization)."""
    layers, last = [], input_size
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
        last = h
    layers.append(nn.Linear(last, output_size))
    return nn.Sequential(*layers)


class FootPredictor(nn.Module):
    # 构造回归网络，对权重矩阵作正交初始化并将偏置置零。
    def __init__(self, input_size: int, output_size: int, hidden_dims: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.net = foot_predictor_mlp(input_size, output_size, hidden_dims)
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # 将每个样本的状态展平，输出各脚的预测坐标。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class NormalizedFootPredictor(nn.Module):
    # 把训练集的均值和标准差注册为缓冲区，使其随模型一起保存和加载。
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("in_mean", mean.reshape(-1))
        self.register_buffer("in_std", std.reshape(-1))

    # 用保存的统计量归一化输入后，调用核心网络预测落点。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.in_mean) / self.in_std)
