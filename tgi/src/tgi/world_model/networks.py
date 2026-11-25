# 定义 TGI 的三个网络。世界模型根据脚下地形预测运动和接触，策略根据观测给出动作，价值网络估计未来回报。

"""World model (dynamics + foot-contact head), policy, and critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import NUM_J
from ..mdp.mdp import INV_DIM, XI_N_DIM, invariant_features
from ..mdp.rotations import so3_exp, so3_log, yaw_mat


# 按给定隐藏层宽度、激活函数和归一化选项搭建全连接网络。
def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    # 为每一维特征保存均值、方差和累计样本数，并设置停止更新的样本数阈值。
    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    # 将新批次的统计量合并到历史均值和方差中，原地更新以兼容 CUDA 图。
    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        # in place: buffers keep their addresses (CUDA-graph replays read them by address)
        self.var.copy_((self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot)
        self.mean += delta * n / tot
        self.count.copy_(tot)

    # 用累计的均值和带数值保护的标准差对特征做标准化。
    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    # 返回实际用于缩放的标准差，供恢复预测量的原始尺度。
    def std(self):
        return torch.sqrt(self.var + self.eps**2)


DYN_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J
NUM_FEET = 2
AMP_DTYPE: torch.dtype | None = None


# 按训练配置选择浮点精度运行网络，混合精度路径将输出转回单精度。
def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic s_{t+1} = f(s_t, a_t, xi) on frame-local increments, plus foot-contact logits c_{t+1}."""

    # 建立同时输出状态增量和双脚接触分数的网络，为输入和动力学增量分别维护归一化统计。
    def __init__(self, terrain_dim: int, hidden=(1024, 1024, 1024)):
        super().__init__()
        self.in_dim = INV_DIM + terrain_dim + 2 * NUM_J + XI_N_DIM
        self.net = mlp(self.in_dim, list(hidden), DYN_OUT + NUM_FEET, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(DYN_OUT, eps=1e-4)

    # 拼接与位置、航向无关的状态特征、地形、当前和上一关节目标，以及随机化的动力学参数。
    def inputs(self, S, tf, a, a_prev, xi_n):
        """tf: terrain features; a_prev: previous action (matters under actuation delay)."""
        return torch.cat((invariant_features(S), tf, a, a_prev, xi_n), -1)

    # 将相邻仿真状态的差异转成局部坐标下的运动增量，用于估计训练目标的尺度。
    @staticmethod
    def targets(S, S1):
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            ((H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ R1),
             (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1), (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1), q1 - q, qd1 - qd), -1)  # fmt: skip

    # 预测并反归一化增量，将其作用于当前状态，返回下一状态和两只脚的接触分数。
    def step(self, S, tf, a, a_prev, xi_n):
        """Returns (S1, contact_logits of S1)."""
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, tf, a, a_prev, xi_n)))
        d = y[..., :DYN_OUT] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., DYN_OUT:]

    # 在参考状态的局部坐标下逐项比较预测与实测运动，用相对旋转计算姿态误差。
    @staticmethod
    def state_error(S_hat, S):
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            ((H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ Rh),
             (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1), (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1), qh - q, qdh - qd), -1)  # fmt: skip


class Policy(nn.Module):
    """Deployable MLP over a short observation history (newest last), with built-in normalization."""

    # 为本体感觉和地形观测分别配置归一化，并建立输出关节动作的网络。
    def __init__(self, obs_dim: int, hist: int = 5, hidden=(512, 256, 128), extra_dim: int = 0):
        super().__init__()
        self.hist, self.obs_dim, self.extra_dim = hist, obs_dim, extra_dim
        self.net = mlp(obs_dim * hist + extra_dim, list(hidden), NUM_J)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.extra_norm = RunningNorm(max(extra_dim, 1))

    # 归一化并展平观测历史，拼接可选的地形观测，输出各关节的动作。
    def forward(self, obs_hist: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        x = self.obs_norm(obs_hist).flatten(-2)
        if self.extra_dim:
            x = torch.cat((x, self.extra_norm(extra)), -1)
        return run_mlp(self.net, x)


class Critic(nn.Module):
    # 建立带输入归一化的标量价值网络，用于估计想象轨迹的后续回报。
    def __init__(self, in_dim: int, hidden=(512, 256)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    # 根据包含地形和接触信息的输入，为每个状态输出一个价值估计。
    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
