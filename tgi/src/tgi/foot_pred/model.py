# 复用 FPM 的网络结构，配上在线更新的输入归一化器，在 TGI 中预测左右脚各自下一次触地的位置。

"""TGI's continually trained FPM: the FPM network (``fpm.foot_pred``) with a running input normalizer.

Input: the 70-dim ``system_state``; output: the next touchdown (x, y) of both feet in the pelvis frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fpm.foot_pred import FootPredictor, NormalizedFootPredictor, foot_predictor_mlp

from ..mdp.interface import SYSTEM_STATE_DIM
from ..world_model.networks import RunningNorm


class FootholdPredictor(nn.Module):
    # 建立从系统状态到四个平面坐标的回归网络，并准备输入统计量。
    def __init__(self, in_dim: int = SYSTEM_STATE_DIM, hidden=(128, 128)):
        super().__init__()
        self.in_dim, self.hidden = in_dim, tuple(hidden)
        self.net = foot_predictor_mlp(in_dim, 4, self.hidden)
        self.norm = RunningNorm(in_dim)

    # 归一化系统状态，将网络输出整理成每只脚一对骨盆坐标系中的 x、y。
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(..., 70) -> (..., 2, 2) footholds in the root frame."""
        return self.net(self.norm(state)).view(*state.shape[:-1], 2, 2)

    # 复制网络权重与当前均值、标准差，构造 FPM 环境能直接使用的推理模型。
    def exportable(self) -> NormalizedFootPredictor:
        """The base env's foot-model format: (B, 70) or (B, 1, 70) -> (B, 4), normalization included."""
        fp = FootPredictor(self.in_dim, 4, self.hidden)
        fp.net.load_state_dict(self.net.state_dict())
        return NormalizedFootPredictor(fp, self.norm.mean.clone(), self.norm.std().clone()).eval()
