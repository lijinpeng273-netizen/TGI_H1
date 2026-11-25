# 利用持续写入的仿真回放训练落脚点预测器。生成标签和网络更新以不同频率执行，因此采样时要检查标签是否仍对应原数据。

"""Continual FPM training from the replay buffer (labels refreshed every ``fpm_label_every`` iterations)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..mdp.mdp import unflatten_state
from .labels import foothold_labels
from .model import FootholdPredictor


class FPMTrainer:
    # 配置回归模型的优化器与 Huber 损失阈值，预留标签、有效掩码和行序号缓存。
    def __init__(self, model: FootholdPredictor, lr: float = 1e-3, delta: float = 0.01):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
        self.loss_fn = nn.HuberLoss(delta=delta)
        self.delta = delta
        self.labels = None
        self.valid = None
        self.stamp = None

    # 从回放历史重算标签，筛出具有有效双脚目标且下一行仍在同一回合的候选样本。
    def refresh_labels(self, buffer, kin):
        self.labels, self.valid, self.stamp = foothold_labels(buffer, kin)
        # labelled (prev, env) pairs with a successor in the same episode (rows are overwritten oldest-first)
        age = torch.arange(buffer.size - 1, device=buffer.device)
        prev, row = buffer._row(age), buffer._row(age + 1)
        ok = self.valid[prev] & (buffer.data["ep_step"][row] > 0)
        a_idx, env = ok.nonzero(as_tuple=True)
        self.cand_prev, self.cand_env = prev[a_idx], env
        self.n_cand = int(env.numel())

    # 配对前一行的落点标签与下一行的输入状态，优先选取尚未被环形缓冲区覆盖的标签。
    def sample(self, buffer, mdp, B: int):
        """States s_r with ep_step > 0 (so s_r continues s1_{r-1}) and their labels, uniform over the pairs whose label
        is still current; returns (x, y, weight) with weight 0 for the rare stale draw (single pass, no host sync)."""
        k = torch.randint(0, self.n_cand, (2 * B,), device=buffer.device)
        prev, env = self.cand_prev[k], self.cand_env[k]
        fresh = buffer.row_stamp[prev] == self.stamp[prev]
        sel = torch.argsort((~fresh).to(torch.int8), stable=True)[:B]  # fresh draws first (>= ~70% of draws are fresh)
        prev, env, w = prev[sel], env[sel], fresh[sel].float()
        row = (prev + 1) % buffer.T
        x = mdp.system_state(unflatten_state(buffer.data["s"][row, env]), buffer.data["cmd"][row, env])
        return x, self.labels[prev, env], w

    # 按批更新输入统计量并拟合双脚位置，计算损失时给已过期的标签零权重。
    def update(self, buffer, mdp, B: int, steps: int = 2):
        loss = torch.zeros((), device=buffer.device)
        if not self.n_cand:
            return loss
        for _ in range(steps):
            x, y, w = self.sample(buffer, mdp, B)
            self.model.norm.update(x)
            per = nn.functional.huber_loss(self.model(x), y, reduction="none", delta=self.delta).mean((-2, -1))
            loss = (per * w).sum() / w.sum().clamp(min=1.0)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return loss.detach()
