# 在 GPU 环形缓冲区中按时间和环境保存仿真转移，为世界模型提供连续训练片段，为策略提供想象起点。

"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from ..constants import NUM_J, STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (NUM_J, torch.float32),
    "a_prev": (NUM_J, torch.float32),
    "c1": (2, torch.float32),
    "cmd": (3, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    # 预分配状态、动作、接触等字段，并为每个物理行保存能识别覆盖的写入序号。
    def __init__(self, T: int, N: int, device: str, extra_fields: dict | None = None):
        self.fields = dict(FIELDS, **(extra_fields or {}))
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0
        self.n_added = 0
        self.row_stamp = torch.full((T,), -1, dtype=torch.long, device=device)

    # 一次写入所有并行环境的一步转移，更新行序号并循环移动写指针。
    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.row_stamp[self.ptr] = self.n_added
        self.n_added += 1
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    # 将从最旧样本起算的时间索引换算成环形缓冲区中的物理行号。
    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    # 抽取同一环境的连续片段，用掩码排除回合结束或速度突变之后的时间步。
    def sample_windows(self, B: int, k: int, keys=("s", "s1", "a", "xi", "c1", "done")):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=self.device))
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in keys}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        jump = ((out["s"][:, 1:, 7:13] - out["s1"][:, :-1, 7:13]).abs().amax(-1) > 1e-4).float()
        brk = torch.clamp(d[:, :-1] + jump, max=1.0)
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - brk), -1), -1)
        return out

    # 抽取想象起点及其历史，向前取样时限制在本回合和当前缓冲区的范围内。
    def sample_starts(self, B: int, hist: int, keys=("s", "a_prev", "xi", "cmd", "c1")):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in keys}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "cmd", "c1")}
        return start, hist_d
