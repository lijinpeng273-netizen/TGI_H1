# 在均匀速度指令采样的基础上加入原地转向，使部分环境保留转向指令、同时将平面线速度设为零。

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
    cfg: TurnInPlaceVelocityCommandCfg

    # 为每个并行环境维护是否进入原地转向模式的标记。
    def __init__(self, cfg: TurnInPlaceVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_turning_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # 沿用基础速度指令的采样方式，再按配置比例抽选本次原地转向的环境。
    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        n = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)  # 3.0 passes slices
        r = torch.empty(n, device=self.device)
        self.is_turning_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turning_envs

    # 更新速度指令后将转向环境的平面线速度清零。
    def _update_command(self):
        super()._update_command()
        turning_ids = self.is_turning_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[turning_ids, :2] = 0.0


# 将原地转向环境的采样比例暴露给任务配置。
@configclass
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TurnInPlaceVelocityCommand
    rel_turning_envs: float = 0.0
