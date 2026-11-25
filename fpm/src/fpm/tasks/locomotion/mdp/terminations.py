# 定义仿真异常及到达走廊终点的判据，为环境的终止管理器生成逐环境布尔标记。

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 检查基座位置与线速度、关节位置与速度，只要含有非有限值就标记该环境。
def nan_physics_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    bad = ~torch.isfinite(asset.data.root_pos_w.torch).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.root_lin_vel_b.torch).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.joint_pos.torch).all(dim=-1)
    bad |= ~torch.isfinite(asset.data.joint_vel.torch).all(dim=-1)
    return bad


# 当基座相对环境原点的水平直线距离达到阈值时标记到达，供走廊验证统计 reached。
def displacement_from_origin(
    env: ManagerBasedRLEnv, distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2], dim=1) >= distance
