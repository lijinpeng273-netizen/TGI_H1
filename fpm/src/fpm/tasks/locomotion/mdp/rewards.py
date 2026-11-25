# 实现 FPM 行走任务的自定义奖励项，分别衡量静止时是否接近默认姿态、动作是否平滑，以及机身是否保持足够高度。

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 在线速度和转向指令都接近零时，根据各关节偏离默认姿态的程度计算奖励。
def stand_still_nominal_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) < threshold) & (cmd[:, 2].abs() < threshold)
    error = torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
    return torch.exp(-error / std) * standing.float()


# 限制相邻动作差的幅值，再求平方和，作为动作平滑惩罚的输入。
def action_rate_l2_bounded(env: ManagerBasedRLEnv, max_delta: float = 5.0) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    delta = torch.nan_to_num(delta, nan=max_delta, posinf=max_delta, neginf=-max_delta).clamp(-max_delta, max_delta)
    return torch.sum(torch.square(delta), dim=1)


# 用基座相对最低一只脚的高度衡量姿态，低于目标高度时降低奖励。
def upright_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.72,
    std: float = 0.06,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lowest_foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].min(dim=1).values
    deficit = (target_height - (asset.data.root_pos_w.torch[:, 2] - lowest_foot_z)).clamp(min=0.0)
    return torch.exp(-torch.square(deficit) / std**2)
