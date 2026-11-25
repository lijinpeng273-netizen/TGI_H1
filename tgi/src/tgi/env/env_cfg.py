# 把 FPM 的 G1 任务改配为使用 Newton 的 TGI 数据采集环境，调整训练地形组成，并让关节目标可延迟若干物理子步生效。

"""The base G1 env configured for world-model training on Newton: terrains, robot USD and lagged actions
(domain randomization is applied by ``tgi.env.sim``).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..constants import G1_NEWTON_USD


# 从 FPM 训练地形派生新配置，为等高和有高差的踏脚石分配列数，并可加入出坑地形。
def isaac_terrains_cfg(stones_share: float = 0.6, num_cols: int = 30, pit_step: float = 0.0):
    """The base env's training terrains, with stepping stones on ``stones_share`` of the columns.

    Columns are assigned explicitly so that every terrain type gets at least one; ``pit_step`` > 0 adds a
    step-out-of-a-pit terrain (5 to 40 cm deep) on one extra column.
    """
    from fpm.terrains import TRAINING_TERRAINS_CFG

    from .terrains import isaac_stepping_stones

    cfg = TRAINING_TERRAINS_CFG.copy()
    subs = dict(cfg.sub_terrains)
    subs["stepping_stones"] = isaac_stepping_stones()  # stone tops +-5 cm at the hardest level
    subs["stepping_stones_flat"] = isaac_stepping_stones(stone_height_limit=(0.0, 0.0))  # all tops level
    if pit_step > 0:
        from isaaclab.terrains import MeshPitTerrainCfg

        subs["pit_step"] = MeshPitTerrainCfg(proportion=pit_step, pit_depth_range=(0.05, 0.40), platform_width=2.0)
    n_stones = round(stones_share * num_cols)
    num_cols += pit_step > 0  # the pit gets its own extra column, the other types keep theirs
    n_flat = n_stones // 2
    STONES = ("stepping_stones", "stepping_stones_flat")
    others = [k for k in subs if k not in STONES]
    base = TRAINING_TERRAINS_CFG.sub_terrains
    w = np.array([base[k].proportion if k in base else subs[k].proportion for k in others])
    free = num_cols - n_stones - len(others)  # one column each first
    share = w / w.sum() * free
    cols = 1 + np.floor(share).astype(int)
    for i in np.argsort(-(share - np.floor(share)))[: free - int(np.floor(share).sum())]:
        cols[i] += 1
    for k, c in zip(others, cols):
        subs[k].proportion = float(c) / num_cols
    subs["stepping_stones"].proportion = (n_stones - n_flat) / num_cols
    subs["stepping_stones_flat"].proportion = n_flat / num_cols
    cfg.sub_terrains = subs
    cfg.num_cols = num_cols
    return cfg


# 选择 Newton 物理和对应 G1 资产，接入训练地形、延迟动作和额外终止判据，由 TGI 自行构造策略观测。
def make_isaac_env_cfg(num_envs: int, seed: int = 0, stones_share: float = 0.6, episode_s: float = 20.0,
                       pit_step: float = 0.0):
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

    import fpm.tasks.locomotion.mdp as fmdp
    from fpm.tasks.locomotion.config.g1.env_cfg import FpmEnvCfg

    cfg = FpmEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = episode_s
    cfg.sim.physics = RoughPhysicsCfg().newton_mjwarp
    # the base env's G1, with collision on the capsules (scripts/convert_g1_usd.py)
    assert os.path.exists(G1_NEWTON_USD), f"{G1_NEWTON_USD} missing: run scripts/convert_g1_usd.py once (README)"
    cfg.scene.robot = cfg.scene.robot.replace(spawn=cfg.scene.robot.spawn.replace(usd_path=G1_NEWTON_USD))
    cfg.scene.terrain.terrain_generator = isaac_terrains_cfg(stones_share, pit_step=pit_step)
    cfg.scene.terrain.max_init_terrain_level = None
    # keep the torso height scanner: Newton's contact sensor needs it
    for k in ("terrain_levels", "gravity_offset_level"):
        setattr(cfg.curriculum, k, None)
    for k in ("physics_material", "add_base_mass", "base_com", "default_joint_pos_offset", "actuator_pd_gains",
              "joint_friction", "gravity_assist", "constant_disturbance", "push_robot"):
        setattr(cfg.events, k, None)

    @configclass
    class _Obs:
        @configclass
        class _Policy(ObservationGroupCfg):  # the pipeline builds its own observations from the state
            base_lin_vel = ObservationTermCfg(func=fmdp.base_lin_vel)

        policy: _Policy = _Policy()

    cfg.observations = _Obs()
    cfg.actions.joint_pos = JointPositionActionCfg(class_type=_lagged_action_cls(), asset_name="robot",
                                                   joint_names=[".*"], scale=0.25, use_default_offset=True)
    cfg.terminations.wm_extra = TerminationTermCfg(func=_extra_termination)
    return cfg


# 调用仿真包装器上设置的额外终止条件，未配置时返回全假的终止标记。
def _extra_termination(env) -> torch.Tensor:
    sim = getattr(env, "_loco_sim", None)
    if sim is None or sim.extra_termination is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    from ..mdp.mdp import unflatten_state

    return sim.extra_termination(unflatten_state(sim.flat_state()))


# 构造支持逐环境执行器延迟的关节位置动作类，供 Isaac Lab 动作管理器使用。
def _lagged_action_cls():
    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    class _Lagged(JointPositionAction):
        """Joint targets reach the actuators ``lag`` physics steps late (per env), as mjlab's actuator delay."""

        # 初始化每个环境的延迟步数，并缓存上一关节目标和当前子步编号。
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.lag = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._prev = self._offset.expand(env.num_envs, -1).clone()
            self._k = 0

        # 新控制周期到来时保留上一目标，处理新动作并重新开始计数子步。
        def process_actions(self, actions):
            self._prev = self.processed_actions.clone()
            super().process_actions(actions)
            self._k = 0

        # 在延迟期内继续施加上一目标，到达设定子步后切换到新目标。
        def apply_actions(self):
            tgt = torch.where((self._k < self.lag).unsqueeze(-1), self._prev, self.processed_actions)
            self._asset.set_joint_position_target_index(target=tgt, joint_ids=self._joint_ids)
            self._k += 1

        # 为重置的环境恢复默认关节目标，避免沿用上一回合的动作。
        def reset(self, env_ids=None):
            super().reset(env_ids)
            ids = slice(None) if env_ids is None else env_ids
            self._prev[ids] = self._offset.expand(self.num_envs, -1)[ids]
            self._processed_actions[ids] = self._offset.expand(self.num_envs, -1)[ids]

    return _Lagged
