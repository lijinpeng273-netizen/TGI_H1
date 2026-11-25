# 在三种墙体与十三种路面的组合上评估前进能力，分别统计距离完成比例、触发终点的比例和因倾倒终止的比例。

"""Forward traversal success through a 1.3 m corridor over 3 wall types and 13 floors."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="FPM-G1-Validate-Corridor-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--rounds", type=int, default=4, help="Episodes per robot.")
parser.add_argument("--save_results", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os

import numpy as np
import torch
from common import make_env_and_policy

from common import hydra_task_config

from fpm.terrains import CORRIDOR_FLOOR_CATEGORY, CORRIDOR_FLOORS, CORRIDOR_TERRAIN_NAMES


# 把并行环境均分给各走廊，每轮记录沿世界 y 轴的最大前进距离和首次终止原因。
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    goal = env_cfg.terminations.reached_goal.params["distance"]
    env, policy, run_dir = make_env_and_policy(args_cli.task, env_cfg, agent_cfg, args_cli.checkpoint, args_cli.device)
    u = env.unwrapped
    dev, n = u.device, u.num_envs
    terrain = u.scene.terrain
    n_types = len(CORRIDOR_TERRAIN_NAMES)
    per_type = n // n_types

    variant = torch.arange(n, device=dev) // per_type
    terrain.terrain_types = variant.long()
    terrain.terrain_levels = torch.zeros_like(variant).long()
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    u.scene.env_origins[:] = terrain.env_origins
    env.reset()

    z_spawn = u.scene["robot"].data.root_pos_w.torch[:, 2].clone()
    tm = u.termination_manager
    round_steps = int(round(u.max_episode_length)) + 20
    rec = {k: [] for k in ("variant", "v_cmd", "dist", "fell", "reached")}

    with torch.inference_mode():
        for rnd in range(args_cli.rounds):
            if rnd:
                env.reset()
            obs = env.get_observations()
            origin_y = u.scene["robot"].data.root_pos_w.torch[:, 1].clone()
            v_cmd = u.command_manager.get_command("base_velocity")[:, 0].abs().clone()
            best = torch.zeros(n, device=dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            reached = torch.zeros(n, dtype=torch.bool, device=dev)
            done_any = torch.zeros(n, dtype=torch.bool, device=dev)
            for t in range(round_steps):
                obs, _, dones, _ = env.step(policy(obs))
                pos = u.scene["robot"].data.root_pos_w.torch
                best = torch.where(done_any, best, torch.maximum(best, pos[:, 1] - origin_y))
                if rnd == 0 and t == 0 and int(((pos[:, 2] - z_spawn) < -0.3).sum()) > n // 2:
                    raise RuntimeError("Robots are falling through the terrain: terrain collision mesh is missing.")
                if dones.any():
                    fell |= tm.get_term("bad_orientation").view(-1) & ~done_any
                    # reached 取环境按水平位移触发的终点标记，crossed 在本轮结束后按前进距离计算。
                    reached |= tm.get_term("reached_goal").view(-1) & ~done_any
                    done_any |= dones.view(-1).bool()
                if bool(done_any.all()):
                    break
            for k, v in (("variant", variant), ("v_cmd", v_cmd), ("dist", best), ("fell", fell), ("reached", reached)):
                rec[k].append(v.cpu().numpy())
            print(f"[INFO] round {rnd + 1}/{args_cli.rounds}: reached {reached.float().mean():.3f}  fell {fell.float().mean():.3f}")

    R = {k: np.concatenate(v) for k, v in rec.items()}
    # 用指令速度与回合时长算出应走距离，并以终点距离为上限，作为 crossed 的分母。
    intended = np.minimum(R["v_cmd"] * u.max_episode_length_s, goal)
    R["crossed"] = np.clip(R["dist"] / np.maximum(intended, 1e-6), 0.0, 1.0)
    floor = np.array([CORRIDOR_TERRAIN_NAMES[v].split("_", 1)[1] for v in R["variant"]])

    # 对选定的一组回合求三项指标的百分比，并返回该组样本数。
    def row(mask):
        return 100 * R["crossed"][mask].mean(), 100 * R["reached"][mask].mean(), 100 * R["fell"][mask].mean(), int(mask.sum())

    print(f"\n{'variant':<26}{'crossed %':>10}{'reached %':>10}{'fell %':>8}{'n':>6}")
    for i, name in CORRIDOR_TERRAIN_NAMES.items():
        print(f"{name:<26}" + "{:>10.1f}{:>10.1f}{:>8.1f}{:>6}".format(*row(R["variant"] == i)))
    print("-" * 60)
    summary = {}
    for cat in ("sparse", "stepped", "continuous"):
        mask = np.isin(floor, [f for f in CORRIDOR_FLOORS if CORRIDOR_FLOOR_CATEGORY[f] == cat])
        summary[cat] = row(mask)
        print(f"{'[' + cat + ']':<26}" + "{:>10.1f}{:>10.1f}{:>8.1f}{:>6}".format(*summary[cat]))
    summary["all"] = row(np.ones_like(floor, dtype=bool))
    print(f"{'[all]':<26}" + "{:>10.1f}{:>10.1f}{:>8.1f}{:>6}".format(*summary["all"]))

    if args_cli.save_results:
        path = os.path.join(run_dir, "corridor_validation.pt")
        torch.save({"records": R, "summary": summary, "terrain_names": CORRIDOR_TERRAIN_NAMES}, path)
        print(f"[INFO] saved {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
