# 宇树官方全身 H1 · 51.65 kg 参数辨识

2026-11-26 · MuJoCo 3.13.0 / Isaac Sim 5.1、Isaac Lab 2.3（RTX 4090）

本包使用 `/home/pjm/resources/robots/h1/h1.xml` 和它的原始 STL。源文件和每份网格的 SHA-256 见 [资产溯源](asset_provenance.json)。移除了不在机器人运动树里的 0.1 kg 占位体、占位电机和对应传感器，并调整了 keyframe 维度。模型保留 19 个关节、20 个刚体、51.649896 kg 机器人本体及官方 300 × 30 mm 足踝碰撞网格。官方刚体质量、质心、惯量、关节轴和限位没有重新拟合。原有 PACE MJCF 和 59.34 kg PhysX USD 均未覆盖。

## 使用

- **MuJoCo**：打开 [校准场景](scene_calibrated.xml) 或 [原始场景](scene_nominal.xml)。模型 XML 和 `assets/` 要放在同一目录。场景是裸仿真，不带站立策略。
- **Isaac Sim**：打开或引用 [校准 USD](h1_unitree_calibrated.usd) 或 [原始 USD](h1_unitree_nominal.usd)。两份 USD 都是自包含的浮动基座模型；外部控制器负责 PD 和限幅，资产不内置运动策略或地面。
- 控制输入、PD 和日志都应按**关节名**映射。官方 MJCF 的执行器排列和旧 PACE 版不同。在训练端也要保留 USD 的 armature/摩擦参数，不要再用统一的 armature 或摩擦覆盖；若训练代码引用旧模型的肘部或足底标记名，需适配 `*_elbow_link_ball_hand`、`*_ankle_link`。本包未改 Holosoma 训练配置，也未验证 PPO 训练。

## 数据和方法

沿用 PACE 的 `datasets/h1/real/full_range_official_manifest.json`（重复采集以清单中的正式重跑为准）。只用每关节两条小幅正弦（共 **38 条**，0.5 和 2 Hz）以 CMA-ES 分关节拟合 `armature`、`frictionloss`、`damping`。其余关节在单关节拟合时保留官方值。每个候选对该关节的两条轨迹等权计算位置 MSE；固定真机记录的 PD，不拟合命令延迟，不使用 39 条留出小幅、5 条大幅验证或 3 条原盲测记录选参。4 条大幅开发数据只旁报，不参与本轮拟合。

两引擎均在固定基座、无地面接触条件下，以 500 Hz 自由回放；每条只用首帧实测角度和速度初始化。两组统一使用记录的 PD 参数、目标角度、角度限位和力矩限幅。Isaac Sim 从同样的资产参数开始，只切换关节动力学；阻尼及关节摩擦映射到 PhysX 的轴属性，未在 PhysX 上二次拟合。读取了物理运行时参数，而不只检查 USD 文本。

## 独立留出回放

表中为平均关节位置 RMSE（rad）；大幅动作先平均每条的活跃关节，再按轨迹等权平均。改善数按整个轨迹统计。

| 评估数据 | MuJoCo：原始 → 校准 | 改善 | Isaac Sim：原始 → 校准 | 改善 |
|---|---:|---:|---:|---:|
| 小幅留出 · 39 条 | 0.009722 → 0.005870 | 39.6%，39/39 | 0.009722 → 0.005318 | 45.3%，37/39 |
| 大幅验证 · 5 条 | 0.009028 → 0.006141 | 32.0%，4/5 | 0.009028 → 0.005682 | 37.1%，4/5 |
| 大幅原盲测 · 3 条 | 0.008388 → 0.005985 | 28.6%，2/3 | 0.008388 → 0.005554 | 33.8%，3/3 |

两种引擎在 90% 幅值的上身肩旋转／肘部动作上都退化（运行记录 `20261120_155811`，MuJoCo 误差 +28.3%，PhysX +23.5%）。MuJoCo 的 0.75 倍速上身盲测也退化 +4.8%；PhysX 的两条小幅退化是右髋俯仰阶跃和腰部阶跃。

两份 MJCF 场景均通过浮动基座双足接触检查；两份 USD 也都在 PhysX 平面上产生双足接触，碰撞网格读回为 300 × 30 mm，最低点 −70 mm。[USD 接触检查](floating_contact_smoke.json) 只验证碰撞正常工作，**不是**落地动作精度验证；接触力峰值来自测试中的自由落地，不能用来评估步态。悬空数据没有辨识地面摩擦、接触柔顺性、负载或步态稳定性。

## 证据和复现

- [中文轨迹图](validation_left_knee.png) · [汇总 JSON](validation_summary.json)
- [MuJoCo 逐项结果](mujoco_joint_metrics.csv) · [Isaac Sim 逐项结果](isaacsim_joint_metrics.csv) · [PhysX 读回](isaacsim_readback.json)
- [19 关节参数和每个关节的拟合记录](joint_parameters.json) · [SHA-256](SHA256SUMS) · [宇树资产许可证](UNITREE_LICENSE)

在项目根目录的 MuJoCo 环境运行（输出目录须是新目录）：

```bash
pixi run -e mujoco python sysid_mujoco/refit_h1_unitree.py \
  --source /home/pjm/resources/robots/h1 --output /tmp/h1_official_refit
pixi run -e mujoco python sysid_isaaclab/export_h1_unitree_usd.py prepare \
  --mjcf /tmp/h1_official_refit/h1_unitree_nominal.xml --payload /tmp/h1_official_geometry.npz
pixi run -e mujoco python sysid_isaaclab/replay_h1_physx_ab.py prepare \
  --unitree-model-dir /tmp/h1_official_refit --output /tmp/h1_official_heldout.npz
```

将几何 NPZ、留出 NPZ 和 `joint_parameters.json` 移至 Isaac Sim 主机。用可导入 `pxr` 的 OpenUSD 环境导出，之后用 Isaac Lab 环境回放：

```bash
python sysid_isaaclab/export_h1_unitree_usd.py export \
  --payload h1_official_geometry.npz --params joint_parameters.json --output ./usd
python sysid_isaaclab/replay_h1_physx_ab.py replay \
  --bundle h1_official_heldout.npz --nominal-usd usd/h1_unitree_nominal.usd \
  --calibrated-usd usd/h1_unitree_calibrated.usd --output ./verification
```
