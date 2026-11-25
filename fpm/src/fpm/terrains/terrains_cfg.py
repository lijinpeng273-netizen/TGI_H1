# 定义自建地形的参数，并通过各类的 function 字段绑定生成函数。训练和验证使用的具体参数组合见同目录的 __init__.py。

from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains, mesh_terrains


# 配置向中心逐级升高的楼梯，包括踏步高宽、外围留白和中央平台宽度。
@configclass
class PyramidStairsTerrainCfg(SubTerrainBaseCfg):
    function = mesh_terrains.pyramid_stairs_terrain
    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    platform_width: float = 1.0


# 沿用楼梯的尺寸参数，将生成函数切换为向中心逐级下降的版本。
@configclass
class InvertedPyramidStairsTerrainCfg(PyramidStairsTerrainCfg):
    function = mesh_terrains.inverted_pyramid_stairs_terrain


# 用托盘尺寸、铺设密度和间隙控制坑底上方可落脚的平台。
@configclass
class PalletsTerrainCfg(SubTerrainBaseCfg):
    function = mesh_terrains.pallets_terrain
    pallet_height_range: tuple[float, float] = MISSING
    pallet_width_range: tuple[float, float] = MISSING
    pallet_length_range: tuple[float, float] = MISSING
    pallet_thickness: float = 0.05
    density: float = MISSING
    holes_depth: float = -1.0
    gap_distance: float = 0.0
    platform_width: float = 1.0


# 给出不同难度下踏脚石的宽度、间距、高差和坑底高度。
@configclass
class SteppingStonesTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.stepping_stones_terrain
    stone_height_limit: tuple[float, float] = MISSING
    stone_width_range: tuple[float, float] = MISSING
    stone_distance_range: tuple[float, float] = MISSING
    holes_depth_range: tuple[float, float] = MISSING
    platform_width: float = 1.0


# 设置中央方坑或高台的尺寸，以及外围交替出现的地面和沟槽。
@configclass
class PitGapTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.pit_gap_terrain
    pit_depth_range: tuple[float, float] = MISSING
    platform_height_range: tuple[float, float] | None = None
    gap_width_range: tuple[float, float] = MISSING
    pit_width: float = 3.5
    ground_width: float = 0.6
    gap_depth: float = 1.0


# 设置围绕中央平台的方环状沟槽和可落脚地带的宽度、高差。
@configclass
class ConcentricGapTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.concentric_gap_terrain
    gap_width_range: tuple[float, float] = MISSING
    ground_width_range: tuple[float, float] = MISSING
    ground_height_max: float = MISSING
    platform_width: float = 1.0


# 用 wall_type 选墙体、floor_variant 选路面，两者组合成一块十字走廊地形。
@configclass
class CorridorTerrainCfg(HfTerrainBaseCfg):
    function = hf_terrains.corridor_terrain

    wall_type: str = "straight"
    floor_variant: str = "flat"
    corridor_width_range: tuple[float, float] = (2.0, 1.0)
    platform_width: float = 1.0
    edge_taper: float = 1.0

    # 以下参数分别控制连续墙、石柱墙和分段方块墙的几何形状。
    wall_height_range: tuple[float, float] = (1.5, 2.5)
    stone_wall_length: float = 0.20
    stone_wall_gap: float = 0.10
    stone_wall_height: float = 2.0
    wall_pit_depth: float = 3.0
    blocks_width: float = 1.5
    blocks_extra_width: float = 0.20
    block_length: float = 0.30
    block_height_range: tuple[float, float] = (1.0, 1.5)

    # 以下参数由对应的路面生成函数读取，例如踏脚石用 stone_gap_range 调节间隙。
    rough_range: tuple[float, float] = (0.02, 0.10)
    rough_step: float = 0.005
    box_width_range: tuple[float, float] = (0.10, 0.25)
    box_height_range: tuple[float, float] = (0.02, 0.25)
    box_height_step: float = 0.02
    stone_width: float = 0.20
    stone_gap_range: tuple[float, float] = (0.20, 0.20)
    stone_gap_depth: float = 3.0
    stair_rise_range: tuple[float, float] = (0.05, 0.18)
    stair_run: float = 0.30
    stair_max_rise: float = 0.60
    noise_range: tuple[float, float] = (0.05, 0.15)
    noise_step: float = 0.02
    noise_cell: float = 0.1
    slope_range: tuple[float, float] = (0.0, 0.11)
    wave_amplitude_range: tuple[float, float] = (0.05, 0.25)
    wave_period: float = 7.5
    obstacle_width_range: tuple[float, float] = (0.5, 1.0)
    obstacle_height_range: tuple[float, float] = (0.1, 0.2)
    obstacle_spacing_range: tuple[float, float] = (1.0, 2.0)
    pallet_size_range: tuple[float, float] = (0.25, 0.5)
    pallet_height_range: tuple[float, float] = (-0.05, 0.05)
    pallet_gap: float = 0.05
    pallet_pit_depth: float = 1.0
    gap_ground_width_range: tuple[float, float] = (0.3, 0.6)
    gap_width_range: tuple[float, float] = (0.2, 0.35)
    gap_depth: float = 1.0
    gap_ground_height_max: float = 0.05
