"""Unitree Go2 robot profile for universal tracking.

抄 data_collect/robots/go2.py 的 profile 范式（PD 40/1、spawn 0.42、
默认站姿 all-hip0 / thigh0.9 / calf-1.8 与 A0 GO2_UNI_DEFAULT_Q 完全等价）。
常量全部模块级（离线可用），只有工厂函数内惰性 import isaaclab：
  - ``make_go2_track_profile`` 才需要 isaaclab assets（AppLauncher 之后调用）；
  - Go2UniTrackEnvCfg / 离线脚本只用常量。

关节序（probed）：Isaac Lab Go2 USD 按类型分组：
    ['FL_hip','FR_hip','RL_hip','RR_hip','FL_thigh',...,'RR_calf'] + '_joint'
足端 body 序 = policy 腿序 FL,FR,RL,RR；body_parts["joints"] 为 motion 序
约定（腿分组三连），与 clip 库/landmark 的 leg 序一致。
"""
from __future__ import annotations

from typing import Dict, Optional

from .robot_profile import RobotProfile

# Isaac Lab 实际关节序（probed；按类型分组，与 UNITREE_GO2_CFG 资产一致）
GO2_JOINT_ORDER_IL = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
# 足端 body（policy 腿序 FL,FR,RL,RR）
GO2_FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# 默认站姿（== A0 GO2_UNI_DEFAULT_Q；== pose_reconstruct go2 跟踪 q0）
GO2_DEFAULT_Q = {
    "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.9, "FR_thigh_joint": 0.9,
    "RL_thigh_joint": 0.9, "RR_thigh_joint": 0.9,
    "FL_calf_joint": -1.8, "FR_calf_joint": -1.8,
    "RL_calf_joint": -1.8, "RR_calf_joint": -1.8,
}
# motion/policy 序 body_parts（joints 腿分组三连；ankles 可配，缺省按 feet 前缀派生）
GO2_BODY_PARTS = {
    "feet": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "base": "base",
    "joints": [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    "hips": ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
}


def make_go2_track_profile(
    spawn_height: float = 0.42,
    stiffness: float = 40.0,
    damping: float = 1.0,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    body_parts: Optional[Dict] = None,
) -> RobotProfile:
    """构建 Go2 跟踪 profile（需要 isaaclab assets，惰性导入）。"""
    # isaaclab 导入放工厂内：常量可离线，只有真正装配 cfg 才拉起 assets
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    bp = dict(GO2_BODY_PARTS)
    if body_parts:
        bp.update(body_parts)

    articulation_cfg = UNITREE_GO2_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, spawn_height),
            joint_pos=dict(GO2_DEFAULT_Q),
            joint_vel={".*": 0.0},
        ),
        actuators={"legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
    )
    return RobotProfile(
        name="go2",
        articulation_cfg=articulation_cfg,
        joint_order=list(GO2_JOINT_ORDER_IL),
        default_joint_pos=dict(GO2_DEFAULT_Q),
        foot_bodies=list(bp.get("feet", GO2_FOOT_BODIES)),
        body_parts=bp,
        metadata={"urdf": "asset/go2/go2.urdf", "pd": (stiffness, damping)},
    )


__all__ = [
    "GO2_JOINT_ORDER_IL",
    "GO2_FOOT_BODIES",
    "GO2_DEFAULT_Q",
    "GO2_BODY_PARTS",
    "make_go2_track_profile",
]