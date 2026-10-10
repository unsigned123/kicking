"""Xiaomi Cyberdog2 robot profile for official-convention velocity-flat.

本体参数单源 = xiaomi 成功快照（cyberdog2_rl_lab 仓库）：
  - 资产/执行器/init_state：``assets/robots/cyberdog2.py``（CYBERDOG2_CFG）
  - body names/命令 ranges：成功训练快照
    ``xiaomi_cyberdog2_rl/logs/rsl_rl/cyberdog2_velocity_flat/
    2026-08-13_08-28-43_vf_500_benchmark/params/velocity_env_cfg.py``

rc1 适配（相对 xiaomi 旧 cfg 逐字段）：
  - ``activate_contact_sensors``：旧 = UrdfConverterCfg 字段；3.0rc1 移到
    spawner 基类（isaaclab.sim.spawner.SpawnerCfg，字段同名）——PhysX
    contact reporter 须 spawn 侧激活（isaaclab_physx ContactSensor
    docstring 明示），故 spawn cfg 上显式 True。
  - 其余 UrdfFileCfg 字段（fix_base/merge_fixed_joints/collision_from_visuals/
    replace_cylinders_with_capsules/self_collision/joint_drive/articulation_props/
    rigid_props）rc1 逐字段同名同型，逐值照抄。
  - ``joint_sdk_names`` 自定义 ArticulationCfg 扩展字段不入本 profile
    （部署序契约放 metadata；训练面 obs/动作序 = PhysX 实际序）。

关节序：CYBERDOG2_JOINT_ORDER_IL = PhysX 实际序（probed：URDF 树 BFS
深度分层 abad→hip→knee；层内序 = URDF 声明序 FR,FL,RR,RL——首训 env
build 实测）。SDK 序（FR-first 三连腿分组）仅 metadata 部署契约，训练面
obs/动作序 = PhysX 序（策略端到端同序自洽，无需置换；与 kick 上层装配
时的按名映射在装配层处理）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .robot_profile import RobotProfile

# 包内资产锚（URDF 相对引用 ../meshes/*.dae；布局 = assets/cyberdog2/{urdf,meshes}）
_CYBERDOG2_ASSET_ROOT = Path(__file__).resolve().parent.parent / "assets" / "cyberdog2"
CYBERDOG2_URDF_PATH = _CYBERDOG2_ASSET_ROOT / "urdf" / "cyberdog2.urdf"

# PhysX 实际关节序（probed：URDF→USD BFS 深度分层 abad→hip→knee；层内序
# = URDF 声明序 FR,FL,RR,RL——首训 env build 实测回填，非字母序）
CYBERDOG2_JOINT_ORDER_IL = [
    "FR_abad_joint", "FL_abad_joint", "RR_abad_joint", "RL_abad_joint",
    "FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint",
    "FR_knee_joint", "FL_knee_joint", "RR_knee_joint", "RL_knee_joint",
]
# SDK/部署序（FR-first，xiaomi CYBERDOG2_JOINT_SDK_NAMES 原序；仅 metadata 契约）
CYBERDOG2_JOINT_ORDER_SDK = [
    "FR_abad_joint", "FR_hip_joint", "FR_knee_joint",
    "FL_abad_joint", "FL_hip_joint", "FL_knee_joint",
    "RR_abad_joint", "RR_hip_joint", "RR_knee_joint",
    "RL_abad_joint", "RL_hip_joint", "RL_knee_joint",
]
# 足端 body（policy 腿序 FL,FR,RL,RR；URDF fixed link，merge_fixed_joints=False 独立体）
CYBERDOG2_FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
# base/head body 名（xiaomi 快照 events/rewards body_names 语义）
CYBERDOG2_BASE_BODY = "body"
CYBERDOG2_HEAD_BODY = "head"
# 默认站姿（CYBERDOG2_CFG init_state.joint_pos 逐项展开）
CYBERDOG2_DEFAULT_Q = {
    "FR_abad_joint": -0.10, "FL_abad_joint": 0.10,
    "RR_abad_joint": -0.10, "RL_abad_joint": 0.10,
    "FL_hip_joint": 0.80, "FR_hip_joint": 0.80,
    "RL_hip_joint": 1.00, "RR_hip_joint": 1.00,
    "FL_knee_joint": -1.50, "FR_knee_joint": -1.50,
    "RL_knee_joint": -1.50, "RR_knee_joint": -1.50,
}
# motion/policy 序 body_parts（joints 腿分组三连，与 go2 范式同构）
CYBERDOG2_BODY_PARTS = {
    "feet": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "base": "body",
    "joints": [
        "FL_abad_joint", "FL_hip_joint", "FL_knee_joint",
        "FR_abad_joint", "FR_hip_joint", "FR_knee_joint",
        "RL_abad_joint", "RL_hip_joint", "RL_knee_joint",
        "RR_abad_joint", "RR_hip_joint", "RR_knee_joint",
    ],
    "hips": ["FL_abad_joint", "FR_abad_joint", "RL_abad_joint", "RR_abad_joint"],
}


def make_cyberdog2_track_profile(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    body_parts: Optional[Dict] = None,
) -> RobotProfile:
    """构建 Cyberdog2 velocity profile（需要 isaaclab，惰性导入）。

    资产/执行器/init_state 逐项抄 xiaomi CYBERDOG2_CFG（见模块 docstring
    rc1 适配清单）；spawn z=0.40 与整机 init_state 同源。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import DCMotorCfg
    from isaaclab.assets import ArticulationCfg

    bp = dict(CYBERDOG2_BODY_PARTS)
    if body_parts:
        bp.update(body_parts)

    articulation_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(CYBERDOG2_URDF_PATH),
            # 以下 UrdfFileCfg 字段逐项 = xiaomi Cyberdog2UrdfFileCfg（rc1
            # 同名同型；activate_contact_sensors 见模块 docstring rc1 适配）
            fix_base=False,
            activate_contact_sensors=True,
            merge_fixed_joints=False,
            collision_from_visuals=False,
            replace_cylinders_with_capsules=True,
            self_collision=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # spawn z=0.30：几何站高 FK ≈0.236m（hip0.8/knee−1.5 × 腿段
            # 0.12+0.174+足 r0.019），关节 scale 采样上限站高 ≈0.27——
            # xiaomi 原值 0.40 造成每回合开局 0.16m 自由落体冲击瞬态
            # （站姿探针实测 qd 冲至 31 rad/s + 身体反弹触地团灭），
            # 按本包 URDF 几何校正（deviation 在案）。
            pos=(0.0, 0.0, 0.30),
            joint_pos=dict(CYBERDOG2_DEFAULT_Q),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_abad_joint", ".*_hip_joint", ".*_knee_joint"],
                effort_limit=12.0,
                effort_limit_sim=12.0,
                saturation_effort=12.0,
                velocity_limit=30.9971,
                velocity_limit_sim=30.9971,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        },
    )
    return RobotProfile(
        name="cyberdog2",
        articulation_cfg=articulation_cfg,
        joint_order=list(CYBERDOG2_JOINT_ORDER_IL),
        default_joint_pos=dict(CYBERDOG2_DEFAULT_Q),
        foot_bodies=list(bp.get("feet", CYBERDOG2_FOOT_BODIES)),
        body_parts=bp,
        metadata={
            "urdf": "hierarchical_kick/assets/cyberdog2/urdf/cyberdog2.urdf",
            "pd": (25.0, 0.5),
            "actuator": "DCMotor(saturation 12.0, vel_limit 30.9971)",
            "base_body": CYBERDOG2_BASE_BODY,
            "head_body": CYBERDOG2_HEAD_BODY,
            "joint_order_sdk": list(CYBERDOG2_JOINT_ORDER_SDK),
            "source": "xiaomi_cyberdog2_rl CYBERDOG2_CFG + vf_500_benchmark snapshot",
        },
    )


__all__ = [
    "CYBERDOG2_URDF_PATH",
    "CYBERDOG2_JOINT_ORDER_IL",
    "CYBERDOG2_JOINT_ORDER_SDK",
    "CYBERDOG2_FOOT_BODIES",
    "CYBERDOG2_BASE_BODY",
    "CYBERDOG2_HEAD_BODY",
    "CYBERDOG2_DEFAULT_Q",
    "CYBERDOG2_BODY_PARTS",
    "make_cyberdog2_track_profile",
]
