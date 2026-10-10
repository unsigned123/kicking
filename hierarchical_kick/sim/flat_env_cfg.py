# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# VENDORED + UPPER ADAPTED — hierarchical_kick.sim.flat_env_cfg
#
# 上游：isaac-sim/IsaacLab @ v2.3.0
#      （commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77）
# 原文 A：isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
#         config/go2/rough_env_cfg.py   （→ UnitreeGo2RoughEnvCfg）
# 原文 B：同目录 flat_env_cfg.py            （→ UnitreeGo2FlatEnvCfg 谱系）
#
# ## VENDOR 段 diff（UnitreeGo2RoughEnvCfg 相对官方原文 A；逐处） ##
#   2) import 面：官方 ``isaaclab_tasks.manager_based.locomotion.velocity.
#       velocity_env_cfg`` → 本包 ``hierarchical_kick.sim.
#       velocity_env_cfg``（stripped fork 无 locomotion 包）。
#   3) robot：官方 ``UNITREE_GO2_CFG.replace(prim_path=...)``（DCMotor
#       0.5/0.5）→ tracking 装配单源 ``make_go2_track_profile()``
#       （ImplicitActuator 40/1 无 effort_limit、spawn 0.42、stance
#       thigh0.9/calf-1.8——decoder 物理契约）+ 补挂
#       ``soft_joint_pos_limit_factor=0.9``（go2.yaml 契约值；单本体工厂
#       缺省 1.0，multi/embodiments 装配为 0.9——对齐 decoder 训练装配）。
#   其余（terrain 缩放 / push_robot=None / add_base_mass(-1,3) /
#   base_com=None / reset_base 零速度 / reset_robot_joints(1,1) /
#   feet_air_time ".*_foot" / undesired_contacts=None / dof_torques -2e-4 /
#   track 1.5/0.75 / dof_acc -2.5e-7 / base_contact=base）= 官方原文逐行。
#
# ## UnitreeGo2FlatEnvCfg = 官方原文 B 逐行，仅一处 diff：
#   4) ``self.observations.policy.height_scan = None`` 行删（UPPER 层
#      observations 整组替换为 UpperObservationsCfg——无 height_scan 项，
#      对不存在属性置 None 会 AttributeError；语义已由整组替换覆盖）。
#
# ## UPPER 适配层（Go2UpperFlatEnvCfg） ##
#   1) obs 整组替换：官方 48+height_scan 组 → 48 维单帧 policy 组
#       （decoder proprio 42 ⊕ base_lin_vel 3 ⊕ cmd 3；逐通道参数 =
#       core/contracts 适配表，⚠️ 官方 velocity 项参数不照抄处见注释）。
#   2) actions.joint_pos.scale：官方标量 0.25 → 逐关节 dict
#       （scale_j = 0.25×effort_j/KP；decoder 训练语义零变换差）。
#   3) commands：heading_command=False（官方 True）。
#   4) terminations：+ root_height(0.15)（官方 flat 仅 time_out +
#       base_contact；fall_height 0.15 对齐，0.2 会误杀步态波谷）。
#   5) rewards：+ survival（恒+1.0，weight=β1=0.25）。
#   6) commands.base_velocity.debug_vis=False（vendor 保真原文 True 不动；
#       headless 材质系统缺 libXt/libGLU → goal marker preview
#       surface spawn 死旋——可视化与训练语义无关）。
#   未触碰：episode_length_s=20 / decimation=4 / sim.dt=0.005 / 命令
#   U(-1,1)³ 10s / 2% 站桩 / push 关（官方 go2 谱系原值）。
# ---------------------------------------------------------------------------

from isaaclab.utils import configclass

from hierarchical_kick.core.contracts import (
    NOISE_BASE_ANG_VEL,
    NOISE_BASE_LIN_VEL,
    NOISE_JOINT_POS,
    NOISE_JOINT_VEL,
    NOISE_PROJ_GRAVITY,
    SCALE_BASE_ANG_VEL,
    SCALE_JOINT_VEL,
)
from hierarchical_kick.core.scale_vec import go2_action_scale_table
from hierarchical_kick.sim import mdp
from hierarchical_kick.sim.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)
from hierarchical_kick.robots.go2 import make_go2_track_profile

##
# Pre-defined configs
##
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # isort: skip
from isaaclab.managers import ObservationTermCfg as ObsTerm  # isort: skip
from isaaclab.managers import RewardTermCfg as RewTerm  # isort: skip
from isaaclab.managers import TerminationTermCfg as DoneTerm  # isort: skip
from isaaclab.utils.noise import UniformNoiseCfg as Unoise  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """官方 go2 rough override（vendor 谱系；diff 见文件头 2/3）。"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # [3] tracking 装配单源（decoder 物理契约）替代官方
        # UNITREE_GO2_CFG（DCMotor 0.5/0.5 = 不同物理，禁用面）
        _profile = make_go2_track_profile(prim_path="{ENV_REGEX_NS}/Robot")
        _profile.articulation_cfg.soft_joint_pos_limit_factor = 0.9
        self.scene.robot = _profile.articulation_cfg
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small（官方原文；
        # flat 化后 terrain_generator=None，本行不生效——vendor 保真保留）
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale（官方原文标量 0.25；Upper 层换逐关节 dict）
        self.actions.joint_pos.scale = 0.25

        # event（官方 go2 rough 原文逐行）
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards（官方 go2 rough 原文逐行）
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations（官方 go2 rough 原文）
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """官方 go2 flat 谱系（原文 B 逐行，零 diff）。"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards（官方原文）
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat（官方原文）
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan（官方原文：self.observations.policy.height_scan =
        # None；UPPER 层 observations 整组替换为 UpperObservationsCfg
        # ——无 height_scan 项，语义已覆盖，对不存在属性置 None 会炸）
        self.scene.height_scanner = None
        # no terrain curriculum（官方原文）
        self.curriculum.terrain_levels = None


# ---------------------------------------------------------------------------
# UPPER 适配层（非 vendor）—— 通道适配表落地
# ---------------------------------------------------------------------------
@configclass
class UpperPolicyCfg(ObsGroup):
    """上层 policy obs（48 维单帧）：decoder proprio 42（首 term）⊕
    base_lin_vel 3 ⊕ velocity_commands 3。

    顺序保留（adapter 表列号）；42 前缀切片契约 = UpperActor.proprio_dim。
    单帧 = 组级 history 缺省（v2.3.0 官方 velocity 组无 history_length
    覆盖，term 缺省 1）；与官方组级 history=10 的形态不同——decoder 契约是单帧 42。
    """

    # -- decoder proprio 42（首 term；共享 noised obs，绝不再加第二噪声源） --
    # [表#1] joint_pos (0:12)：U±0.01，scale 1.0（官方同参）
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-NOISE_JOINT_POS, n_max=NOISE_JOINT_POS))
    # [表#2] joint_vel (12:24)：U±0.5 × scale 0.05 —— ⚠️ 官方项 U±1.5×0.01
    #        不照抄（decoder 契约）
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-NOISE_JOINT_VEL, n_max=NOISE_JOINT_VEL),
        scale=SCALE_JOINT_VEL,
    )
    # [表#3] base_ang_vel (24:27)：U±0.2 × scale 0.2 —— ⚠️ 官方 scale=1 不照抄
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-NOISE_BASE_ANG_VEL, n_max=NOISE_BASE_ANG_VEL),
        scale=SCALE_BASE_ANG_VEL,
    )
    # [表#4] projected_gravity (27:30)：U±0.05，scale 1.0（官方同参）
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-NOISE_PROJ_GRAVITY, n_max=NOISE_PROJ_GRAVITY),
    )
    # [表#5] last_action (30:42)：无噪声；= 本步采样动作（pre-scale，
    #        解码 mean + 探索噪声）；done reset 清零由 action term 继承
    actions = ObsTerm(func=mdp.last_action)
    # -- 策略面 --
    # [表#6] base_lin_vel (42:45)：U±0.1（官方同参）
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-NOISE_BASE_LIN_VEL, n_max=NOISE_BASE_LIN_VEL))
    # [表#7] cmd (45:48)：UniformVelocityCommand 3 维 vx/vy/ωz
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class UpperObservationsCfg(ObservationsCfg):
    """obs 面：policy 组整体替换。"""

    policy: UpperPolicyCfg = UpperPolicyCfg()


@configclass
class Go2UpperFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """π_upper velocity-flat go2 环境。

    官方 flat 谱系 + 五处 UPPER 适配（文件头 1–5 表）。
    """

    observations: UpperObservationsCfg = UpperObservationsCfg()

    def __post_init__(self):
        # post init of parent（官方 flat 谱系全链）
        super().__post_init__()

        # [2] 逐关节语义 scale（decoder 训练语义：0.25×effort/KP；PhysX 槽位序
        # 键面 = core/scale_vec 单源）
        self.actions.joint_pos.scale = go2_action_scale_table()

        # [3] heading_command=False（官方 True）
        self.commands.base_velocity.heading_command = False

        # [4] 自建 height 终止 0.15（官方 flat 无 height 项；fall_height
        # 对齐 tracking 契约——0.2 会误杀步态波谷）
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height, params={"threshold": 0.15})

        # [5] survival 项（恒 +1.0 func × weight=β1=0.25）
        self.rewards.survival = RewTerm(func=mdp.alive_reward, weight=0.25)

        # [6] debug 可视化关闭（headless 材质 spawn 死旋；见文件头注）
        self.commands.base_velocity.debug_vis = False


__all__ = [
    "UnitreeGo2RoughEnvCfg",
    "UnitreeGo2FlatEnvCfg",
    "UpperPolicyCfg",
    "UpperObservationsCfg",
    "Go2UpperFlatEnvCfg",
]
