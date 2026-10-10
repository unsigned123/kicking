# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.velocity_official_cfg
#
# 官方 velocity-flat 口径训练面（任务 HK-Velocity-Flat-Unitree-Go2-v0）：
# 冻结低层 official 口径 ckpt 产出面。基类 = site-packages isaaclab_tasks
# 的 ``UnitreeGo2FlatEnvCfg``（core/velocity/config/go2，
# 官方 Isaac-Velocity-Flat-Unitree-Go2-v0 同一 cfg 类），``__post_init__``
# 先修 rc1 runtime（见下 R1-R3），再覆写回官方 obs/动作/命令口径：
#
#   ① observations.policy.joint_vel.scale = 1.0（官方 raw rad/s；SP 链
#      本就无 scale——显式 1.0 = train_velocity.py 装配自检锚）
#   ② observations.policy.history_length = None（单帧直出、无 CircularBuffer；
#      SP 链 PolicyCfg 无 history 覆盖——显式 None = 装配自检锚）
#   ③ actions.joint_pos.scale = 0.5（官方基座值；SP 链 rough 段降 0.25——
#      回官方均匀 0.5，与冻结低层集成语义 default + 0.5⊙action 对锚）
#   ④ commands.base_velocity.heading_command = False（直接 wz
#      跟踪，匹配 kick 部署语义；上游链缺省 True——须显式关）
#
# ## [isaaclab 3.0rc1 runtime 修复]（上游链为 newton-era
#    isaaclab_tasks——构造期合法、env build 期三重 _Preset 拒收，镜像
#    kick 面 sim/velocity_env_cfg.py 的修法）##
#   R1) sim.physics：上游链 = RoughPhysicsCfg（PresetCfg，newton/ovphysx
#       多后端打包）→ rc1 SimulationCfg.physics 只认 PhysxAutoCfg。
#       修法 = PhysxAutoCfg(isaacsim_physx=PhysxCfg(
#       gpu_max_rigid_patch_count=10*2**15))——数值与 RoughPhysicsCfg.
#       isaacsim_physx 逐字同源（kick 面 velocity_env_cfg.py 同款）；
#       super 链 flat 段的 newton_mjwarp 调参随 RoughPhysicsCfg 一并弃用。
#       同时 sim.use_newton_actuators 关回 False（rough 段置 True；
#       我方资产 = 标准 ImplicitActuatorCfg physx 直驱，kick env 同款）。
#   R2) events.base_com：上游链 = preset(default=EventTerm(...))（_Preset
#       被 rc1 EventManager 拒收）→ 解包 ``.default``（preset 仅含 default
#       键 → 解包语义 = 上游 EventTerm 逐字）。
#   R3) scene.robot：上游链 UNITREE_GO2_CFG = DCMotorCfg + 官方异 default q
#       + actuators["base_legs"].armature=_Preset（rc1 拒收）→ **整只替换
#       为包内 go2 profile**（make_go2_track_profile：Implicit kp40/d 1
#       + default q (0,0.9,-1.8) + soft_joint_pos_limit_factor 0.9 =
#       kick env flat_env_cfg.py 同款装配）。资产取舍：低层与 kick env
#       资产一致优先，官方 obs/动作/奖励约定不变（"官方口径 + 本包资产"）。
#
# 其余（奖励/终止/事件/命令/flat 化/仿真参数）原样继承上游链官方谱系
# （push 事件、±25% log-uniform 质量随机化、reset ±0.5m/±3.14rad、
# commands ranges ±1.0 等）。
# ---------------------------------------------------------------------------

"""Official-convention velocity-flat training cfg (frozen low-level source)."""

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

# 旧路径 manager_based.locomotion 层级已移除：
#   isaaclab_tasks.core.velocity.config.go2.flat_env_cfg（类名不变）
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import (  # noqa: E501
    UnitreeGo2FlatEnvCfg,
)

from hierarchical_kick.robots.go2 import make_go2_track_profile


@configclass
class VelocityOfficialGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """官方口径 velocity-flat 面：rc1 runtime 修复 + 上游口径覆写。

    产物：官方 48 维单帧 policy obs（linvel→ang_vel→gravity→cmd→q→qd→
    last_action，joint_vel raw）+ 均匀 0.5 动作 scale 的 locomotion
    checkpoint（core/low_level_actor 官方口径装载体消费）。
    """

    def __post_init__(self):
        # post init of parent（官方 flat 谱系全链：plane terrain、无
        # height_scan、无 terrain 课程、flat 奖励覆写等）
        super().__post_init__()

        # --- [rc1 runtime 修复 R1] sim.physics → PhysxAutoCfg -----------
        self.sim.physics = PhysxAutoCfg(
            isaacsim_physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
        # newton actuator 路由关回缺省（rough 段置 True；Implicit 走 physx）
        self.sim.use_newton_actuators = False

        # --- [rc1 runtime 修复 R2] events.base_com 预设解包 -------------
        self.events.base_com = self.events.base_com.default

        # --- [rc1 runtime 修复 R3 + 资产替换] 包内 go2 profile -----------
        _profile = make_go2_track_profile(prim_path="{ENV_REGEX_NS}/Robot")
        _profile.articulation_cfg.soft_joint_pos_limit_factor = 0.9
        self.scene.robot = _profile.articulation_cfg

        # --- 官方口径覆写（上游对齐面） ----------------------------------
        # ① joint_vel scale → 1.0（官方 raw rad/s）
        self.observations.policy.joint_vel.scale = 1.0
        # ② history_length → None（单帧直出）
        self.observations.policy.history_length = None
        # ③ 动作 scale → 官方均匀 0.5（rough 段 0.25 → 基座值）
        self.actions.joint_pos.scale = 0.5
        # ④ heading_command → False（直接 wz 跟踪，匹配 kick 部署语义）
        self.commands.base_velocity.heading_command = False
        # debug marker 关（headless goal marker 材质死旋防护；训练无关）
        self.commands.base_velocity.debug_vis = False


__all__ = [
    "VelocityOfficialGo2FlatEnvCfg",
]
