# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.velocity_official_cfg
#
# 官方 velocity-flat 口径训练面：
#   - go2：任务 HK-Velocity-Flat-Unitree-Go2-v0。基类 = site-packages
#     isaaclab_tasks 的 ``UnitreeGo2FlatEnvCfg``（core/velocity/config/go2，
#     官方 Isaac-Velocity-Flat-Unitree-Go2-v0 同一 cfg 类），``__post_init__``
#     先修 rc1 runtime（见下 R1-R3），再覆写回官方 obs/动作/命令口径；
#   - cyberdog2：任务 HK-Velocity-Flat-CyberDog2-v0。基类 = 上游通用
#     ``LocomotionVelocityRoughEnvCfg``（多本体基座），同套 rc1 修复 +
#     官方口径覆写，body names/命令 ranges 以 cyberdog2 本体为准
#     （文件尾注块）。
#
# 官方口径覆写四件（两版同一套）：
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
# 上游通用多本体基座（cyberdog2 链用；不含 go2 专属资产/body names）
from isaaclab_tasks.core.velocity.velocity_env_cfg import (  # noqa: E501
    LocomotionVelocityRoughEnvCfg,
)
import isaaclab_tasks.core.velocity.mdp as velocity_mdp  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402

from hierarchical_kick.robots import make_robot_profile


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

        # --- [rc1 runtime 修复 R3 + 资产替换] 包内 profile（注册表分派） ----
        _profile = make_robot_profile("go2", prim_path="{ENV_REGEX_NS}/Robot")
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


# ---------------------------------------------------------------------------
# Cyberdog2 官方口径面（任务 HK-Velocity-Flat-CyberDog2-v0）
#
# 基类 = 上游通用 ``LocomotionVelocityRoughEnvCfg``（rc1 多本体基座，不含
# 任何 go2 专属资产/body names）——**不继承 go2-flat 链**（其 rewards
# body_names/资产全 go2 专属）。cyberdog2 专属化复刻 go2 rough→flat 的
# 覆写谱系（同一套 rc1 三修复 + 官方口径四覆写 + go2-flat 最终奖励数值），
# body names 以 cyberdog2 URDF/xiaomi 成功快照为准：
#   - base body = ``body``（head 为独立 fixed link）；足端 = ``.*_foot``；
#   - undesired_contacts = None（go2-flat 同款：rough 段置 None）；
#   - contact_forces prim_path → ``Robot/.*``（URDF 转换为嵌套 prim 树，
#     基座 ``[^/]*` 仅匹配直接子级＝root link——xiaomi 成功快照同款 .*）；
#   - 命令 ranges = xiaomi 成功快照本体参数（x ±1.6 / y ±0.55 / yaw ±2.5
#     m/s、m/s、rad/s），命令器结构/重采样/standing 份额保持官方基座原值。
# ---------------------------------------------------------------------------


@configclass
class VelocityOfficialCyberDog2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Cyberdog2 官方口径 velocity-flat 面（48 维单帧 obs + 均匀 0.5 动作）。

    产物语义与 go2 版一致：官方口径 locomotion checkpoint
    （core/low_level_actor 官方口径装载体可消费的 MLP[128,128,128] ELU
    低层，48 维单帧 policy obs）。
    """

    def __post_init__(self):
        # post init of parent（通用基座：decimation/dt/传感器 update period/
        # terrain 课程 flag 接线）
        super().__post_init__()

        # --- [rc1 runtime 修复 R1] sim.physics → PhysxAutoCfg -----------
        self.sim.physics = PhysxAutoCfg(
            isaacsim_physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
        # newton actuator 路由关回缺省（DCMotor Implicit 走 physx 直驱）
        self.sim.use_newton_actuators = False

        # --- [rc1 runtime 修复 R2] events.base_com 预设解包 -------------
        self.events.base_com = self.events.base_com.default

        # --- [rc1 runtime 修复 R3 + 资产替换] 包内 cyberdog2 profile ----
        _profile = make_robot_profile("cyberdog2",
                                      prim_path="{ENV_REGEX_NS}/Robot")
        _profile.articulation_cfg.soft_joint_pos_limit_factor = 0.9
        self.scene.robot = _profile.articulation_cfg

        # --- flat 化（go2-flat 覆写同款） -------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # contact sensor 全 body 匹配（URDF 嵌套 prim 树；xiaomi 快照同款）
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # --- body names（基座 go2/anymal 语义名 → cyberdog2 名） ---------
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_com.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"]\
            .body_names = "body"
        self.terminations.base_contact.params["sensor_cfg"]\
            .body_names = "body"
        self.rewards.feet_air_time.params["sensor_cfg"]\
            .body_names = ".*_foot"
        # go2-flat 同款（rough 段置 None；base 的 .*THIGH 名不属 cyberdog2）
        self.rewards.undesired_contacts = None

        # --- 奖励谱系（go2 rough→flat 最终数值复刻） ---------------------
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.flat_orientation_l2.weight = -2.5

        # --- 命令 ranges（xiaomi 成功快照本体参数；结构=官方基座） -------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.6, 1.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.55, 0.55)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)

        # --- 官方口径覆写（与 go2 版同一套四件） -------------------------
        # ① joint_vel scale → 1.0（官方 raw rad/s）
        self.observations.policy.joint_vel.scale = 1.0
        # ② history_length → None（单帧直出；rc1 通用基座本就无 history
        #    字段——getattr 缺省 None 已是装配自检锚，此处仅在场即覆写）
        if hasattr(self.observations.policy, "history_length"):
            self.observations.policy.history_length = None
        # ③ 动作 scale → 官方均匀 0.5（通用基座值；显式锚）
        self.actions.joint_pos.scale = 0.5
        # ④ heading_command → False（直接 wz 跟踪，匹配 kick 部署语义）
        self.commands.base_velocity.heading_command = False
        # debug marker 关（headless goal marker 材质死旋防护；训练无关）
        self.commands.base_velocity.debug_vis = False

        # --- 抗塌陷塑形（round-2；奖励/reset = 非契约面，obs/动作/命令
        #     契约不动。round-1 诊断：官方奖励谱系下 cyberdog2（kp25 弱
        #     执行器）陷入"死亡寻求+高频关节抖动"平衡（ep-len 12、reward
        #     -1.7、探针 qd ±25 rad/s 原地滑移）——三件对策：生存正信号/
        #     抖动加罚/温和 reset（reset 面取 xiaomi 成功快照口径）） ----
        self.rewards.survival = RewTerm(
            func=velocity_mdp.is_alive, weight=0.25)
        self.rewards.action_rate_l2.weight = -0.10
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)}


def VelocityOfficialFlatEnvCfg(robot: str = "go2"):
    """注册表工厂：robot → 官方口径 velocity-flat cfg 实例（Unknown fail-fast）。

    go2 → VelocityOfficialGo2FlatEnvCfg（行为 = 历史零变化路径）；
    cyberdog2 → VelocityOfficialCyberDog2FlatEnvCfg。
    """
    _CLS_BY_ROBOT = {
        "go2": VelocityOfficialGo2FlatEnvCfg,
        "cyberdog2": VelocityOfficialCyberDog2FlatEnvCfg,
    }
    if robot not in _CLS_BY_ROBOT:
        raise ValueError(
            f"[hkick][velocity] Unknown robot: {robot!r}"
            f"（registered: {sorted(_CLS_BY_ROBOT)}）")
    return _CLS_BY_ROBOT[robot]()


__all__ = [
    "VelocityOfficialGo2FlatEnvCfg",
    "VelocityOfficialCyberDog2FlatEnvCfg",
]