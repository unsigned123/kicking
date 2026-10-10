# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# hierarchical_kick.sim.ball_env_cfg — U-Kick 球 env 配置（自建，非 vendor）：
# 单 go2 + 单球 + 目标球门（解析判线，无门柱物理块）。训练面
# Go2KickFlatEnvCfg；Go2KickFlatEnvCfgOracle 为 oracle/冒烟面（冻结 velocity
# ckpt 直驱，仅对照/冒烟用，禁止用于策略训练）。
#
# 继承链（零改动复用；本文件只做 kick 面追加/替换）：
#   LocomotionVelocityRoughEnvCfg（velocity_env_cfg.py，vendor 只读）
#     → UnitreeGo2FlatEnvCfg（flat_env_cfg.py，vendor+upper 只读）
#     → Go2KickFlatEnvCfg（本文件）
#
# 球物理/传感器/reset 移植自上游 soccer_absolute 实现（只抄参数/公式/语义，
# 零 import）：球物理全套 ← env/scene.py:215-247 逐值（radius 0.11 / mass
# 0.43 / μs 1.0 μd 0.95 / restitution 0.25 / lin+ang damping 2.5「松脚即停」
# 高阻尼）；球侧接触传感器 ← env/scene.py:250-256（filter 收敛为 base + 4
# 足端——踢击检测所需的保守 filter 面，base 兜底躯干触球）；球 reset 写入 ←
# env/reset.py:257-280 语义（default_root_state + env_origins → write → reset）。
#
# KICK 适配（对 flat 链的全部 override，逐处标注）：
#   1) scene：+ 球 RigidObjectCfg（scene.ball）+ 球侧 ContactSensor
#      （scene.ball_contact_forces）；terrain/光照/height_scanner=None 沿用继承链。
#   2) events：+ reset_ball（func 单源 = mdp.reset_ball_root_state——随机化/
#      课程采样逻辑移入 ball_mdp 离线可测面）；reset_base 的 pose_range 钳零
#      —— 狗固定 spawn 零采样语义（保留 velocity_range=(0,0) 与 stand 姿态
#      的现款零速度语义）；狗 spawn 常量面 = mdp.DOG_SPAWN_XY=(0,0)（原点、
#      进攻轴 +y）+ yaw=π/2（init_state 单源，rc1 default⊗delta 语义下
#      reset 后稳定存活）。
#   3) observations.policy → KickPolicyCfg（55 维 = 42-prefix 契约逐参不动
#      ⊕ base_lin_vel 3 ⊕ ball_rel_b 3 ⊕ goal_rel_b 4 ⊕ push_geometry_b 3）。
#   4) commands.base_velocity = None（任务无 cmd 面）。
#   5) rewards task-only：奖励规则接口公式面对齐 soccer_absolute——权重按
#      go2 平台适配不照搬，权重单源 = KickRewardWeightsCfg（量纲理由见其
#      docstring）；velocity track 系 + 全部 regularization 项仍显式 = None。
#      挂载 8 项规则 + 自建第 9 项 kick_event（缺省权重 0.0 = 零行为影响）。
#   6) terminations：root_height(0.15) → robot_fallen（双条件 grav>−0.5 ∨
#      body_z<0.16——高度支 0.16 严格覆盖 0.15，见挂载注释）+
#      robot_out_of_field（工作区框，默认启用）+ goal_scored / ball_out
#      （既有）+ time_out / base_contact（继承保留）；wall_stuck 接口在位
#      不挂载（无墙）。
#   7) episode_length_s = 10.0（500 步 @50Hz）。
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from hierarchical_kick.core.contracts import (
    NOISE_BASE_ANG_VEL,
    NOISE_BASE_LIN_VEL,
    NOISE_JOINT_POS,
    NOISE_JOINT_VEL,
    NOISE_PROJ_GRAVITY,
    SCALE_BASE_ANG_VEL,
    SCALE_JOINT_VEL,
)
from hierarchical_kick.core.scale_vec import action_scale_table_for
from hierarchical_kick.robots import CYBERDOG2_FOOT_BODIES, make_cyberdog2_track_profile
from hierarchical_kick.robots.go2 import GO2_FOOT_BODIES
from hierarchical_kick.sim import mdp
from hierarchical_kick.sim.flat_env_cfg import UnitreeGo2FlatEnvCfg, UpperPolicyCfg
from hierarchical_kick.sim.mdp.frozen_loco_action import (
    FrozenLocoJointPositionActionCfg,
    low_level_last_action_obs,
)

##
# spawn 常量面：狗 spawn 原点 (0,0)、yaw=π/2 固定（常量单源 = mdp.DOG_SPAWN_XY）；
# 球出生随机化 + 距离轴课程（dist ~ U[d_min, d_max(t)]，见 KickRandomizationCfg /
# mdp.ball_spawn_* 单源）。本文件仅存 enabled=False 固定 spawn 逃逸口的
# init_state 常量（门线前 0.2m）。
##


DOG_SPAWN_YAW_QUAT_XYZW = (0.0, 0.0, math.sin(0.25 * math.pi), math.cos(0.25 * math.pi))
"""狗 spawn 朝向四元数（**xyzw**，isaaclab 3.0 通道序）：yaw=π/2 面朝球门
（进攻轴 +y）。reset 存活性：rc1 ``reset_root_state_uniform`` = default ⊗
delta、本链 reset_base pose_range 钳零（delta=identity）→ 本 quat 每次
reset 后逐位回到。"""

BALL_SPAWN_XY = (0.0, mdp.GOAL_Y - 0.2)
"""球固定 spawn（env 系 xy；kick_randomization.enabled=False 逃逸口）：
门线前 0.2m（GOAL_Y=3.0 → y=2.8）。缺省 enabled=True 走随机化 + 课程，
本常量仅供 A/B 与调试面。"""


##
# 随机化 + 距离轴课程 cfg（纯 python 数据类，不依赖 isaaclab——采样/调度
# 逻辑单源在 mdp.ball_spawn_* / mdp.reset_ball_root_state）
##


@dataclass
class KickRewardWeightsCfg:
    """U-Kick 奖励权重单源（公式面按上游 soccer_absolute 规则接口对齐 +
    go2 平台适配，**权重不照搬**；全仓库唯一权重定义点）。

    量纲理由（go2 站高 0.227 / 门距 3m / 500 步 @50Hz / 无全局 reward_scale）：

    | 项                      | 默认 | 量纲理由 |
    |-------------------------|---------|---------|
    | goal_scored             | +10.0   | 稀疏主奖励（进球即终止单次） |
    | robot_fallen            | −5.0    | 一次性 mask；≈半个 goal 的痛感 |
    | survival                | +0.25   | β1；fall penalty 在场后仍提供"活下去的底薪"（500 步全存活 = 125 ≈ 12.5 个 goal——底薪大于痛感，但摔即失全额生存流，结构性反摔） |
    | approach_ball_velocity  | +0.15   | 投影速度量纲 m/s × coeff 1.0、cap max_speed 0.5 → 单步 ≤0.075 |
    | ball_distance           | +0.05   | exp 核 ∈(0,1] 无量纲、3m 域全程底薪引导（≈survival 的 20%） |
    | behind_ball             | +0.05   | exp 核 × scale 0.35 → 单步 ≤0.0175，弱位形先验不与主引导抢幅值 |
    | ball_progress           | +0.10   | Δy 差分 × scale 5：0.5m/s 球速@50Hz → 单步 0.005、全程 3m 推进 ≈1.5（goal 的 15%） |
    | ball_out_of_bounds      | −2.0    | 步 mask 罚；≈8 步正项流水等价（偏严防胡踢） |

    调参零摩擦（单点改法，任选其一）：
      A) 代码默认：改本数据类字段默认值（全仓库唯一权重定义点）；
      B) 实例覆写：``cfg = Go2KickFlatEnvCfg();
         cfg.kick_reward_weights.ball_progress = 0.0;
         cfg.sync_kick_reward_weights()``（一行同步到已挂 RewTerm.weight，
         幂等）；或直接 ``cfg.rewards.ball_progress.weight = 0.0``（挂载面即
         消费面——RewardManager 在 env 构造期才读权重）；
      C) yaml 桥（预留）：训练 yaml 加 ``reward_weights: {字段: 值}`` 段后由
         run_upper 构造期填 kick_reward_weights 再 sync（字段面已冻结，接线
         属调参期）。
    kernel 形参（max_dist/temperature/coeff 等）= 上游语义默认，见
    ball_mdp 各 wrapper 签名，可经 ``RewTerm(params={...})`` 覆写。
    """

    goal_scored: float = 10.0
    robot_fallen: float = -5.0
    survival: float = 0.25
    approach_ball_velocity: float = 0.15
    ball_distance: float = 0.05
    # ball_distance exp 核温度（核宽）入单源：缺省 2.0 = 上游语义默认
    # （基线逐位不变）；可经 --kick_rw_overrides 调小陡化核
    ball_distance_temperature: float = 2.0
    behind_ball: float = 0.05
    ball_progress: float = 0.10
    ball_out_of_bounds: float = -2.0
    # 踢击事件回报（自建过程引导项）：接触力范数 >0.1N 判踢 + 0.5s/25 步
    # 冷却，1.0 mask × weight = 事件回报幅值。
    # **缺省 0.0 = 零行为影响**（红线纪律：基线面默认无此项贡献；经
    # --kick_rw_overrides '{"kick_event": 0.5, ...}' 启用）
    kick_event: float = 0.0


@dataclass
class KickRandomizationCfg:
    """kick 面 spawn 随机化 + 距离轴课程。

    enabled=True（**缺省**）：reset_ball 事件（mdp.
    reset_ball_root_state 单源）把球出生位整体覆写为狗 spawn 前方采样位：
      - dist ~ U[d_min, d_max(t)]：距离轴随机化本身就是课程——d_max(t) =
        0.5 + (d_max_final−0.5)×clamp(common_step_counter/warmup_steps, 0, 1)
        （mdp.ball_spawn_d_max 单源；warmup_steps = warmup_iters ×
        steps_per_iter）。steps_per_iter 口径 = 24 步/it
        （common_step_counter 每 env.step +1、rollout num_steps_per_env=24）
        → warmup_iters=500 处恰满域（口径见 manager_based_rl_env.py:249）；
      - 横向 x ~ U(±ball_xy_range[0])（缺省 ±0.3；y 通道
        不消费——y 轴由 dist 承载）；
      - y clamp ≤ GOAL_Y − 0.2（mdp.CURRICULUM_Y_MARGIN，防 reset 即在
        门内假 goal）。
    curriculum_enabled=False：跳过调度，dist ~ U[d_min, d_max_final] 恒满域。
    ball_dist_override：eval-only 球距覆写（eval_kick.py --ball_dist 单一口），
    非 None 时 dist 轴取常量、跳过采样（横向仍采样）——球距响应扫描用。
    enabled=False 且无 override：完全固定 spawn BALL_SPAWN_XY（init_state
    单源）。狗恒固定 spawn mdp.DOG_SPAWN_XY +
    yaw=π/2（不随机；init_state 单源 + reset_base pose_range 钳零）。
    dog_yaw_range：预留字段（接线后再消费；当前实现不驱动狗位姿）。
    """

    enabled: bool = True
    ball_xy_range: tuple[float, float] = (0.3, 0.0)
    dog_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    # 距离轴课程（规格公式字段化）
    curriculum_enabled: bool = True
    warmup_iters: int = 500
    steps_per_iter: float = 24.0
    d_min: float = 0.1
    d_max_final: float = 2.7
    # eval-only 球距覆写（None = 不覆写；eval_kick.py --ball_dist）
    ball_dist_override: float | None = None


##
# reset 事件（isaaclab 面）：reset_ball func 单源 = mdp.reset_ball_root_state
# （随机化/课程采样逻辑在 ball_mdp——离线可测；挂载见
# Go2KickFlatEnvCfg.__post_init__ 的 events.reset_ball）
##


##
# obs 组（55 维 kick 契约）
##


@configclass
class KickPolicyCfg(ObsGroup):
    """kick 面 policy obs（55 维单帧 = KICK_OBS_DIM）。

    前 5 项与 UpperPolicyCfg（flat_env_cfg.py 42-prefix 契约）**逐参相同**
    （噪声/scale 原样，joint_pos/joint_vel/base_ang_vel/projected_gravity/
    actions = 42 维 decoder proprio 契约，绝不再加第二噪声源）；尾部追加
    base_lin_vel 3 ⊕ ball_rel_b 3 ⊕ goal_rel_b 4 ⊕ push_geometry_b 3。
    单帧无 history（与 upper 现栈一致，不引入 obs history buffer）。
    """

    # -- decoder proprio 42（首 term；逐参 = UpperPolicyCfg，契约切片不动） --
    # [表#1] joint_pos (0:12)：U±0.01，scale 1.0
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-NOISE_JOINT_POS, n_max=NOISE_JOINT_POS))
    # [表#2] joint_vel (12:24)：U±0.5 × scale 0.05（decoder 契约；官方项不照抄）
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-NOISE_JOINT_VEL, n_max=NOISE_JOINT_VEL),
        scale=SCALE_JOINT_VEL,
    )
    # [表#3] base_ang_vel (24:27)：U±0.2 × scale 0.2
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-NOISE_BASE_ANG_VEL, n_max=NOISE_BASE_ANG_VEL),
        scale=SCALE_BASE_ANG_VEL,
    )
    # [表#4] projected_gravity (27:30)：U±0.05，scale 1.0
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-NOISE_PROJ_GRAVITY, n_max=NOISE_PROJ_GRAVITY),
    )
    # [表#5] last_action (30:42)：无噪声
    actions = ObsTerm(func=mdp.last_action)
    # -- 策略面 + kick 几何（42:55） --
    # [表#6] base_lin_vel (42:45)：U±0.1（与 UpperPolicyCfg 同参）
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-NOISE_BASE_LIN_VEL, n_max=NOISE_BASE_LIN_VEL))
    # [表#8] ball_rel_b (45:48)：body 系 xy/3 + 距离/3（上游 observations.py:73-79）
    ball_rel_b = ObsTerm(func=mdp.ball_rel_b)
    # [表#9] goal_rel_b (48:52)：body 系 xy/3 + sin + cos（上游 observations.py:81-88）
    goal_rel_b = ObsTerm(func=mdp.goal_rel_b)
    # [表#10] push_geometry_b (52:55)：cos_behind + fwd-vs-btog sin/cos
    #         （上游 observations.py:205-235）
    push_geometry_b = ObsTerm(func=mdp.push_geometry_b)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


##
# 训练面
##


@configclass
class Go2KickFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """U-Kick 球 env 训练面。

    官方 flat 谱系 + KICK 适配 1–7（文件头表）；奖励规则接口公式面对齐
    soccer_absolute、权重按 go2 平台适配，单源 KickRewardWeightsCfg。
    obs/终止/球物理全部继承链逐位同。
    """

    def __post_init__(self):
        # post init of parent（官方 flat 谱系全链：plane terrain、逐关节 scale、
        # root_height 终止、survival 项等）
        super().__post_init__()

        # [1] 球场景（scene.py:215-247 逐值；init_state pos = 固定 spawn
        #      BALL_SPAWN_XY=(0, GOAL_Y−0.2, 0.11)——enabled=True 缺省走随机化
        #      + 课程时此 init_state 仅作 default_root_state 底板（xy 被采样
        #      覆写，z=0.11 沿用）
        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.11,  # 球半径（scene.py:218，对齐 cyberdog2_rl）
                # visual_material 同上游 PreviewSurfaceCfg（白色）——
                # **headless 不挂**：缺 libXt/libGLU 的会话 preview 材质
                # spawn 死旋（录像阶段在带 --enable_cameras 的会话内再启用）
                # 物理材质（scene.py:223-227）
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.95,
                    restitution=0.25,
                ),
                # 碰撞属性（scene.py:229-231）
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                # 刚体属性（scene.py:233-237；高阻尼「松脚即停」）
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=2.5,
                    angular_damping=2.5,
                ),
                # 质量属性（scene.py:238-240）
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.43,
                ),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                # 球固定 spawn 逃逸口常量面（BALL_SPAWN_XY 门线前 0.2m）——
                # 缺省 enabled=True 时此值仅作 default_root_state 底板。
                # ⚠️ 贴脚档注意：dist 下限 0.1m 处球面（r0.11 表面延伸进
                # 机体）与狗前腿初始穿透、PhysX 解算弹飞——随机化下
                # dist ∈ [0.1, ·] 仍含贴脚档，若首 iter 出现 goal 假信号/
                # 球飞优先归因于此（弹飞朝 +y = 瞬间过门线假 goal；
                # ball_progress 首步重播种对 spawn 跳变免疫）。
                pos=(BALL_SPAWN_XY[0], BALL_SPAWN_XY[1], 0.11),
                rot=(0.0, 0.0, 0.0, 1.0),  # 无旋转（xyzw 单位元；isaaclab 3.0 通道序）
            ),
        )
        # [1] 球侧接触传感器（scene.py:250-256 语义；filter 表达式适配
        #      go2 prim 布局：上游 4 机 17 体列表 → 本包单狗 base + 4 足端
        #      的保守 filter 面——足端主踢击检测 + base 兜底躯干触球；
        #      kick_count 指标消费本传感器）
        self.scene.ball_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            update_period=0.0,
            history_length=20,
            # filter 显式枚举（单源 GO2_FOOT_BODIES；**不用 .* 通配**——
            # env_*/Robot/.*_foot 在多 env 下跨 env 匹配翻倍触发 tensor API
            # expected/found 警告且力矩阵形状漂移；显式名逐 env 恰 5 prim）
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/base",
                *[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in GO2_FOOT_BODIES],
            ],
        )

        # [2] 随机化 + 距离轴课程挂载（enabled=True 缺省——球出生狗前方
        #      dist ~ U[d_min, d_max(t)] + 横向 U(±0.3)，课程调度面 =
        #      mdp.ball_spawn_d_max 单源）
        self.kick_randomization = KickRandomizationCfg()

        #      狗 spawn 常量面：原点 (0,0)、yaw=π/2 面朝球门（常量单源 =
        #      mdp.DOG_SPAWN_XY）。实现走 init_state 单源（spawn z 沿用
        #      profile 现款，不硬编码）：rc1 reset_root_state_uniform =
        #      default ⊗ delta、本链 pose_range 钳零（delta=identity）→
        #      每次 reset 后位姿逐位回到本值，yaw 稳定存活；下方 reset_base
        #      零采样语义保持（狗固定 spawn）。
        _dog_init = self.scene.robot.init_state
        _dog_init.pos = (mdp.DOG_SPAWN_XY[0], mdp.DOG_SPAWN_XY[1],
                         float(_dog_init.pos[2]))
        _dog_init.rot = DOG_SPAWN_YAW_QUAT_XYZW

        # [2] 狗固定 spawn 零采样（velocity_range 已由 flat 链钳零 = 现款
        #      零速度语义；stand 姿态经 joint reset 继承）
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # [2] 球 reset 事件（写入 root state；func 单源 = mdp.
        #      reset_ball_root_state——随机化/课程采样逻辑在 ball_mdp，
        #      离线可测；ball_progress_dense 缓存由 wrapper"每集首步重播种"
        #      对 reset 跳变免疫，随机 spawn 下正是其发挥作用的面）
        self.events.reset_ball = EventTerm(
            func=mdp.reset_ball_root_state,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("ball")},
        )

        # [3] obs 组替换（55 维 kick 契约；42-prefix 逐参不动）
        self.observations.policy = KickPolicyCfg()

        # [4] 无命令面（任务无 cmd；KickPolicyCfg 无 velocity_commands 项，
        #      悬空引用不存在）
        self.commands.base_velocity = None

        # [5] 奖励 task-only——velocity track 系 +
        #      regularization 全部显式置 None（真实字段名逐项在位）：
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.lin_vel_z_l2 = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.action_rate_l2 = None
        self.rewards.undesired_contacts = None  # rough 链已 None；显式幂等在位
        self.rewards.dof_pos_limits = None
        # 权重单源（KickRewardWeightsCfg；量纲理由表见类 docstring）
        self.kick_reward_weights = KickRewardWeightsCfg()
        w = self.kick_reward_weights
        # —— 规则接口 8 项（公式面 provenance 见 ball_mdp 头）——
        # 进球稀疏主奖励（幅值由 weight 承载；goal_scored 即终止，无重复触发）
        self.rewards.goal_scored = RewTerm(func=mdp.goal_scored_reward, weight=w.goal_scored)
        # 摔倒罚：状态直检 1.0 mask（奖励先于终止计算，不依赖 termination
        # manager；上游 rewards.py:37 单狗化）；阈值与 robot_fallen 终止同源
        self.rewards.robot_fallen = RewTerm(
            func=mdp.robot_fallen_penalty, weight=w.robot_fallen,
            params={"grav_threshold": mdp.FALLEN_GRAV_THRESHOLD,
                    "body_z_threshold": mdp.FALLEN_BODY_Z_THRESHOLD})
        # survival（恒 +1.0 × β1）显式挂载（继承 flat 链 value 语义不变，
        # 权重入单源；fall penalty 在场后仍提供活下去的底薪）
        self.rewards.survival = RewTerm(func=mdp.alive_reward, weight=w.survival)
        # 稠密引导族（默认启用；kernel 形参 = ball_mdp 签名默认，
        # 可经 RewTerm(params={...}) 覆写）
        self.rewards.approach_ball = RewTerm(func=mdp.approach_ball_velocity_reward, weight=w.approach_ball_velocity)
        self.rewards.ball_distance = RewTerm(func=mdp.ball_distance_reward, weight=w.ball_distance, params={"temperature": w.ball_distance_temperature})
        self.rewards.behind_ball = RewTerm(func=mdp.behind_ball_reward, weight=w.behind_ball)
        # 球向门 Δy 进攻方向差分（上游 ball_progress_dense；替换旧
        # ball_progress_shaping ε 预注册档——其 prev-cur 距离缓存及 reset
        # 钩子已整体移除，ball_progress 自带缓存对 reset 跳变免疫）
        self.rewards.ball_progress = RewTerm(func=mdp.ball_progress_dense, weight=w.ball_progress)
        # 踢击事件回报（自建项；检测面 = kick_metrics
        # 单源。缺省 weight 0.0 → 挂载但零贡献——基线行为面逐位不变）
        self.rewards.kick_event = RewTerm(func=mdp.kick_event_reward, weight=w.kick_event)
        # 球出界步 mask 罚（复用 ball_out 半径判据）
        self.rewards.ball_out_of_bounds = RewTerm(func=mdp.ball_out_of_bounds_penalty, weight=w.ball_out_of_bounds)

        # [6] 终止：
        #  - robot_fallen 替换继承 root_height(0.15)：双条件
        #    （proj_grav_z > −0.5 —— 站立 ≈−1，倾斜超 ~60°；或 base 高 <
        #    0.16m —— go2 站高 0.227，余量 0.067m ≈ 30% 站高，蹲伏步态波谷
        #    不误杀、瘫倒/侧翻必触发）。高度支 0.16 **严格覆盖** root_height
        #    的 0.15（body_z<0.15 ⇒ body_z<0.16，fallen 必先触发），重力支
        #    补"倾斜未低"的翻倒面 → root_height 显式置 None 防双挂载语义漂移。
        self.terminations.root_height = None
        self.terminations.robot_fallen = DoneTerm(
            func=mdp.robot_fallen,
            params={"grav_threshold": mdp.FALLEN_GRAV_THRESHOLD,
                    "body_z_threshold": mdp.FALLEN_BODY_Z_THRESHOLD})
        #  - robot_out_of_field：机器人工作区框（env 系相对逐 env origin；
        #    默认启用——单狗无墙球会滚远，球逃逸由 ball_out 处置，机器人本身
        #    不该为追球跑出工作区；上游场界矩形判据 → 本包任务域非对称框）
        self.terminations.robot_out_of_field = DoneTerm(
            func=mdp.robot_out_of_field,
            params={"x_min": mdp.ROBOT_FIELD_X_MIN, "x_max": mdp.ROBOT_FIELD_X_MAX,
                    "y_min": mdp.ROBOT_FIELD_Y_MIN, "y_max": mdp.ROBOT_FIELD_Y_MAX})
        #  - wall_stuck 接口在位**不挂载**（无墙；mdp.wall_stuck 备用，
        #    上游 dist+speed 双条件，引入场地墙段后再挂）
        #  - goal_scored / ball_out（既有）+ time_out / base_contact（继承；
        #    base_contact 与 robot_fallen 互补：贴地躯干触地即终止，兜底
        #    双条件判据覆盖不到的接触形态，如腹部擦地倾斜角未超阈值瞬间）
        self.terminations.goal_scored = DoneTerm(func=mdp.goal_scored)
        self.terminations.ball_out = DoneTerm(func=mdp.ball_out)

        # 权重同步（默认值刷一遍；实例级调参 = 改 kick_reward_weights 字段
        # 后再调本方法，或直接改 rewards.<term>.weight——幂等）
        self.sync_kick_reward_weights()

        # [7] episode 时长 10s（500 步 @50Hz）
        self.episode_length_s = 10.0

    def sync_kick_reward_weights(self) -> None:
        """把 kick_reward_weights 刷到已挂 reward term 权重（调参单行口）。

        用法（实例级覆写，env 构造前任意时刻）：
            cfg = Go2KickFlatEnvCfg()
            cfg.kick_reward_weights.ball_progress = 0.0
            cfg.sync_kick_reward_weights()
        幂等；__post_init__ 末尾自动调用一次（默认值同步）。
        """
        w = self.kick_reward_weights
        self.rewards.goal_scored.weight = w.goal_scored
        self.rewards.robot_fallen.weight = w.robot_fallen
        self.rewards.survival.weight = w.survival
        self.rewards.approach_ball.weight = w.approach_ball_velocity
        self.rewards.ball_distance.weight = w.ball_distance
        self.rewards.ball_distance.params["temperature"] = w.ball_distance_temperature
        self.rewards.behind_ball.weight = w.behind_ball
        self.rewards.ball_progress.weight = w.ball_progress
        self.rewards.kick_event.weight = w.kick_event
        self.rewards.ball_out_of_bounds.weight = w.ball_out_of_bounds


##
# oracle / 冒烟面（冻结 velocity ckpt 直驱；验收门用）
##


@configclass
class Go2KickFlatEnvCfgOracle(Go2KickFlatEnvCfg):
    """oracle 面：48 维 velocity 契约 + 命令管理器在位（teacher 直写 cmd）。

    ⚠️ 仅 oracle/冒烟评测用，禁止用于策略训练。冻结 velocity policy 的
    obs 契约 = proprio 42 ⊕ base_lin_vel 3 ⊕ cmd 3（48 维），命令管理器每步
    由 scripted teacher（几何状态机）直写 command buffer；球场景/终止/奖励与
    训练面（Go2KickFlatEnvCfg）逐位同 —— oracle 链：teacher 几何 → 速度命令
    [vx,vy,ωz] → 冻结 ckpt → 关节动作（验收门：10 episode ≥3 进球、零摔）。
    """

    def __post_init__(self):
        # post init of parent（kick 面全链：球场景/终止/奖励/obs 55 维等）
        super().__post_init__()

        # 恢复命令管理器（UniformVelocityCommand 语义；heading_command=False，
        # ranges 任意——teacher 每步直写 command buffer，不依赖命令采样）
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
            ),
        )

        # 恢复 48 维 velocity 契约 obs（UpperPolicyCfg；ckpt 免适配直驱）
        self.observations.policy = UpperPolicyCfg()


##
# cmd「真分层」面（冻结低层 ckpt + 3 维速度命令高层）
##


@configclass
class KickPolicyCfgCmd(KickPolicyCfg):
    """cmd 高层 obs 组（55 维契约保持；仅 actions 项换源）。

    唯一 diff（对照 :class:`KickPolicyCfg`）：``actions`` 项从
    ``mdp.last_action``（= action_manager.action，cmd 面下是 3 维命令）换成
    :func:`low_level_last_action_obs`（冻结低层上一步 12 维 pre-scale 输出）
    ——42-prefix 通道序/维数与训练面逐位同构（55 = 42+3+3+4+3），通道语义
    同为「关节级 last action」。其余项（joint_pos/joint_vel/base_ang_vel/
    projected_gravity/base_lin_vel/ball_rel_b/goal_rel_b/push_geometry_b）
    **零改动继承**（噪声/缩放逐参同源）；奖励/终止不在本类——继承训练面。
    """

    # [表#5] last_action (30:42)：无噪声；cmd 面语义 = 冻结低层上一步输出
    #        （3 维命令动作不进本通道——见 frozen_loco_action.
    #        low_level_last_action_obs docstring）
    actions = ObsTerm(func=low_level_last_action_obs)

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2KickFlatEnvCfgCommand(Go2KickFlatEnvCfg):
    """cmd「真分层」面（任务 Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0）。

    与训练面（Go2KickFlatEnvCfg）的**全部差异只在动作接口**：
      - ``actions.joint_pos`` → :class:`FrozenLocoJointPositionActionCfg`
        （3 维 [vx,vy,wz] 命令 → 写命令 buffer → 冻结低层 → 12
        关节位置目标；低层约定缺省 official = 官方 velocity-flat 口径
        （weights/velocity_flat_official/model_walk.pt，均匀 0.5 动作
        scale），--low_level_convention legacy 切旧契约
        model_2999.pt + 逐关节 scale_vec；ckpt 路径 launch 期填充，
        构造期存在性 fail-fast）；
      - ``commands.base_velocity`` 恢复（照 :class:`Go2KickFlatEnvCfgOracle`
        的恢复写法——命令面在位，ranges 任意：每步被高层动作覆写，采样
        值不消费；resample 竞争分析见 frozen_loco_action 模块头）；
      - ``observations.policy`` → :class:`KickPolicyCfgCmd`（仅 actions 项
        换低层 last_action，55 维契约保持）。
    obs 尾部（ball/goal/push）/奖励（8 项）/终止/球
    物理/PPO 面全部继承训练面**零改动**。
    """

    def __post_init__(self):
        # post init of parent（kick 训练面全链：球场景/奖励/终止/obs 55 维）
        super().__post_init__()

        # 恢复命令管理器（UniformVelocityCommand 语义；写法照 oracle 面——
        # heading_command=False、ranges 任意：命令 buffer 每步被冻结动作
        # term 覆写为高层输出，采样值/课程/站桩分支均不消费）
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
            ),
        )

        # 动作项替换：3 维命令 → 冻结低层 → 关节目标。低层约定缺省 official
        # （官方 velocity-flat 口径；--low_level_convention 可切 legacy）：
        #   - official：均匀 0.5 动作 scale（scale dict 不消费，占位 legacy 面）
        #     + ckpt 缺省 weights/velocity_flat_official/model_walk.pt；
        #   - legacy：逐关节 scale_vec（单源表）+ legacy ckpt model_2999.pt。
        # ckpt 路径留空 = 按约定缺省（run_upper launch 期经 --low_level_ckpt
        # 填绝对路径，构造期存在性 fail-fast）
        self.actions.joint_pos = FrozenLocoJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=action_scale_table_for("go2"),
        )

        # obs 组：actions 项换低层 last_action（55 维契约保持；其余零改动）
        self.observations.policy = KickPolicyCfgCmd()


# ---------------------------------------------------------------------------
# cyberdog2 本体置换（kick 面接入；go2 kick 链全部适配点逐项对账）
#
# go2 耦合点清单（Go2KickFlatEnvCfg 继承链逐项）→ cyberdog2 处置：
#   C1) scene.robot：make_go2_track_profile（Implicit kp40/d1、spawn z0.42、
#       站高 0.227）→ make_cyberdog2_track_profile（DCMotor kp25/d0.5/
#       effort 12、spawn z0.30、FK 站高 0.236——与 velocity 低层 r12 ckpt
#       训练装配同参同源，kick env 执行器契约 = 低层执行器契约）；
#   C2) 狗 spawn xy/yaw：_dog_init pos/rot 覆写在父链打到 go2 cfg 实例上
#       ——本体置换**重建 robot cfg**，须在本体后重放（z 沿 cyberdog2
#       profile 现款 0.30，xy/yaw = mdp.DOG_SPAWN_XY + π/2 同款）；
#   C3) ball 接触 filter：Robot/base + GO2_FOOT_BODIES → Robot/body +
#       CYBERDOG2_FOOT_BODIES（足端 FL/FR/RL/RR_foot 同名面；base 名异）；
#   C4) events body names：add_base_mass / base_external_force_torque 的
#       "base" → "body"（cyberdog2 root link 名）；
#   C5) terminations.base_contact sensor body_names "base" → "body"；
#   C6) robot_fallen 高度阈值：终止 + 惩罚两处 params 同源覆写 0.16 →
#       CYBERDOG2_FALLEN_BODY_Z_THRESHOLD（0.20；依据见 ball_mdp 常量
#       docstring——xiaomi 0.28 基准 + FK 站高 0.236 + 探针步态波谷证据）；
#   C7) actions scale 表：go2 表 → cyberdog2 表（xiaomi deploy 快照
#       uniform 0.25；official 约定不消费——仅 legacy 口径值面就位）；
#   C8) 低层计算频率 per-substep→per-step（50Hz 与低层训练采样率对齐）：
#       kp25 DCMotor 速度饱和（|qd|→31 时力矩→0）+ 200Hz 反应环 = whiplash
#       失稳（diag 实测：2it 零命令 ep_len≈13-16 步、qd 恒钳 ±31、body_z
#       Sag 0.236→0.17 折叠致死；50Hz 同 ckpt 零摔 150 步 body_z 0.23-0.25）
#       ——[cyberdog2-kick] deviation 在案；go2 面保持历史 200Hz 语义不动。
#   无需改：contact_forces prim_path 已 "Robot/.*"（vendored 基座，嵌套
#   prim 树即匹配）；feet_air_time/undesired_contacts 已被 kick 面 None；
#   球物理/obs 55 维/奖励权重/课程几何本体无关；behind_offset 0.45 ≈
#   2×站高比例在双本体下偏差 <5%（go2 0.45/0.227=1.98 vs cyber 0.45/0.236
#   =1.91），几何面沿用不调。
# ---------------------------------------------------------------------------


def _apply_cyberdog2_body_swap(cfg: "Go2KickFlatEnvCfg") -> None:
    """go2→cyberdog2 本体置换（C1-C6；训练面/cmd 面共用单点）。

    在父链 ``__post_init__()`` 完成后调用：整只替换 scene.robot、重放狗
    spawn 常量（C2）、换球接触 filter 与 events/终止 body names、覆写
    robot_fallen 高度阈值（终止 + 奖励两处同源）。
    """
    # C1 本体置换（profile 内含 spawn z0.30 + DCMotor kp25 契约 + soft
    #    limit 0.9；与 velocity 面装配同款——低层 ckpt 执行器契约同参）
    _profile = make_cyberdog2_track_profile(prim_path="{ENV_REGEX_NS}/Robot")
    _profile.articulation_cfg.soft_joint_pos_limit_factor = 0.9
    cfg.scene.robot = _profile.articulation_cfg

    # C2 狗 spawn 常量重放（父链打到旧 go2 cfg 实例上；z 沿 profile 现款）
    _dog_init = cfg.scene.robot.init_state
    _dog_init.pos = (mdp.DOG_SPAWN_XY[0], mdp.DOG_SPAWN_XY[1],
                     float(_dog_init.pos[2]))
    _dog_init.rot = DOG_SPAWN_YAW_QUAT_XYZW

    # C3 球接触 filter（base→body + cyberdog2 足端；显式枚举理由同 go2 面）
    cfg.scene.ball_contact_forces.filter_prim_paths_expr=[
        "{ENV_REGEX_NS}/Robot/body",
        *[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in CYBERDOG2_FOOT_BODIES],
    ]

    # C4/C5 events + 终止 body names（root link 名 body）
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "body"
    cfg.events.base_external_force_torque.params["asset_cfg"]\
        .body_names = "body"
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = "body"

    # C6 robot_fallen 高度阈值 0.20（终止 + 惩罚两处同源；0.16→0.20 依据
    #    见 mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD docstring）
    cfg.terminations.robot_fallen.params["body_z_threshold"] = \
        mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD
    cfg.rewards.robot_fallen.params["body_z_threshold"] = \
        mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD


@configclass
class CyberDog2KickFlatEnvCfg(Go2KickFlatEnvCfg):
    """cyberdog2 U-Kick 球 env 训练面。

    go2 kick 训练面（Go2KickFlatEnvCfg）全部适配 + 本体置换（C1-C6，见
    文件尾 cyberdog2 段头表）：球物理/obs 55 维/奖励权重单源/终止结构/
    课程全部继承零改动——差异只在**本体资产与本体绑定名**（执行器契约与
    velocity 低层 ckpt 训练装配同参：DCMotor kp25/d0.5/effort 12 +
    default_q + PhysX 关节序 CYBERDOG2_JOINT_ORDER_IL）。
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cyberdog2_body_swap(self)


@configclass
class CyberDog2KickFlatEnvCfgCommand(Go2KickFlatEnvCfgCommand):
    """cyberdog2 cmd「真分层」面（任务 Isaac-Kick-Flat-CyberDog2-Upper-Cmd-v0）。

    go2 cmd 面（Go2KickFlatEnvCfgCommand）全部差异只在动作接口的分层
    （3 维命令 → 冻结低层 → 关节目标）+ 本体置换（C1-C6）+ 动作 scale 表
    换 cyberdog2（C7——official 约定不消费该表，均匀 0.5；
    legacy 口径值面就位）。低层 ckpt 缺省按 robot 分派（frozen_weights：
    cyberdog2 → weights/velocity_flat_official/cyberdog2/model_walk.pt，
    launch 期 train.py --robot 注入）。
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cyberdog2_body_swap(self)
        # C7 动作 scale 表换 cyberdog2（父链已挂 go2 表——official 约定
        #    不消费仅打印提示；legacy 口径须本表）
        self.actions.joint_pos.scale = action_scale_table_for("cyberdog2")
        # C8 低层计算频率 → 每环境步 50Hz（kp25 饱和 + 200Hz 反应环失稳
        #    实证；见文件尾 C8 注与 frozen_loco_action 模块头）
        self.actions.joint_pos.low_level_compute_per_substep = False


__all__ = [
    "KickRewardWeightsCfg",
    "KickRandomizationCfg",
    "KickPolicyCfg",
    "Go2KickFlatEnvCfg",
    "Go2KickFlatEnvCfgOracle",
    "KickPolicyCfgCmd",
    "Go2KickFlatEnvCfgCommand",
    "CyberDog2KickFlatEnvCfg",
    "CyberDog2KickFlatEnvCfgCommand",
]
