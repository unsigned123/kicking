# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.ball_env_cfg
#
# U-Kick 球 env（docs/UKICK_BALL_ENV_DESIGN.md §3 规格、§7 拍板、§8-WP2 名称
# 契约）：单 go2 + 单球 + 目标球门（解析判线，无门柱物理块），三臂对照共用
# 训练面 Go2KickFlatEnvCfg；Go2KickFlatEnvCfgOracle 为 oracle/冒烟面（U4b B 臂
# ckpt 直驱，**仅 oracle/冒烟用，禁止用于三臂训练**）。
#
# 继承链（零改动复用；本文件只做 kick 面追加/替换）：
#   LocomotionVelocityRoughEnvCfg（velocity_env_cfg.py，vendor 只读）
#     → UnitreeGo2FlatEnvCfg（flat_env_cfg.py，vendor+upper 只读）
#     → Go2KickFlatEnvCfg（本文件）
#
# donor 移植（soccer_absolute @ master fc51c47，参考副本 /tmp/opencode/
# ukick_donor/；只抄参数/公式/语义，零 import）：
#   - 球物理全套      ← donor env/scene.py:215-247 逐值照抄（radius 0.11 /
#                       mass 0.43 / μs 1.0 μd 0.95 / restitution 0.25 /
#                       lin+ang damping 2.5「松脚即停」高阻尼）
#   - 球侧接触传感器  ← donor env/scene.py:250-256（filter 表达式适配 ut go2
#                       prim 布局：donor 为 4 机 17 体列表，ut 单狗收敛为
#                       base + 4 足端——踢击检测（WP3 kick_count）所需的
#                       保守 filter 面；base 命中兜底躯干触球）
#   - 球 reset 写入   ← donor env/reset.py:257-280 语义（default_root_state +
#                       env_origins → write_root_pose/velocity_to_sim → reset）
#
# KICK 适配表（对 flat 链的全部 override，逐处标注）：
#   K1) scene：+ 球 RigidObjectCfg（scene.ball）+ 球侧 ContactSensor
#       （scene.ball_contact_forces）；terrain/光照/height_scanner=None 沿用继承链。
#   K2) events：+ reset_ball（球 root state 写入）；reset_base 的 pose_range
#       钳零 —— 狗固定 spawn 零采样语义（owner 拍板 #3，保留 velocity_range=
#       (0,0) 与 stand 姿态的现款零速度语义）；spawn 位姿常量面 [X-HK4 H 臂]
#       = DOG_SPAWN_XY + yaw=π/2（init_state 单源，rc1 default⊗delta 语义下
#       reset 后稳定存活）。
#   K3) observations.policy → KickPolicyCfg（55 维 = 42-prefix 契约逐参不动
#       ⊕ base_lin_vel 3 ⊕ ball_rel_b 3 ⊕ goal_rel_b 4 ⊕ push_geometry_b 3）。
#   K4) commands.base_velocity = None（任务无 cmd 面）。
#   K5) rewards task-only（owner 二次拍板 2026-09-30：奖励规则接口对齐
#       soccer_absolute phase0——8 项公式面移植、权重按 go2 平台适配不照搬，
#       权重单源 = KickRewardWeightsCfg（phase0 raw 对照列见其 docstring）；
#       velocity track 系 + 全部 regularization 项仍显式 = None）：
#       goal_scored / robot_fallen / survival / approach_ball / ball_distance /
#       behind_ball / ball_progress（稠密项默认启用，取代旧 ε 预注册档）/
#       ball_out_of_bounds；[X-HK2] 追加自建第 9 项 kick_event（缺省权重
#       0.0 = 零行为影响，D 臂经 overrides 启用——非 phase0 面）。
#   K6) terminations（phase0 active_terminations 对齐）：root_height(0.15)
#       → robot_fallen（donor 双条件 grav>−0.5 ∨ body_z<0.16——高度支 0.16
#       严格覆盖 0.15，见挂载注释）+ robot_out_of_field（工作区框，默认
#       启用）+ goal_scored / ball_out（既有）+ time_out / base_contact
#       （继承保留）；wall_stuck 接口在位不挂载（我方无墙）。
#   K7) episode_length_s = 10.0（500 步 @50Hz；design §3.4）。
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
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
from hierarchical_kick.robots.go2 import GO2_FOOT_BODIES
from hierarchical_kick.core.scale_vec import go2_action_scale_table
from hierarchical_kick.sim import mdp
from hierarchical_kick.sim.flat_env_cfg import UnitreeGo2FlatEnvCfg, UpperPolicyCfg
from hierarchical_kick.sim.mdp.frozen_loco_action import (
    FrozenLocoJointPositionActionCfg,
    low_level_last_action_obs,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# [X-HK4 H 臂] 冷启动饱和 spawn 常量面（owner 指令 2026-10-09：狗出生在球门
# 前 0.3m、球 spawn 离狗 0.1m——"球在脚边+门在鼻尖"整条奖励闭环全速验证）
##


DOG_SPAWN_XY = (0.0, mdp.GOAL_Y - 0.3)
"""狗 spawn（env 系 xy）：门线前 0.3m（GOAL_Y=3.0 → y=2.7）；站球后，
behind_ball 位形天然正确。"""

DOG_SPAWN_YAW_QUAT_XYZW = (0.0, 0.0, math.sin(0.25 * math.pi), math.cos(0.25 * math.pi))
"""狗 spawn 朝向四元数（**xyzw**，isaaclab 3.0 通道序）：yaw=π/2 面朝球门
（进攻轴 +y）。reset 存活性：rc1 ``reset_root_state_uniform`` = default ⊗
delta、本链 reset_base pose_range 钳零（delta=identity）→ 本 quat 每次
reset 后逐位回到。"""

BALL_SPAWN_XY = (0.0, mdp.GOAL_Y - 0.2)
"""球 spawn（env 系 xy）：狗前 0.1m、距门线 0.2m（GOAL_Y=3.0 → y=2.8）。"""


##
# 随机化接口（纯 python，不依赖 isaaclab；owner 拍板 #3：只留接口不启用）
##


@dataclass
class KickRewardWeightsCfg:
    """U-Kick 奖励权重单源（owner 二次拍板 2026-09-30：phase0 规则接口对齐 +
    go2 平台适配，**权重不照搬**）。

    phase0 raw 列 = donor config/phases/phase0_solo.yaml reward_weights（其
    平台 cyberdog2、全局 reward_scale=0.1、episode 3750 步全场赛制——raw 权重
    不可直接搬，先对公式面再调参）。我方适配列量纲理由（go2 站高 0.227 /
    门距 3m / 500 步 @50Hz / 无全局 reward_scale）：

    | 项                      | phase0 raw | 我方默认 | 量纲理由 |
    |-------------------------|-----------|---------|---------|
    | goal_scored             | 500.0     | +10.0   | 保持现值；稀疏主奖励（进球即终止单次） |
    | robot_fallen            | 30.0      | −5.0    | 一次性 mask；≈半个 goal 的痛感（phase0 等效 30×0.1=3，同量级） |
    | survival                | 0.02      | +0.25   | 保持现值 β1（flat U5）；fall penalty 在场后仍提供"活下去的底薪"（500 步全存活 = 125 ≈ 12.5 个 goal——底薪大于痛感，但摔即失全额生存流，结构性反摔） |
    | approach_ball_velocity  | 0.15      | +0.15   | phase0 同值起步；投影速度量纲 m/s × coeff 1.0、cap max_speed 0.5 → 单步 ≤0.075 |
    | ball_distance           | 0.03      | +0.05   | exp 核 ∈(0,1] 无量纲、3m 域全程底薪引导（≈survival 的 20%） |
    | behind_ball             | 0.03      | +0.05   | exp 核 × donor scale 0.35 → 单步 ≤0.0175，弱位形先验不与主引导抢幅值 |
    | ball_progress           | 0.0（旧 tier2 ε） | +0.10 | Δy 差分 × donor scale 5：0.5m/s 球速@50Hz → 单步 0.005、全程 3m 推进 ≈1.5（goal 的 15%，owner 点名启用） |
    | ball_out_of_bounds      | 10.0      | −2.0    | 步 mask 罚；≈8 步正项流水等价（phase0 等效 10×0.1=1，同量级偏严防胡踢） |

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
    kernel 形参（max_dist/temperature/coeff 等）= donor 语义默认，见
    ball_mdp 各 wrapper 签名，可经 ``RewTerm(params={...})`` 覆写。
    """

    goal_scored: float = 10.0
    robot_fallen: float = -5.0
    survival: float = 0.25
    approach_ball_velocity: float = 0.15
    ball_distance: float = 0.05
    # [X-HK3 F 臂] ball_distance exp 核温度（核宽）入单源：缺省 2.0 = donor
    # 语义默认（基线逐位不变）；F 臂经 --kick_rw_overrides 降到 0.5 陡化核
    # （1.0m/0.5m/0.2m 核值 0.135/0.368/0.670，全程增益 ~53/集 vs T=2 的 ~7.4）
    ball_distance_temperature: float = 2.0
    behind_ball: float = 0.05
    ball_progress: float = 0.10
    ball_out_of_bounds: float = -2.0
    # [X-HK2] 踢击事件回报（自建过程引导项，非 phase0 面）：接触力范数
    # >0.1N 判踢 + 0.5s/25 步冷却，1.0 mask × weight = 事件回报幅值。
    # **缺省 0.0 = 零行为影响**（红线纪律：基线面默认无此项贡献；D 臂经
    # --kick_rw_overrides '{"kick_event": 0.5, ...}' 启用）
    kick_event: float = 0.0


@dataclass
class KickRandomizationCfg:
    """kick 面 spawn 随机化接口（默认完全关闭 → 固定 spawn）。

    enabled=False（默认）：球固定 spawn BALL_SPAWN_XY=(0, GOAL_Y−0.2, 0.11)
    （scene init_state 单源——[X-HK4 H 臂] 门线前 0.2m 冷启动饱和课程，旧值
    (0.3, 0, 0.11) G 臂贴脚）、狗固定 spawn DOG_SPAWN_XY + yaw=π/2（
    init_state 单源；reset_base pose_range 钳零保持零采样语义）——owner
    指令 2026-10-09。
    enabled=True：reset_ball 事件在默认 spawn 附近做均匀采样：
      - ball_xy_range：球横向偏移半幅 (x, y)（design §3.1 ±0.3 档）；
      - dog_yaw_range：狗 yaw 采样幅（预留字段——pilot 期接线 reset_base
        pose_range 的 yaw 通道后再消费；当前实现不驱动狗位姿）。
    """

    enabled: bool = False
    ball_xy_range: tuple[float, float] = (0.0, 0.3)
    dog_yaw_range: tuple[float, float] = (-math.pi, math.pi)


##
# reset 事件（isaaclab 面；供 EventTerm 挂载）
##


def reset_ball_root_state(
    env: 'ManagerBasedRLEnv',
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """球 root state reset：写 pose/velocity 进 sim 后 reset。

    donor reset.py:257-280 语义：default_root_state（= scene.ball init_state
    单源 BALL_SPAWN_XY=(0, GOAL_Y−0.2, 0.11)——[X-HK4 H 臂] 门线前 0.2m 冷
    启动饱和课程，manager-based 姿态零旋转、速度零）+ env_origins 偏移
    → write_root_pose_to_sim / write_root_velocity_to_sim → reset（同步内部
    buffer）。kick_randomization.enabled=True 时叠加横向均匀采样（owner 拍板
    #3：接口在位、默认关闭）。（旧 ball_progress_shaping 的清缓存钩子已随
    该项移除——ball_progress_dense 的 prev_y 缓存由 wrapper 侧"每集首步
    重播种"（episode_length_buf==1 → prev 取 cur，差分记 0）对 reset 跳变
    免疫，无需 reset 钩子，见 ball_mdp.ball_progress_dense 口径。）
    """
    ball = env.scene[asset_cfg.name]
    # 默认 root state（pos+quat+vel，init_state 单源）+ env origin 偏移
    root_state = ball.data.default_root_state.torch.clone()[env_ids]
    root_state[:, :3] += env.scene.env_origins[env_ids]
    # 随机化接口（owner 拍板 #3：默认 enabled=False → 完全固定 spawn）
    kr = env.cfg.kick_randomization
    if kr.enabled:
        rand = torch.rand(len(env_ids), 2, device=root_state.device)
        half = ball.data.root_pos_w.torch.new_tensor(kr.ball_xy_range)
        root_state[:, 0] += (rand[:, 0] * 2.0 - 1.0) * half[0]
        root_state[:, 1] += (rand[:, 1] * 2.0 - 1.0) * half[1]
    # 写 sim（donor reset.py:279-280 同款 manager-based 写法）+ 同步内部 buffer
    ball.write_root_pose_to_sim(root_state[:, :7], env_ids)
    ball.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    ball.reset(env_ids=env_ids)


##
# obs 组（55 维 kick 契约；K3）
##


@configclass
class KickPolicyCfg(ObsGroup):
    """kick 面 policy obs（55 维单帧 = KICK_OBS_DIM）。

    前 5 项与 UpperPolicyCfg（flat_env_cfg.py 42-prefix 契约）**逐参相同**
    （噪声/scale 原样，joint_pos/joint_vel/base_ang_vel/projected_gravity/
    actions = 42 维 decoder proprio 契约，绝不再加第二噪声源）；尾部追加
    base_lin_vel 3 ⊕ ball_rel_b 3 ⊕ goal_rel_b 4 ⊕ push_geometry_b 3。
    单帧无 history（与 upper 现栈一致，不抄 donor obs history buffer）。
    """

    # -- decoder proprio 42（首 term；逐参 = UpperPolicyCfg，契约切片不动） --
    # [表#1] joint_pos (0:12)：U±0.01，scale 1.0
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-NOISE_JOINT_POS, n_max=NOISE_JOINT_POS))
    # [表#2] joint_vel (12:24)：U±0.5 × scale 0.05（V6 契约；官方项不照抄）
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
    # [表#8] ball_rel_b (45:48)：body 系 xy/3 + 距离/3（donor observations.py:73-79）
    ball_rel_b = ObsTerm(func=mdp.ball_rel_b)
    # [表#9] goal_rel_b (48:52)：body 系 xy/3 + sin + cos（donor observations.py:81-88）
    goal_rel_b = ObsTerm(func=mdp.goal_rel_b)
    # [表#10] push_geometry_b (52:55)：cos_behind + fwd-vs-btog sin/cos
    #         （donor observations.py:205-235）
    push_geometry_b = ObsTerm(func=mdp.push_geometry_b)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


##
# 训练面（三臂共用）
##


@configclass
class Go2KickFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """U-Kick 球 env 训练面（任务 Isaac-Kick-Flat-Unitree-Go2-Upper-v0）。

    官方 flat 谱系 + KICK 适配 K1–K7（文件头表）；奖励 = phase0 规则接口
    8 项（owner 二次拍板 2026-09-30：公式语义对齐 soccer_absolute phase0、
    权重按 go2 平台适配，单源 KickRewardWeightsCfg；叙事："奖励/终止配方
    结构对齐前人 phase0，权重按我方平台适配，唯一变量是动作接口"——三臂
    逐位同奖励，公平性不变）。obs/终止/球物理三臂逐位同。
    """

    def __post_init__(self):
        # post init of parent（官方 flat 谱系全链：plane terrain、逐关节 scale、
        # root_height 终止、survival 项等）
        super().__post_init__()

        # [K1] 球场景（donor scene.py:215-247 逐值照抄 + provenance 行内注；
        #      init_state pos = 固定 spawn BALL_SPAWN_XY=(0, GOAL_Y−0.2, 0.11)
        #      ——[X-HK4 H 臂] 门线前 0.2m 冷启动饱和课程，旧值 (0.3, 0, 0.11)
        #      为 X-HK3 G 臂贴脚）
        self.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.11,  # 球半径（donor scene.py:218，对齐 cyberdog2_rl）
                # visual_material=donor 同款 PreviewSurfaceCfg（白色）——
                # **headless 不挂**：A100 缺 libXt/libGLU 时 preview 材质
                # spawn 死旋（flat_env_cfg.py 头 U6 同型实测；录像阶段在
                # 带 --enable_cameras 的会话内再启用）
                # 物理材质（donor scene.py:223-227）
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.95,
                    restitution=0.25,
                ),
                # 碰撞属性（donor scene.py:229-231）
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                # 刚体属性（donor scene.py:233-237；高阻尼「松脚即停」）
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=2.5,
                    angular_damping=2.5,
                ),
                # 质量属性（donor scene.py:238-240）
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=0.43,
                ),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                # [X-HK4 H 臂] 球 spawn 门线前 0.2m（BALL_SPAWN_XY，owner
                # 指令"球 spawn 离狗 0.1m"）——dog/球/门三点一线 +y。⚠️ 已知
                # 前案（X-HK3 G 臂裁定）：0.1m 处球面（r0.11 表面延伸进机体）
                # 与狗前腿初始穿透、PhysX 解算弹飞——H 臂按 owner 指令故意
                # 复现该距离直接试效果；若训练首 iter 出现 goal 假信号/球飞，
                # 优先归因于此（弹飞朝 +y = 瞬间过门线假 goal）。旧值
                # (0.3, 0, 0.11) 为 G 臂贴脚（X-HK3 修法=鼻端/足端接触线）。
                pos=(BALL_SPAWN_XY[0], BALL_SPAWN_XY[1], 0.11),
                rot=(0.0, 0.0, 0.0, 1.0),  # 无旋转（xyzw 单位元；isaaclab 3.0 通道序）
            ),
        )
        # [K1] 球侧接触传感器（donor scene.py:250-256 语义；filter 表达式适配
        #      ut go2 prim 布局：donor 4 机 17 体列表 → ut 单狗 base + 4 足端
        #      的保守 filter 面——足端主踢击检测 + base 兜底躯干触球；WP3
        #      kick_count（Lake 指标）消费本传感器）
        self.scene.ball_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            update_period=0.0,
            history_length=20,
            # filter 显式枚举（单源 GO2_FOOT_BODIES；**不用 .* 通配**——
            # env_*/Robot/.*_foot 在 10 env 下匹配 40 prim 触发 tensor API
            # expected/found 警告且力矩阵形状漂移；显式名逐 env 恰 5 prim）
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/base",
                *[f"{{ENV_REGEX_NS}}/Robot/{b}" for b in GO2_FOOT_BODIES],
            ],
        )

        # [K2] 随机化接口挂载（owner 拍板 #3：enabled=False 默认完全关闭）
        self.kick_randomization = KickRandomizationCfg()

        # [X-HK4 H 臂] 狗 spawn 常量面（owner 指令 2026-10-09）：门线前 0.3m、
        #      yaw=π/2 面朝球门——站球后，behind_ball 位形天然正确。实现走
        #      init_state 单源（spawn z 沿用 profile 现款，不硬编码）：rc1
        #      reset_root_state_uniform = default ⊗ delta、本链 pose_range 钳零
        #      （delta=identity）→ 每次 reset 后位姿逐位回到本值，yaw 稳定
        #      存活；下方 reset_base 零采样语义保持（狗固定 spawn）。
        _dog_init = self.scene.robot.init_state
        _dog_init.pos = (DOG_SPAWN_XY[0], DOG_SPAWN_XY[1], float(_dog_init.pos[2]))
        _dog_init.rot = DOG_SPAWN_YAW_QUAT_XYZW

        # [K2] 狗固定 spawn 零采样（owner 拍板 #3；velocity_range 已由
        #      flat 链钳零 = 现款零速度语义；stand 姿态经 joint reset 继承）
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # [K2] 球 reset 事件（写入 root state；旧 shaping 清缓存钩子已移除——
        #      ball_progress_dense 缓存由 wrapper"每集首步重播种"对 reset
        #      跳变免疫，[X-HK4]）
        self.events.reset_ball = EventTerm(
            func=reset_ball_root_state,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("ball")},
        )

        # [K3] obs 组替换（55 维 kick 契约；42-prefix 逐参不动）
        self.observations.policy = KickPolicyCfg()

        # [K4] 无命令面（任务无 cmd；KickPolicyCfg 无 velocity_commands 项，
        #      悬空引用不存在）
        self.commands.base_velocity = None

        # [K5] 奖励 task-only（owner 二次拍板 2026-09-30）——velocity track 系 +
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
        # 权重单源（KickRewardWeightsCfg；phase0 raw 对照表见类 docstring）
        self.kick_reward_weights = KickRewardWeightsCfg()
        w = self.kick_reward_weights
        # —— phase0 规则接口 8 项（公式面移植 provenance 见 ball_mdp 头）——
        # 进球稀疏主奖励（幅值由 weight 承载；goal_scored 即终止，无重复触发）
        self.rewards.goal_scored = RewTerm(func=mdp.goal_scored_reward, weight=w.goal_scored)
        # 摔倒罚：状态直检 1.0 mask（奖励先于终止计算，不依赖 termination
        # manager；donor rewards.py:37 单狗化）；阈值与 robot_fallen 终止同源
        self.rewards.robot_fallen = RewTerm(
            func=mdp.robot_fallen_penalty, weight=w.robot_fallen,
            params={"grav_threshold": mdp.FALLEN_GRAV_THRESHOLD,
                    "body_z_threshold": mdp.FALLEN_BODY_Z_THRESHOLD})
        # survival（恒 +1.0 × β1）显式挂载（继承 flat U5 的 value 语义不变，
        # 权重入单源；fall penalty 在场后仍提供活下去的底薪）
        self.rewards.survival = RewTerm(func=mdp.alive_reward, weight=w.survival)
        # 稠密引导族（owner 拍板：默认启用；kernel 形参 = ball_mdp 签名默认，
        # 可经 RewTerm(params={...}) 覆写）
        self.rewards.approach_ball = RewTerm(func=mdp.approach_ball_velocity_reward, weight=w.approach_ball_velocity)
        self.rewards.ball_distance = RewTerm(func=mdp.ball_distance_reward, weight=w.ball_distance, params={"temperature": w.ball_distance_temperature})
        self.rewards.behind_ball = RewTerm(func=mdp.behind_ball_reward, weight=w.behind_ball)
        # 球向门 Δy 进攻方向差分（donor ball_progress_dense；替换旧
        # ball_progress_shaping ε 预注册档——其 prev-cur 距离缓存及 reset
        # 钩子已整体移除，ball_progress 自带缓存对 reset 跳变免疫）
        self.rewards.ball_progress = RewTerm(func=mdp.ball_progress_dense, weight=w.ball_progress)
        # [X-HK2] 踢击事件回报（自建项，非 phase0；检测面 = kick_metrics
        # 单源。缺省 weight 0.0 → 挂载但零贡献——基线行为面逐位不变）
        self.rewards.kick_event = RewTerm(func=mdp.kick_event_reward, weight=w.kick_event)
        # 球出界步 mask 罚（复用 ball_out 半径判据）
        self.rewards.ball_out_of_bounds = RewTerm(func=mdp.ball_out_of_bounds_penalty, weight=w.ball_out_of_bounds)

        # [K6] 终止（phase0 active_terminations 对齐）：
        #  - robot_fallen 替换继承 root_height(0.15)：donor 双条件
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
        #    不该为追球跑出工作区；donor 场界矩形判据 → 我方任务域非对称框）
        self.terminations.robot_out_of_field = DoneTerm(
            func=mdp.robot_out_of_field,
            params={"x_min": mdp.ROBOT_FIELD_X_MIN, "x_max": mdp.ROBOT_FIELD_X_MAX,
                    "y_min": mdp.ROBOT_FIELD_Y_MIN, "y_max": mdp.ROBOT_FIELD_Y_MAX})
        #  - wall_stuck 接口在位**不挂载**（我方无墙；mdp.wall_stuck 备用，
        #    donor dist+speed 双条件，引入场地墙段后再挂）
        #  - goal_scored / ball_out（既有）+ time_out / base_contact（继承；
        #    base_contact 与 robot_fallen 互补：贴地躯干触地即终止，兜底
        #    双条件判据覆盖不到的接触形态，如腹部擦地倾斜角未超阈值瞬间）
        self.terminations.goal_scored = DoneTerm(func=mdp.goal_scored)
        self.terminations.ball_out = DoneTerm(func=mdp.ball_out)

        # 权重同步（默认值刷一遍；实例级调参 = 改 kick_reward_weights 字段
        # 后再调本方法，或直接改 rewards.<term>.weight——幂等）
        self.sync_kick_reward_weights()

        # [K7] episode 时长 10s（500 步 @50Hz；design §3.4）
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
# oracle / 冒烟面（U4b B 臂 ckpt 直驱；WP4 验收门用）
##


@configclass
class Go2KickFlatEnvCfgOracle(Go2KickFlatEnvCfg):
    """oracle 面：48 维 velocity 契约 + 命令管理器在位（teacher 直写 cmd）。

    ⚠️ 仅 oracle/冒烟用，禁止用于三臂训练。U4b B 臂 policy 的 obs 契约 =
    proprio 42 ⊕ base_lin_vel 3 ⊕ cmd 3（48 维），命令管理器每步由 scripted
    teacher（WP4：几何状态机）直写 command buffer；球场景/终止/奖励与训练面
    （Go2KickFlatEnvCfg）逐位同 —— oracle 链：teacher 几何 → 速度命令 [vx,vy,ωz]
    → U4b B 臂 ckpt → 关节动作（design §4 验收门：10 episode ≥3 进球、零摔）。
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
# arm D「真分层」面（U4b B 臂 ckpt 冻结低层 + 3 维速度命令高层）
##


@configclass
class KickPolicyCfgCmd(KickPolicyCfg):
    """arm D 高层 obs 组（55 维契约保持；仅 actions 项换源）。

    唯一 diff（对照 :class:`KickPolicyCfg`）：``actions`` 项从
    ``mdp.last_action``（= action_manager.action，arm D 下是 3 维命令）换成
    :func:`low_level_last_action_obs`（冻结低层上一步 12 维 pre-scale 输出）
    ——42-prefix 通道序/维数与三臂逐位同构（55 = 42+3+3+4+3），通道语义
    同为「关节级 last action」。其余项（joint_pos/joint_vel/base_ang_vel/
    projected_gravity/base_lin_vel/ball_rel_b/goal_rel_b/push_geometry_b）
    **零改动继承**（噪声/缩放逐参同源）；奖励/终止不在本类——继承训练面。
    """

    # [表#5] last_action (30:42)：无噪声；arm D 语义 = 冻结低层上一步输出
    #        （3 维命令动作不进本通道——见 frozen_loco_action.
    #        low_level_last_action_obs docstring）
    actions = ObsTerm(func=low_level_last_action_obs)

    def __post_init__(self):
        super().__post_init__()


@configclass
class Go2KickFlatEnvCfgCommand(Go2KickFlatEnvCfg):
    """arm D「真分层」面（任务 Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0）。

    与三臂训练面（Go2KickFlatEnvCfg）的**全部差异只在动作接口**：
      - ``actions.joint_pos`` → :class:`FrozenLocoJointPositionActionCfg`
        （3 维 [vx,vy,wz] 命令 → 写命令 buffer → 冻结低层 → 12
        关节位置目标；低层约定缺省 official = 官方 velocity-flat 口径
        （weights/velocity_flat_official/model_walk.pt，均匀 0.5 动作
        scale），--low_level_convention legacy 切旧 U4b B 臂
        model_2999.pt + 逐关节 scale_vec；ckpt 路径 launch 期填充，
        构造期存在性 fail-fast）；
      - ``commands.base_velocity`` 恢复（照 :class:`Go2KickFlatEnvCfgOracle`
        的恢复写法——命令面在位，ranges 任意：每步被高层动作覆写，采样
        值不消费；resample 竞争分析见 frozen_loco_action 模块头）；
      - ``observations.policy`` → :class:`KickPolicyCfgCmd`（仅 actions 项
        换低层 last_action，55 维契约保持）。
    obs 尾部（ball/goal/push）/奖励（phase0 8 项）/终止（phase0 集）/球
    物理/PPO 面全部继承训练面**零改动**——三臂公平性纪律。
    ⚠️ 仅供 arm D（--actor command）；三臂训练禁用本面。
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
        #   - legacy：逐关节 scale_vec（单源表）+ U4b B model_2999.pt。
        # ckpt 路径留空 = 按约定缺省（run_upper launch 期经 --low_level_ckpt
        # 填绝对路径，构造期存在性 fail-fast）
        self.actions.joint_pos = FrozenLocoJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=go2_action_scale_table(),
        )

        # obs 组：actions 项换低层 last_action（55 维契约保持；其余零改动）
        self.observations.policy = KickPolicyCfgCmd()


__all__ = [
    "KickRewardWeightsCfg",
    "KickRandomizationCfg",
    "KickPolicyCfg",
    "Go2KickFlatEnvCfg",
    "Go2KickFlatEnvCfgOracle",
    "KickPolicyCfgCmd",
    "Go2KickFlatEnvCfgCommand",
    "reset_ball_root_state",
]
