# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# hierarchical_kick.sim.mdp.ball_mdp — U-Kick 球 env 的 mdp 函数面（自建，非
# vendor）。公式/几何语义对齐上游 soccer_absolute 实现（只移植语义，零
# import、零代码路径接触），逐函数 provenance：
#   - ball_rel_b      ← env/observations.py:73-79（ball_rel_xy + dist）
#   - goal_rel_b      ← env/observations.py:81-88（goal_rel_xy + sin/cos）
#   - push_geometry_b ← env/observations.py:205-235（cos_behind + fwd-vs-btog）
#   - goal_scored     ← mdp/terminations.py 判线（球过线且 |x| < 门半宽）
#   - ball_out        ← mdp/terminations.py:11-25 出界判据的半径化改写
#   - 球参数常量      ← env/scene.py:215-247（radius 0.11 / mass 0.43）
#   - robot_fallen[_penalty] ← terminations.py:42-59 双条件 /
#                       rewards.py:37-102 单狗直检 mask 化
#   - approach_ball_velocity_reward ← rewards.py:267-311（视向投影 + 封顶 +
#                       正/负非对称 + 距离窗门控）
#   - ball_distance_reward     ← rewards.py:723-737（exp 核）
#   - behind_ball_reward       ← rewards.py:778-795（球后位形 exp 核）
#   - ball_progress_dense      ← rewards.py:677-702（Δy 单边 clamp + 近球门控）
#   - ball_out_of_bounds_penalty ← rewards.py:461-479 的步 mask 化
#   - robot_out_of_field       ← terminations.py:27-40 场界框判据
#   - wall_stuck               ← terminations.py:78-93（接口在位、cfg 不挂载）
#   - kick_event_reward / reset_ball_root_state / ball_spawn_* ← 本包自建
#                       （无上游对应；权重单源见 ball_env_cfg
#                       .KickRewardWeightsCfg，kick_event 缺省 0 = 零行为影响）
#
# 契约：
#   - body 系变换：isaaclab.utils.math.quat_apply_inverse + root_quat_w
#     （**禁手搓旋转矩阵**）。离线（本地无 isaaclab，仅 pytest）降级到仓库
#     离线数学单源 hierarchical_kick.sim.math_utils.quat_rotate_inverse
#     （vendored 等价实现，见该文件头注）；sim 路径恒走 isaaclab 原函数。
#   - 逐 env goal 帧：goal 相关几何一律 ``goal = env_origins[:, :2] +
#     GOAL_CENTER``（与球 reset 的 default_root_state + env_origins 同帧同源；
#     世界系常量在网格布局 env_origins 下会在远端 env 全部错位）。
#   - 离线可测性：纯计算拆成模块级 _*_impl 纯函数（tensor/float in-out），
#     isaaclab-facing wrapper 只做 env.scene 取数与调用。
# ---------------------------------------------------------------------------

"""Ball-kick MDP functions for the upper U-Kick environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg`
/ :class:`isaaclab.managers.TerminationTermCfg` / :class:`isaaclab.managers.RewardTermCfg`
objects of the U-Kick env (``ball_env_cfg.Go2KickFlatEnvCfg``).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

# body 系旋转原语（见文件头：sim 路径 = isaaclab 原函数；离线 = vendored 等价面）。
# 防呆：仅当绑定面确系 isaaclab 树内原函数时才走 isaaclab 分支——
# 本地测试套件存在注入 ``sys.modules["isaaclab"]`` 假桩的用例（无 utils.math
# 或语义不受控），一律降级到仓库离线数学单源，保证离线绑定面确定性。
# 绑定判定用 ``__module__.startswith("isaaclab")``（树内即真）：精确匹配
# "isaaclab.utils.math" 对不同 isaaclab 构建（源码树 vs pip 包）过严——同一
# 函数可能经包内 re-export 落在别名模块上；测试桩函数的 ``__module__`` 是
# 测试模块名，仍被排除。
try:
    from isaaclab.utils.math import quat_apply_inverse as _isaaclab_quat_apply_inverse

    if getattr(_isaaclab_quat_apply_inverse, "__module__", "").startswith("isaaclab"):
        _quat_apply_inverse = _isaaclab_quat_apply_inverse
    else:
        raise ImportError("isaaclab.utils.math 绑定面非原函数（测试桩）→ 离线降级")
except ImportError:
    from hierarchical_kick.sim.math_utils import quat_rotate_inverse as _quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# 任务几何常量（单一位置定义；ball_env_cfg.py 等消费方 import 此处）。
##

GOAL_Y = 3.0
"""球门线 y 坐标（任务域 3m）。"""

GOAL_HALF_WIDTH = 0.75
"""球门半宽（mouth 宽 1.5m）。"""

GOAL_CENTER = (0.0, GOAL_Y)
"""球门中点（**env 系**水平面 xy 偏移；判线/出界/几何特征共用锚点）。

世界系 goal center = env.scene.env_origins[:, :2] + GOAL_CENTER（逐 env；
与球 reset 的 default_root_state + env_origins 同帧同源——网格布局
env_origins 下用世界系常量判线/出界会在远端 env 错位）。"""

DOG_SPAWN_XY = (0.0, 0.0)
"""狗 spawn（env 系 xy）：原点、yaw=π/2 面朝球门（进攻轴 +y），固定不随机。
单一位置定义：ball_env_cfg 消费于 robot init_state（reset_base pose_range
钳零 → reset 后稳定回到），本文件 reset_ball_root_state 消费于球出生位
（狗前方沿 +y 推算）。"""

CURRICULUM_D_MAX_START = 0.5
"""d_max 课程起点（规格公式定值 0.5）：warmup 起点球距狗 0.5m（近距快反馈），
随训练线性扩张到全任务域。"""

CURRICULUM_Y_MARGIN = 0.2
"""球出生 y 上限 margin：``y ≤ GOAL_Y − CURRICULUM_Y_MARGIN``
（防 reset 即在门内 → 首步假 goal；与固定 spawn BALL_SPAWN_XY 的
GOAL_Y−0.2 同 margin 口径）。"""

BALL_OUT_RADIUS = 6.0
"""球出界半径（球心水平距 goal center > 6m 终止；design §3.4）。"""

KICK_OBS_DIM = 55
"""kick 面 policy obs 总维 = 42(proprio 契约) + 3(lin_vel) + 3(ball) + 4(goal) + 3(push)。"""

BALL_RADIUS = 0.11
"""球半径（标准足球，对齐 cyberdog2_rl）。"""

BALL_MASS = 0.43
"""球质量（标准足球 0.43kg）。"""

FALLEN_GRAV_THRESHOLD = -0.5
"""摔判定重力阈值（上游 terminations.py:42 签名默认）。

站立时 projected_gravity_b[:,2] ≈ −1（重力沿机体 −z），倒地趋 0/正值；
``proj_grav_z > −0.5`` 即机体相对竖直倾斜超 ~60°（cos60°=0.5）。"""

FALLEN_BODY_Z_THRESHOLD = 0.16
"""摔判定 base 高度阈值（上游 terminations.py:42 签名默认）。

go2 站高 0.227（flat 链 root_height 阈值 0.15 的同一站高参考）→ 阈值
0.16 余量 0.067m ≈ 30% 站高：蹲伏步态的质心波谷不误杀，瘫倒/侧翻必触发。
（双条件之一即摔，与重力支互补——直立姿态下被压低的高度由本支兜底。）"""

ROBOT_FIELD_X_MIN = -1.0
ROBOT_FIELD_X_MAX = 5.0
ROBOT_FIELD_Y_MIN = -4.0
ROBOT_FIELD_Y_MAX = 4.0
"""机器人工作区框（robot_out_of_field 终止；env 系，相对逐 env origin）。

任务域 3m（门 y=+3）+ 球出界半径 6m 追球缓冲：x∈[−1,5]（门线后 1m 缓冲
到门前 2m），y∈[−4,4] 对称横幅。单狗无墙球会滚远——球逃逸由 ball_out
处置，机器人本身不该为追球跑出工作区，故默认启用本框。"""

# /3 归一尺度（ball/goal 相对量统一除 3.0 —— 对应 3m 任务域）
_DIST_NORM = 3.0
# 零向量防除零（上游 observations.py:218/222 同款）
_EPS = 1e-8
# ball_progress_dense 的 prev ball_y 缓存挂载点（env 对象属性名）
_PROGRESS_CACHE_ATTR = "_kick_prev_ball_y"
# kick_event_reward 的缓存挂载点（env 对象属性名；[N,2] float =
# (prev_call_step, last_kick_step)，NaN 哨兵懒播种——ball_progress 缓存同款）
_KICK_EVENT_CACHE_ATTR = "_kick_event_last_step"


# ---------------------------------------------------------------------------
# 纯函数实现面（tensor/float in-out；零 isaaclab 依赖 → 离线 pytest 直测）
# ---------------------------------------------------------------------------


def _goal_center_batch(ref: torch.Tensor, goal_center) -> torch.Tensor:
    """goal center 归一面：tuple → 广播 [N,2]；[N,2] tensor → 逐 env 直用。

    tuple 入口 = 标量便利面（env_origins=0 时与旧世界系版本逐位一致）；
    [N,2] tensor 入口 = wrapper 侧逐 env goal 帧（env_origins + GOAL_CENTER）。
    """
    if isinstance(goal_center, torch.Tensor):
        return goal_center.to(device=ref.device, dtype=ref.dtype)
    return ref.new_tensor(goal_center).expand(ref.shape[0], 2)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """xyzw 四元数 → yaw（标准公式）。

    四元数约定全链单序 xyzw（数据层 ``.torch`` 视图与 isaaclab.utils.math
    同序；本包离线降级 math_utils 同序）。
    """
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ball_rel_b_impl(
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    quat: torch.Tensor,
    norm_scale: float = _DIST_NORM,
) -> torch.Tensor:
    """[N,3] = [ball_rel_xy/norm_scale, ‖ball_rel_xy‖/norm_scale]（body 系 xy）。

    上游 observations.py:73-79 语义：世界系 xy 差 → pad z=0 → 逆旋转到
    body 系 → xy 除尺度、距离对「除尺度后的 xy」取模。
    """
    diff_xy = ball_pos[:, :2] - base_pos[:, :2]
    # pad z=0 后整向量逆旋转（quat_apply_inverse + root_quat_w；禁手搓旋转矩阵）
    diff = torch.cat([diff_xy, torch.zeros_like(diff_xy[:, :1])], dim=-1)
    rel = _quat_apply_inverse(quat, diff)
    rel_xy = rel[:, :2] / norm_scale
    rel_dist = torch.linalg.norm(rel_xy, dim=-1, keepdim=True)
    return torch.cat([rel_xy, rel_dist], dim=-1)


def _goal_rel_b_impl(
    base_pos: torch.Tensor,
    quat: torch.Tensor,
    goal_center: tuple[float, float] | torch.Tensor = GOAL_CENTER,
    norm_scale: float = _DIST_NORM,
) -> torch.Tensor:
    """[N,4] = [goal_rel_xy/norm_scale, sin(atan2(y,x)), cos(atan2(y,x))]（body 系）。

    上游 observations.py:81-88 语义：目标方向角 = body 系相对向量的 atan2
    （除正标量不改角度，故直接用未除尺度的 rel 取角，同上游）。
    goal_center：tuple（广播，世界系便利入口）或 [N,2] 逐 env tensor（逐 env
    goal 帧 = env_origins + GOAL_CENTER）。
    """
    goal = _goal_center_batch(base_pos, goal_center)
    diff_xy = goal - base_pos[:, :2]
    diff = torch.cat([diff_xy, torch.zeros_like(diff_xy[:, :1])], dim=-1)
    rel = _quat_apply_inverse(quat, diff)
    rel_xy = rel[:, :2] / norm_scale
    angle = torch.atan2(rel[:, 1:2], rel[:, 0:1])
    return torch.cat([rel_xy, torch.sin(angle), torch.cos(angle)], dim=-1)


def _push_geometry_b_impl(
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    quat: torch.Tensor,
    goal_center: tuple[float, float] | torch.Tensor = GOAL_CENTER,
) -> torch.Tensor:
    """[N,3] = [cos_behind, sin(fwd×btog), cos(fwd·btog)]。

    上游 observations.py:205-235 语义：显式推球几何特征——
      cos_behind = dot(unit(狗→球), unit(球→门))（=1 即狗在球正后方、对准门线）；
      fwd = (cos yaw, sin yaw) 为机体朝向（世界系水平面）；
      sin/cos(fwd vs btog) 显式给出机体朝向与球→门方向的夹角三角量。
    零向量处 +1e-8 防除零（同上游）。
    goal_center：tuple（广播，世界系便利入口）或 [N,2] 逐 env tensor（逐 env
    goal 帧 = env_origins + GOAL_CENTER）。
    """
    ball_xy = ball_pos[:, :2]
    robot_xy = base_pos[:, :2]
    robot_yaw = _yaw_from_quat(quat).squeeze(-1)

    # 球→门方向（世界系水平面；逐 env goal 帧）
    goal = _goal_center_batch(base_pos, goal_center)
    btog = goal - ball_xy
    btog_dir = btog / (torch.linalg.norm(btog, dim=-1, keepdim=True) + _EPS)

    # 狗→球方向（世界系水平面）
    to_ball = ball_xy - robot_xy
    to_ball_dir = to_ball / (torch.linalg.norm(to_ball, dim=-1, keepdim=True) + _EPS)

    # cos(behind)：dot(狗→球, 球→门)
    cos_behind = (to_ball_dir * btog_dir).sum(dim=-1)

    # 机体朝向（世界系水平面单位向量）
    fwd_x = torch.cos(robot_yaw)
    fwd_y = torch.sin(robot_yaw)

    # 朝向 vs 球→门方向 的夹角三角量（sin 用 2D 叉积符号角，同上游）
    cos_fwd_btog = fwd_x * btog_dir[:, 0] + fwd_y * btog_dir[:, 1]
    sin_fwd_btog = fwd_x * btog_dir[:, 1] - fwd_y * btog_dir[:, 0]

    return torch.stack([cos_behind, sin_fwd_btog, cos_fwd_btog], dim=-1)


def _goal_scored_impl(
    ball_pos: torch.Tensor,
    goal_center: tuple[float, float] | torch.Tensor = GOAL_CENTER,
    half_width: float = GOAL_HALF_WIDTH,
) -> torch.Tensor:
    """[N] bool：球过门线（env 系 y > 0）且 |env 系 x| < half_width（解析判线，无门柱块）。

    判线以逐 env goal center 为原点：``ball_xy − goal_center`` 的 y 分量 > 0
    且 |x 分量| < half_width（goal_center 为默认 (0, GOAL_Y) 时与旧世界系
    版本逐位一致）。
    """
    goal = _goal_center_batch(ball_pos, goal_center)
    rel = ball_pos[:, :2] - goal
    return (rel[:, 1] > 0) & (rel[:, 0].abs() < half_width)


def _ball_out_impl(
    ball_pos: torch.Tensor,
    center: tuple[float, float] | torch.Tensor = GOAL_CENTER,
    radius: float = BALL_OUT_RADIUS,
) -> torch.Tensor:
    """[N] bool：球心水平距 goal center > radius（上游出界判据的半径化改写）。

    center：tuple（广播，世界系便利入口）或 [N,2] 逐 env tensor（逐 env
    goal 帧 = env_origins + GOAL_CENTER——远端 env 必须逐 env 判距）。
    """
    center_b = _goal_center_batch(ball_pos, center)
    offset = ball_pos[:, :2] - center_b
    return torch.linalg.norm(offset, dim=-1) > radius


def _fallen_mask_impl(
    proj_grav_z: torch.Tensor,
    body_z: torch.Tensor,
    grav_threshold: float = FALLEN_GRAV_THRESHOLD,
    body_z_threshold: float = FALLEN_BODY_Z_THRESHOLD,
) -> torch.Tensor:
    """[N] bool：摔判定双条件（上游 terminations.py:42-59 逐行同语义）。

    ``fallen = (proj_grav_z > grav_threshold) | (body_z < body_z_threshold)``
    ——站立 proj_grav_z≈−1，倾斜/倒地趋 0（> −0.5 即倾斜超 ~60°）；base 高
    < 0.16m 兜底直立被压低的高度（go2 站高 0.227，余量 0.067m ≈ 30% 站高）。
    两阈值均严格不等号（同上游）。
    """
    return (proj_grav_z > grav_threshold) | (body_z < body_z_threshold)


def _robot_fallen_penalty_impl(
    proj_grav_z: torch.Tensor,
    body_z: torch.Tensor,
    grav_threshold: float = FALLEN_GRAV_THRESHOLD,
    body_z_threshold: float = FALLEN_BODY_Z_THRESHOLD,
) -> torch.Tensor:
    """[N] float32：摔倒罚 1.0 mask（幅值由 RewardTermCfg(weight<0) 承载）。

    上游 rewards.py:37-102 的单狗化：状态直检 mask（**不依赖
    termination manager**——manager-based 每步时序为物理步 → interval 事件
    → 奖励 compute → 终止 compute，奖励面先于终止计算，读 termination
    manager 会拿到上一步结果）；上游的碰撞责任归因与上升沿面为多智能体
    设计，单狗无撞击者、且摔后即被 robot_fallen 终止（无"倒地期间每步
    累加"问题），故收敛为直检 mask，self-fall 语义（无碰撞 → 罚倒地者
    自身）逐位保留。
    """
    return _fallen_mask_impl(proj_grav_z, body_z,
                             grav_threshold, body_z_threshold).to(dtype=torch.float32)


def _approach_ball_velocity_reward_impl(
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    base_vel_w: torch.Tensor,
    max_dist: float = 3.0,
    min_dist: float = 0.0,
    positive_coeff: float = 1.0,
    negative_coeff: float = 2.0,
    speed_threshold: float = 0.0,
    max_speed: float = 0.5,
) -> torch.Tensor:
    """[N] float32：朝球速度在"狗→球"视向投影的方向奖励（方向奖励非速度奖励）。

    上游 rewards.py:267-311 公式逐行同语义：
      proj_vel = dot(ball−pos, vel) / (‖ball−pos‖ + 1e-8)   （视向投影 m/s）
      capped   = clamp(proj_vel, ±max_speed)                （封顶：慢速正确
                方向 = 全奖励，高速冲撞不额外给分）
      raw      = +positive_coeff·capped（近球）/ −negative_coeff·|capped|（远离，
                负支更重）
      reward   = raw · (dist ∈ (min_dist, max_dist)) · (speed ≥ speed_threshold)
    参数面差异（仅默认值，公式不变）：上游 max_dist=0.5 为近球门控；本包
    max_dist=3.0 覆盖 3m 任务域全程（球 spawn 1m、门 3m——远球引导语义），
    其余系数同上游签名默认。
    """
    vec_to_ball = ball_pos - base_pos                              # [N,3]
    dist = torch.norm(vec_to_ball, dim=-1)                         # [N]
    speed = torch.norm(base_vel_w, dim=-1)                         # [N]
    dot = (vec_to_ball * base_vel_w).sum(dim=-1)                   # [N]
    proj_vel = dot / (dist + _EPS)
    # 封顶速度：超过 max_speed 不额外给分
    capped_vel = torch.clamp(proj_vel, min=-max_speed, max=max_speed)
    reward_raw = torch.where(
        capped_vel > 0,
        positive_coeff * capped_vel,
        -negative_coeff * torch.abs(capped_vel),
    )
    within_dist = (dist < max_dist) & (dist > min_dist)
    reward = reward_raw * within_dist.float()
    return torch.where(speed < speed_threshold,
                       torch.zeros_like(reward), reward)


def _ball_distance_reward_impl(
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    scale: float = 1.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """[N] float32：始终激活的球距 exp 核（上游 rewards.py:723-737 同语义）。

    ``scale · exp(−dist/temperature)``，dist = 机器人-球水平 xy 距（上游同
    2D 口径）。填补 approach 项在静止时无信号的缺陷。temperature=2.0（上游
    签名默认）在 3m 域给出核值域 exp(0)=1 … exp(−2)≈0.14，动态范围充分。
    """
    dist = torch.linalg.norm(base_pos[:, :2] - ball_pos[:, :2], dim=-1)
    return scale * torch.exp(-dist / temperature)


def _behind_ball_reward_impl(
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    scale: float = 0.35,
    behind_offset: float = 0.45,
    temperature: float = 1.5,
    attack_sign: float = 1.0,
) -> torch.Tensor:
    """[N] float32：球后位形 exp 核（上游 rewards.py:778-795 同语义）。

    behind_point = 球位沿**己方球门方向**平移 behind_offset 后的目标点；
    reward = scale · exp(−‖behind_point − robot_xy‖ / temperature)（=1 即狗
    恰在球正后方、与进攻方向对准）。上游的 shift 方向由 own_goal_y 符号
    推出；本包进攻方向固定 +y（goal line y=GOAL_Y>0）→ shift 恒 −y，以
    attack_sign 参数化（=+1 进攻 +y，与上游对应分支逐位一致）。
    scale/behind_offset/temperature = 上游签名默认逐值。
    """
    behind_point = ball_pos[:, :2].clone()
    behind_point[:, 1] += -attack_sign * behind_offset
    behind_dist = torch.linalg.norm(behind_point - base_pos[:, :2], dim=-1)
    return scale * torch.exp(-behind_dist / temperature)


def _ball_progress_dense_impl(
    prev_ball_y: torch.Tensor,
    ball_y: torch.Tensor,
    base_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    sign: float = 1.0,
    scale: float = 5.0,
    max_dist: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """([N] float32, [N] cache)：球向门的 y 向位移差分核（上游 rewards.py:
    677-702 同语义，蓝队分支泛化为 sign）。

    ``advance = scale · clamp(sign·Δy, min=0)``（只奖进攻方向位移，不罚后退
    ——同上游单边 clamp）；``reward = advance · (‖robot_xy−ball_xy‖ <
    max_dist)``（近球门控：只在贴身控球时给分，强制精确控球）。cache = 本步
    ball_y（读 prev 后覆写，调用方每步恰一次）。

    缓存口径（lazy NaN 哨兵 = 上游懒初始化同语义）：
      - 首步 NaN → prev 取 cur，差分记 0（懒重播种）；
      - reset 跳变的免疫由 wrapper 侧"每集首步重播种"承接：
        ``episode_length_buf == 1`` → prev 取 cur，差分记 0——spawn 面不
        要求 y=0 为全局最小，上一集球停在线后（y > spawn y）时 reset 跳变
        Δy>0 的正向假 progress（可被策略利用）由重播种封死；impl 保持纯
        函数（重播种逻辑不在本层）。
    """
    fresh = torch.isnan(prev_ball_y)
    prev = torch.where(fresh, ball_y, prev_ball_y)
    delta_y = ball_y - prev
    advance = scale * torch.clamp(sign * delta_y, min=0.0)
    dist = torch.linalg.norm(base_pos[:, :2] - ball_pos[:, :2], dim=-1)
    reward = advance * (dist < max_dist).float()
    return reward, ball_y.clone()


def _kick_event_reward_impl(
    prev_call_step: torch.Tensor,
    last_kick_step: torch.Tensor,
    episode_step: torch.Tensor,
    kick_now: torch.Tensor,
    cooldown_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """([N] float32, [N], [N])：踢击事件回报核（事件化稀疏塑形，检测面复用
    kick_metrics 单源）。

    语义：
      - ``event = kick_now & cooled``：本步足端接触力超阈（kick_now）且
        距上次计费踢击 ≥ cooldown_steps（防连续步重复计费同一次物理踢击
        ——接触持续多步时只记首个超阈步）；
      - 冷却：``episode_step − last_kick_step ≥ cooldown_steps``（NaN seed
        → 永冷却完成）；
      - **reset 跳变免疫（无 reset 钩子）**：``prev_call_step`` 缓存上一次
        奖励调用时的 episode 步计数；episode 步计数（env.episode_length_buf）
        reset 归零 → ``episode_step ≤ prev_call_step`` 判 reset 跳变 →
        last_kick_step 清 NaN（"reset 时清"——新 episode 首踢立即可计，且
        上一 episode 的残留冷却不泄漏）。步计数单调递增面内首调用
        prev_call_step = NaN → 不误判；
      - 回报 1.0 mask（幅值由 RewardTermCfg(weight) 承载）；
      - cache 返回 (new_prev_call_step, new_last_kick_step)，调用方每步
        恰一次 read-modify-write（ball_progress 同款纪律）。
    """
    finite_prev = torch.isfinite(prev_call_step)
    reset_jump = finite_prev & (episode_step <= prev_call_step)
    last = torch.where(reset_jump,
                       torch.full_like(last_kick_step, float("nan")),
                       last_kick_step)
    diff = episode_step - last
    cooled = ~torch.isfinite(last) | (diff >= float(cooldown_steps))
    event = kick_now & cooled
    reward = event.to(dtype=torch.float32)
    new_prev = episode_step.to(dtype=torch.float32)
    new_last = torch.where(event, episode_step.to(dtype=torch.float32), last)
    return reward, new_prev, new_last


def _robot_out_of_field_impl(
    pos_xy_rel: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """[N] bool：机器人中心越出工作区框（上游 terminations.py:27-40 的
    场界矩形判据；上游对称半宽 + margin → 本包任务域非对称框，仅判据面
    参数化差异）。pos_xy_rel 为 env 系（世界系 − env_origins）坐标。"""
    return ((pos_xy_rel[:, 0] < x_min) | (pos_xy_rel[:, 0] > x_max)
            | (pos_xy_rel[:, 1] < y_min) | (pos_xy_rel[:, 1] > y_max))


def _wall_stuck_impl(
    pos_xy_rel: torch.Tensor,
    vel_b_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dist_threshold: float = 0.1,
    speed_threshold: float = 0.3,
) -> torch.Tensor:
    """[N] bool：贴边界低速停滞双条件（上游 terminations.py:78-93 同语义）。

    dist_to_wall = min(到四框边距离)；stuck = (dist_to_wall < dist_threshold)
    & (‖vel_b_xy‖ < speed_threshold)（body 系 xy 速率，同上游局部速度口径）。
    **本包无墙——接口在位备用，cfg 不挂载**（挂载需先引入场地墙段）。
    """
    dist_to_wall = torch.min(
        torch.min(pos_xy_rel[:, 0] - x_min, x_max - pos_xy_rel[:, 0]),
        torch.min(pos_xy_rel[:, 1] - y_min, y_max - pos_xy_rel[:, 1]),
    )
    speed = torch.norm(vel_b_xy, dim=-1)
    return (dist_to_wall < dist_threshold) & (speed < speed_threshold)


# ---------------------------------------------------------------------------
# 球 spawn 随机化 + 距离轴课程（纯函数面，tensor/float in-out 零 isaaclab
# 依赖 → 离线 pytest 直测）
# ---------------------------------------------------------------------------


def ball_spawn_d_max(
    common_step_counter: float,
    warmup_steps: float,
    d_max_start: float = CURRICULUM_D_MAX_START,
    d_max_final: float = 2.7,
) -> float:
    """d_max 课程调度（规格公式）：``d_max(t) = d_max_start + (d_max_final −
    d_max_start) × clamp(t / warmup_steps, 0, 1)``。

    t = env.common_step_counter（全局 env 步计数；reset 事件调用时读实时值）。
    warmup_steps = warmup_iters × steps_per_iter（cfg 单源换算；steps_per_iter
    口径 = num_steps_per_env 24 步/it——common_step_counter 每环境步 +1
    （manager_based_rl_env.py:249），24 步 rollout/it → warmup_iters 处恰
    满域）。课程语义：距离轴随机化本身就是课程——早期球近（近距快反馈），
    随训练单调扩张到全任务域 d_max_final=2.7m。
    """
    frac = common_step_counter / max(warmup_steps, 1e-9)
    frac = min(max(frac, 0.0), 1.0)
    return d_max_start + (d_max_final - d_max_start) * frac


def ball_spawn_dist(dist_u: torch.Tensor, d_min: float, d_max: float) -> torch.Tensor:
    """[N] 球出生距离采样：``dist = d_min + U(0,1) × (d_max − d_min)``。

    d_max < d_min（误配置）时退化为常量 d_min（防呆）。"""
    lo = float(d_min)
    hi = max(float(d_max), lo)
    return lo + dist_u * (hi - lo)


def ball_spawn_xy(
    lateral_u: torch.Tensor,
    dist: torch.Tensor,
    lateral_half: float,
    dog_spawn_x: float,
    dog_spawn_y: float,
    y_max: float,
) -> torch.Tensor:
    """[N,2] 球出生 env 系 xy：狗 spawn 前方沿 +y ``dist`` + 横向 U(±lateral_half)。

    距离轴沿狗朝向（+y 进攻轴）；横向偏移独立采样（复用/扩展现有
    kick_randomization.ball_xy_range 的 x 半幅通道）。y 支路 clamp ≤ y_max
    （= GOAL_Y − CURRICULUM_Y_MARGIN，防 reset 即在门内判 goal 假信号）。"""
    x = dog_spawn_x + (2.0 * lateral_u - 1.0) * float(lateral_half)
    y = (dog_spawn_y + dist).clamp(max=float(y_max))
    return torch.stack([x, y], dim=-1)


# ---------------------------------------------------------------------------
# isaaclab-facing 薄 wrapper（取数 + 调 impl；manager-based term 签名）
# ---------------------------------------------------------------------------


def _per_env_goal_center(env: ManagerBasedRLEnv) -> torch.Tensor:
    """逐 env goal center（世界系）[N,2] = env_origins[:, :2] + GOAL_CENTER。

    网格布局 env_origins 下，goal 相关几何必须逐 env 加 origin——与球 reset
    写入 ``default_root_state + env_origins`` 同帧同源；env_origins=0（单 env
    原点场景）时退化为 GOAL_CENTER 常量，与旧世界系版本逐位一致。
    """
    origins = env.scene.env_origins
    return origins[:, :2] + origins.new_tensor(GOAL_CENTER)


def reset_ball_root_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: Any | None = None,
) -> None:
    """球 root state reset（上游 reset.py:257-280 语义 + spawn 随机化/课程）。

    default_root_state（= scene.ball init_state 单源 BALL_SPAWN_XY，manager
    -based 姿态零旋转、速度零）+ env_origins 偏移 → write_root_pose_to_sim /
    write_root_velocity_to_sim → reset（同步内部 buffer）。

    spawn 随机化：``kick_randomization.enabled``（缺省 True）时球出生位
    **整体覆写**为狗 spawn（DOG_SPAWN_XY）前方沿 +y 的采样位：
      - dist ~ U[d_min, d_max(t)]——距离轴随机化本身就是课程；d_max(t) 由
        :func:`ball_spawn_d_max` 调度（curriculum_enabled=False 时恒
        d_max_final 满域）；``ball_dist_override`` 非 None 时 dist 轴取覆写
        常量（eval-only 球距响应扫描口，跳过采样）；
      - 横向 x ~ dog_x ± ball_xy_range[0]（缺省 ±0.3；
        ball_xy_range 的 y 通道在本机制下不消费——y 轴由 dist 承载）；
      - y clamp ≤ GOAL_Y − CURRICULUM_Y_MARGIN（防 reset 即在门内假 goal）。
    enabled=False 且无 override → 完全固定 spawn（init_state 常量面）。
    狗位姿不在此事件消费（DOG_SPAWN_XY 固定 spawn 由 init_state + reset_base
    pose_range 钳零承载）。

    asset_cfg 形参：isaaclab 面 cfg 传 SceneEntityCfg("ball")（.name 消费）；
    缺省 None → "ball" 直用（本模块离线可 import 纪律——模块级不拉
    isaaclab.managers，与 quat 原语守卫式 import 同款考虑）。

    ball_progress_dense 的 prev_y 缓存由 wrapper 侧"每集首步重播种"
    （episode_length_buf==1 → prev 取 cur，差分记 0）对 reset 跳变免疫，
    随机 spawn 下正是其发挥作用的面，无需 reset 钩子。
    """
    asset_name = getattr(asset_cfg, "name", None) or "ball"
    ball = env.scene[asset_name]
    # 默认 root state（pos+quat+vel，init_state 单源）+ env origin 偏移
    root_state = ball.data.default_root_state.torch.clone()[env_ids]
    origins = env.scene.env_origins[env_ids]
    root_state[:, :3] += origins
    # 随机化 + 课程（override 为 eval-only 球距覆写口，独立于 enabled）
    kr = getattr(env.cfg, "kick_randomization", None)
    if kr is not None and (getattr(kr, "enabled", False)
                           or getattr(kr, "ball_dist_override", None) is not None):
        n = int(len(env_ids))
        dev = root_state.device
        if kr.ball_dist_override is not None:
            dist = torch.full((n,), float(kr.ball_dist_override), device=dev)
        else:
            if getattr(kr, "curriculum_enabled", False):
                warmup_steps = float(kr.warmup_iters) * float(kr.steps_per_iter)
                d_max = ball_spawn_d_max(
                    float(getattr(env, "common_step_counter", 0)), warmup_steps,
                    d_max_final=float(kr.d_max_final))
            else:
                d_max = float(kr.d_max_final)
            dist = ball_spawn_dist(torch.rand(n, device=dev),
                                   float(kr.d_min), d_max)
        xy = ball_spawn_xy(
            torch.rand(n, device=dev), dist,
            lateral_half=float(kr.ball_xy_range[0]),
            dog_spawn_x=float(DOG_SPAWN_XY[0]),
            dog_spawn_y=float(DOG_SPAWN_XY[1]),
            y_max=float(GOAL_Y - CURRICULUM_Y_MARGIN))
        root_state[:, 0] = origins[:, 0] + xy[:, 0]
        root_state[:, 1] = origins[:, 1] + xy[:, 1]
    # 写 sim（上游 reset.py:279-280 同款 manager-based 写法）+ 同步内部 buffer
    ball.write_root_pose_to_sim(root_state[:, :7], env_ids)
    ball.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    ball.reset(env_ids=env_ids)


def ball_rel_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """球相对机体的位置（body 系 xy + 归一距离）[N,3]。

    obs term（KickPolicyCfg 第 43:46 维）。归一尺度 /3.0（3m 任务域）。
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return _ball_rel_b_impl(robot.data.root_pos_w.torch, ball.data.root_pos_w.torch, robot.data.root_quat_w.torch)


def goal_rel_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """球门相对机体的方向（body 系 xy/norm + sin + cos）[N,4]。

    obs term（KickPolicyCfg 第 46:50 维）；goal 逐 env（env_origins +
    GOAL_CENTER——远端 env 方向 obs 必须逐 env goal 帧）。
    """
    robot = env.scene["robot"]
    return _goal_rel_b_impl(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                            goal_center=_per_env_goal_center(env))


def push_geometry_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """推球几何特征（cos_behind + fwd-vs-btog sin/cos）[N,3]。

    obs term（KickPolicyCfg 第 50:53 维）；显式三角特征，减 MLP 学习负担。
    goal 逐 env（同 goal_rel_b）。
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return _push_geometry_b_impl(robot.data.root_pos_w.torch, ball.data.root_pos_w.torch,
                                 robot.data.root_quat_w.torch,
                                 goal_center=_per_env_goal_center(env))


def goal_scored(
    env: ManagerBasedRLEnv,
    half_width: float = GOAL_HALF_WIDTH,
) -> torch.Tensor:
    """终止：球过门线且 |x| < half_width → [N] bool（成功 done）。

    判线逐 env（env_origins + GOAL_CENTER 为门线原点——世界系常量下远端
    env 会永判不中/误触）。
    """
    ball = env.scene["ball"]
    return _goal_scored_impl(ball.data.root_pos_w.torch,
                             goal_center=_per_env_goal_center(env),
                             half_width=half_width)


def ball_out(
    env: ManagerBasedRLEnv,
    radius: float = BALL_OUT_RADIUS,
) -> torch.Tensor:
    """终止：球心水平距 goal center > radius → [N] bool（失败 done）。

    center 逐 env（远端 env 球距必须相对自身 env 系 goal center 判定）。
    """
    ball = env.scene["ball"]
    return _ball_out_impl(ball.data.root_pos_w.torch,
                          center=_per_env_goal_center(env), radius=radius)


def goal_scored_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励：进球事件处置 1.0 [N] float。

    幅值由 RewardTermCfg(weight=10.0) 承载（ball_env_cfg task-only 面）；
    进球即终止（goal_scored DoneTerm），无重复触发问题。判线逐 env
    （与 goal_scored DoneTerm 同帧同判）。
    """
    ball = env.scene["ball"]
    return _goal_scored_impl(ball.data.root_pos_w.torch,
                             goal_center=_per_env_goal_center(env)).to(dtype=torch.float32)


def robot_fallen_penalty(
    env: ManagerBasedRLEnv,
    grav_threshold: float = FALLEN_GRAV_THRESHOLD,
    body_z_threshold: float = FALLEN_BODY_Z_THRESHOLD,
) -> torch.Tensor:
    """奖励：摔倒步 1.0 mask [N] float32（幅值由 RewardTermCfg(weight<0) 承载）。

    状态直检（proj_grav_z / root z），不依赖 termination manager（奖励先于
    终止计算的时序，见 _robot_fallen_penalty_impl）；阈值默认与 robot_fallen
    DoneTerm 同源（FALLEN_* 常量，ball_env_cfg params 显式传同值）。
    """
    robot = env.scene["robot"]
    return _robot_fallen_penalty_impl(robot.data.projected_gravity_b.torch[:, 2],
                                      robot.data.root_pos_w.torch[:, 2],
                                      grav_threshold, body_z_threshold)


def approach_ball_velocity_reward(
    env: ManagerBasedRLEnv,
    max_dist: float = 3.0,
    min_dist: float = 0.0,
    positive_coeff: float = 1.0,
    negative_coeff: float = 2.0,
    speed_threshold: float = 0.0,
    max_speed: float = 0.5,
) -> torch.Tensor:
    """奖励：朝球速度视向投影的方向奖励 [N] float32（上游 rewards.py:267-311）。

    max_dist=3.0 = 本包 3m 任务域全程引导（上游默认 0.5 为其全场近球门控；
    公式逐行同上游，仅任务域默认值适配，见 impl docstring）。
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return _approach_ball_velocity_reward_impl(
        robot.data.root_pos_w.torch, ball.data.root_pos_w.torch,
        robot.data.root_lin_vel_w.torch,
        max_dist, min_dist, positive_coeff, negative_coeff,
        speed_threshold, max_speed)


def ball_distance_reward(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """奖励：球距 exp 核（始终激活）[N] float32（上游 rewards.py:723-737）。"""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return _ball_distance_reward_impl(robot.data.root_pos_w.torch,
                                      ball.data.root_pos_w.torch, scale, temperature)


def behind_ball_reward(
    env: ManagerBasedRLEnv,
    scale: float = 0.35,
    behind_offset: float = 0.45,
    temperature: float = 1.5,
    attack_sign: float = 1.0,
) -> torch.Tensor:
    """奖励：球后位形 exp 核 [N] float32（上游 rewards.py:778-795）。

    进攻方向固定 +y（attack_sign=1 → behind_point 沿 −y 偏移，与上游对应
    分支逐位一致）。
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    return _behind_ball_reward_impl(robot.data.root_pos_w.torch, ball.data.root_pos_w.torch,
                                    scale, behind_offset, temperature, attack_sign)


def ball_progress_dense(
    env: ManagerBasedRLEnv,
    sign: float = 1.0,
    scale: float = 5.0,
    max_dist: float = 1.0,
) -> torch.Tensor:
    """奖励：球向门 y 向位移差分（近球门控）[N] float32（上游 rewards.py:
    677-702；**替换**旧 ball_progress_shaping——其 prev-cur 距离缓存与
    reset 钩子已整体移除）。

    Δy 为 world 系差分（平移不变 → 逐 env origin 无关，无需 goal 帧）；cache
    挂 env._kick_prev_ball_y（lazy NaN 重播种）。**每集首步重播种**（reset
    感知）：``episode_length_buf == 1``（reset 后首步）→ prev 取 cur、差分
    记 0——封死 reset 跳变假 progress（上一集球停线后 → reset 前跳 = 正向
    白给奖励）。episode_length_buf 缺省缺（离线 fake env 无该面）→ 退化为
    NaN 懒播种行为（仍覆盖首步）。
    """
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    ball_y = ball.data.root_pos_w.torch[:, 1]
    prev = getattr(env, _PROGRESS_CACHE_ATTR, None)
    if prev is None:
        prev = torch.full_like(ball_y, float("nan"))
    # 每集首步重播种：reset 后首步 prev 取 cur（差分记 0）
    ep_step = getattr(env, "episode_length_buf", None)
    if ep_step is not None:
        first = ep_step.detach() == 1
        if bool(first.any()):
            prev = torch.where(first, ball_y, prev)
    reward, cache = _ball_progress_dense_impl(prev, ball_y,
                                              robot.data.root_pos_w.torch,
                                              ball.data.root_pos_w.torch,
                                              sign, scale, max_dist)
    # 缓存更新点：读 prev 后覆写为 cur（每步恰一次，同上游 read-modify-write）
    setattr(env, _PROGRESS_CACHE_ATTR, cache)
    return reward.to(dtype=torch.float32)


def kick_event_reward(
    env: ManagerBasedRLEnv,
    force_threshold: float = 0.1,
    cooldown_s: float = 0.5,
) -> torch.Tensor:
    """奖励：踢击事件回报 1.0 mask [N] float32（过程引导项）。

    足端/躯干对球的接触力范数 > force_threshold 判踢——检测函数与阈值常量
    **复用 core/kick_metrics.py 单一实现源**（``KickMetricsTracker
    ._robot_contact_force`` + ``KICK_FORCE_THRESHOLD_N``，与 eval 侧
    kick_count 同函数同阈；force_matrix_w filter 面，net_forces_w 地面反力
    不可用——见该 staticmethod 注释）。签名默认 0.1 = 该常量的镜像值
    （kick_metrics 模块尾交叉断言防漂移——KICK_OBS_DIM mirror 同款）。

    冷却 cooldown_s（秒）换算步数 = round(cooldown_s / env.step_dt)（50Hz
    → 0.5s = 25 步）；事件缓存挂 env._kick_event_last_step（NaN 懒播种 +
    reset 跳变清缓存，无 reset 钩子——缓存面对 reset 跳变免疫，impl
    docstring 口径）。回报幅值由 RewardTermCfg.weight 承载；
    ``KickRewardWeightsCfg.kick_event`` 缺省 0.0 = 零行为影响（红线纪律）。
    """
    # 函数级 import（ball_mdp 离线路径加载零 isaaclab/零 core 包依赖的
    # 模块级纪律；sim 运行期两模块均已初始化，无环）
    from hierarchical_kick.core.kick_metrics import (  # noqa: F401
        KICK_FORCE_THRESHOLD_N,
        KickMetricsTracker,
    )
    force = KickMetricsTracker._robot_contact_force(env).to(dtype=torch.float32)
    kick_now = force > float(force_threshold)
    episode_step = env.episode_length_buf.detach().to(dtype=torch.float32)
    dt = float(getattr(env, "step_dt", 0.02))
    cooldown_steps = max(1, int(round(float(cooldown_s) / dt)))
    cache = getattr(env, _KICK_EVENT_CACHE_ATTR, None)
    if cache is None:
        prev_call = torch.full_like(episode_step, float("nan"))
        last_kick = torch.full_like(episode_step, float("nan"))
    else:
        prev_call, last_kick = cache
    reward, new_prev, new_last = _kick_event_reward_impl(
        prev_call, last_kick, episode_step, kick_now, cooldown_steps)
    # 缓存更新点：每步恰一次 read-modify-write（ball_progress 同款纪律）
    setattr(env, _KICK_EVENT_CACHE_ATTR, (new_prev, new_last))
    return reward


def ball_out_of_bounds_penalty(
    env: ManagerBasedRLEnv,
    radius: float = BALL_OUT_RADIUS,
) -> torch.Tensor:
    """奖励：球出界步 1.0 mask [N] float32（幅值由 RewardTermCfg(weight<0) 承载）。

    判据复用 ball_out 的 _ball_out_impl（逐 env goal 帧半径版）；上游
    rewards.py:461-479 为场界矩形 + 门嘴豁免（|x|<0.75 不判出）+ 速度阈值
    分支——本包单狗无墙无门柱，半径判据保留，差异理由：门嘴豁免在半径判据
    下天然成立（门内距 goal center 必 < 6m），速度阈值分支属"卡墙角"物理，
    无墙场景不适用。
    """
    ball = env.scene["ball"]
    return _ball_out_impl(ball.data.root_pos_w.torch,
                          center=_per_env_goal_center(env),
                          radius=radius).to(dtype=torch.float32)


def robot_fallen(
    env: ManagerBasedRLEnv,
    grav_threshold: float = FALLEN_GRAV_THRESHOLD,
    body_z_threshold: float = FALLEN_BODY_Z_THRESHOLD,
) -> torch.Tensor:
    """终止：机器人倒地（双条件）→ [N] bool（上游 terminations.py:42-59）。

    上游为多机全场 done 面（任意 agent 倒地 → 全场 reset）；本包单狗收敛
    为本体判定，判据逐行同：proj_grav_z > grav_threshold（倾斜超 ~60°）或
    root z < body_z_threshold（go2 站高 0.227 / 阈值 0.16 / 余量 0.067m
    ≈30% 站高——注释论证见 FALLEN_BODY_Z_THRESHOLD）。**替换**继承链的
    root_height(0.15) 单条件挂载（高度支 0.16 严格覆盖 0.15，重力支补倾斜
    未低的高度面）。
    """
    robot = env.scene["robot"]
    return _fallen_mask_impl(robot.data.projected_gravity_b.torch[:, 2],
                             robot.data.root_pos_w.torch[:, 2],
                             grav_threshold, body_z_threshold)


def robot_out_of_field(
    env: ManagerBasedRLEnv,
    x_min: float = ROBOT_FIELD_X_MIN,
    x_max: float = ROBOT_FIELD_X_MAX,
    y_min: float = ROBOT_FIELD_Y_MIN,
    y_max: float = ROBOT_FIELD_Y_MAX,
) -> torch.Tensor:
    """终止：机器人中心越出工作区框 → [N] bool（上游 terminations.py:
    27-40 场界矩形判据的逐 env origin 化）。

    判据以 env 系（root_pos_w − env_origins）计算（网格布局 env_origins 下
    必须逐 env 减 origin，与球 reset/goal 判据同帧同源）；**默认启用**——
    单狗无墙球会滚远，球逃逸由 ball_out 处置，机器人不该追出工作区。
    """
    robot = env.scene["robot"]
    pos_rel = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return _robot_out_of_field_impl(pos_rel, x_min, x_max, y_min, y_max)


def wall_stuck(
    env: ManagerBasedRLEnv,
    x_min: float = ROBOT_FIELD_X_MIN,
    x_max: float = ROBOT_FIELD_X_MAX,
    y_min: float = ROBOT_FIELD_Y_MIN,
    y_max: float = ROBOT_FIELD_Y_MAX,
    dist_threshold: float = 0.1,
    speed_threshold: float = 0.3,
) -> torch.Tensor:
    """终止：贴边界低速停滞 → [N] bool（上游 terminations.py:78-93）。

    **接口在位、cfg 不挂载**（本包工作区无墙——上游 wall_stuck 服务于其
    场界墙的卡墙角清理；待引入场地墙段后再挂载，dist+speed 双条件公式已
    就绪）。
    """
    robot = env.scene["robot"]
    pos_rel = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return _wall_stuck_impl(pos_rel, robot.data.root_lin_vel_b.torch[:, :2],
                            x_min, x_max, y_min, y_max,
                            dist_threshold, speed_threshold)


__all__ = [
    # 常量（名称契约冻结）
    "GOAL_Y",
    "GOAL_HALF_WIDTH",
    "GOAL_CENTER",
    "DOG_SPAWN_XY",
    "CURRICULUM_D_MAX_START",
    "CURRICULUM_Y_MARGIN",
    "BALL_OUT_RADIUS",
    "KICK_OBS_DIM",
    "BALL_RADIUS",
    "BALL_MASS",
    "FALLEN_GRAV_THRESHOLD",
    "FALLEN_BODY_Z_THRESHOLD",
    "ROBOT_FIELD_X_MIN",
    "ROBOT_FIELD_X_MAX",
    "ROBOT_FIELD_Y_MIN",
    "ROBOT_FIELD_Y_MAX",
    # obs funcs
    "ball_rel_b",
    "goal_rel_b",
    "push_geometry_b",
    # term funcs（robot_fallen 替换 root_height 挂载；wall_stuck 接口在位、
    # cfg 不挂载）
    "goal_scored",
    "ball_out",
    "robot_fallen",
    "robot_out_of_field",
    "wall_stuck",
    # reward funcs（survival = mdp.alive_reward）
    "goal_scored_reward",
    "robot_fallen_penalty",
    "approach_ball_velocity_reward",
    "ball_distance_reward",
    "behind_ball_reward",
    "ball_progress_dense",
    "kick_event_reward",
    "ball_out_of_bounds_penalty",
    # spawn 随机化 + 距离轴课程（reset 事件 func + 纯函数调度面）
    "reset_ball_root_state",
    "ball_spawn_d_max",
    "ball_spawn_dist",
    "ball_spawn_xy",
]
