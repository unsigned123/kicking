# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.mdp.frozen_loco_math
#
# U-Kick arm D「真分层」的冻结低层数学面（纯 torch，零 isaaclab——离线
# 单测可按路径加载直测，ball_mdp 同款拆分纪律）。isaaclab-facing 的
# ActionTerm 薄壳在 :mod:`frozen_loco_action`（本目录同级）。
#
# 【官方口径 2026-10-09】缺省低层约定 = 上游官方 velocity-flat（obs 48 维
# 单帧：linvel→ang_vel→gravity→cmd→q→qd→last_action，joint_vel/base_ang_vel
# raw 无缩放，动作 default + 0.5⊙action 均匀 0.5）——低层 ckpt 由
# HK-Velocity-Flat-Unitree-Go2-v0 训练面产出（sim/velocity_official_cfg.py）。
# 旧 U4b B 臂契约（proprio42 前缀 + scale_vec 逐关节）降级为 legacy 分支：
#   - assemble_low_level_obs          → 官方布局（缺省）；
#   - assemble_low_level_obs_legacy   → 旧 48 布局（proprio42 + linvel + cmd；
#     joint_vel ×0.05 / ang_vel ×0.2 适配表参数）；
#   - assemble_proprio42              → 旧 42 前缀（legacy 专用，不再进缺省路径）。
# 本模块把组装公式做成纯函数（偏移常量复用 core/contracts.py 单一真相源，
# **禁止**复制参数），供：
#   - frozen_loco_action.apply_actions 的逐步组装（sim 路径）；
#   - 离线 fake tensor 单测（对 contracts 偏移常量逐位断言）。
#
# ⚠️ 推理面：组装**不加噪声**（Unoise 只在训练 obs 管线由 isaaclab 施加；
# 冻结低层以真值 obs 跑 deterministic mean——play 探针/oracle 链同款口径）。
# ---------------------------------------------------------------------------

"""Frozen low-level observation assembly math (pure torch; zero isaaclab)."""

from __future__ import annotations

import torch

from hierarchical_kick.core.contracts import (
    ACTION_DIM,
    BASE_LIN_VEL_DIM,
    CMD_DIM,
    OFF_BASE_ANG_VEL,
    OFF_JOINT_POS,
    OFF_JOINT_VEL,
    OFF_LAST_ACTION,
    OFF_PROJ_GRAVITY,
    OFFICIAL_OBS_ANG_VEL,
    OFFICIAL_OBS_CMD,
    OFFICIAL_OBS_GRAVITY,
    OFFICIAL_OBS_JOINT_POS,
    OFFICIAL_OBS_JOINT_VEL,
    OFFICIAL_OBS_LAST_ACTION,
    OFFICIAL_OBS_LIN_VEL,
    PROPRIO_DIM,
    SCALE_BASE_ANG_VEL,
    SCALE_JOINT_VEL,
)

LOW_LEVEL_OBS_DIM = PROPRIO_DIM + BASE_LIN_VEL_DIM + CMD_DIM
"""冻结低层 obs 契约 48（两套布局同维——布局区分靠约定分支，模块尾部交叉断言）。"""

# 每关节维（go2 双前肢 4×3；contracts.ACTION_DIM=12 同源）
_JOINT_PER = ACTION_DIM // 4


# --------------------------------------------------------------------------- #
# 官方口径（缺省）——velocity-flat 48 维单帧
# --------------------------------------------------------------------------- #
def assemble_low_level_obs(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    last_action: torch.Tensor,
    base_lin_vel: torch.Tensor,
    command: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """组装冻结低层 48 维 obs（**官方 velocity-flat 口径**；纯函数——单测对象）。

    通道序 = 上游官方 velocity policy obs 顺序（偏移常量 = contracts.py
    OFFICIAL_OBS_* 单一真相源）::

        [0:3)    base_lin_vel（机体系；raw）
        [3:6)    base_ang_vel（机体系；raw，官方无 scale）
        [6:9)    projected_gravity（scale 1）
        [9:12)   velocity_commands = 刚写入的 [vx,vy,wz]（无噪声）
        [12:24)  joint_pos_rel = joint_pos − default_joint_pos（scale 1）
        [24:36)  joint_vel（**raw rad/s，官方无 scale**）
        [36:48)  last_action = 低层自己上一步 pre-scale 输出（无噪声）

    与 legacy 布局的差异：通道顺序整体重排（linvel 前置）+ joint_vel/ang_vel
    不缩放。推理口径：无噪声（Unoise 训练面才施加）；全部输入为 env 真值
    buffer。单帧拼接（官方 policy 组 history_length 缺省 None → 单帧）。

    参数形状（N = num_envs）：joint_pos/default_joint_pos/joint_vel/
    last_action (N,12)；base_ang_vel/projected_gravity/base_lin_vel/command
    (N,3)。``out`` 给定时就地写入（ActionTerm 复用 buffer 零分配）。
    """
    n = joint_pos.shape[0]
    if out is None:
        out = torch.zeros(n, LOW_LEVEL_OBS_DIM, device=joint_pos.device,
                          dtype=joint_pos.dtype)
    for name, t, shape in (
        ("joint_pos", joint_pos, (n, ACTION_DIM)),
        ("default_joint_pos", default_joint_pos, (n, ACTION_DIM)),
        ("joint_vel", joint_vel, (n, ACTION_DIM)),
        ("last_action", last_action, (n, ACTION_DIM)),
        ("base_ang_vel", base_ang_vel, (n, 3)),
        ("projected_gravity", projected_gravity, (n, 3)),
        ("base_lin_vel", base_lin_vel, (n, 3)),
        ("command", command, (n, CMD_DIM)),
    ):
        if tuple(t.shape) != shape:
            raise ValueError(
                f"[upper][armD] 低层 obs 组装形状不符: {name} "
                f"{tuple(t.shape)} != {shape}")
    # [0:3) base_lin_vel / [3:6) base_ang_vel / [6:9) projected_gravity
    out[:, OFFICIAL_OBS_LIN_VEL:OFFICIAL_OBS_ANG_VEL] = base_lin_vel
    out[:, OFFICIAL_OBS_ANG_VEL:OFFICIAL_OBS_GRAVITY] = base_ang_vel
    out[:, OFFICIAL_OBS_GRAVITY:OFFICIAL_OBS_CMD] = projected_gravity
    # [9:12) velocity_commands
    out[:, OFFICIAL_OBS_CMD:OFFICIAL_OBS_JOINT_POS] = command
    # [12:24) joint_pos_rel / [24:36) joint_vel raw
    out[:, OFFICIAL_OBS_JOINT_POS:OFFICIAL_OBS_JOINT_VEL] = (
        joint_pos - default_joint_pos)
    out[:, OFFICIAL_OBS_JOINT_VEL:OFFICIAL_OBS_LAST_ACTION] = joint_vel
    # [36:48) last_action
    out[:, OFFICIAL_OBS_LAST_ACTION:
        OFFICIAL_OBS_LAST_ACTION + ACTION_DIM] = last_action
    return out


# --------------------------------------------------------------------------- #
# legacy 分支——U4b B 臂旧 48 布局（proprio42 前缀 + linvel + cmd）
# --------------------------------------------------------------------------- #
def assemble_proprio42(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    last_action: torch.Tensor,
    *,
    joint_vel_scale: float = SCALE_JOINT_VEL,
    ang_vel_scale: float = SCALE_BASE_ANG_VEL,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """组装 decoder proprio 42 维前缀（contracts §1.1 适配表；**legacy 专用**）。

    ⚠️ legacy：旧 U4b 契约（joint_vel ×0.05 / ang_vel ×0.2）；官方口径低层
    不走本函数。通道序（= 旧 48 维低层 obs 的前 42 通道，单一实现源）::

        [0:12)   joint_pos_rel = joint_pos − default_joint_pos   （scale 1）
        [12:24)  joint_vel × joint_vel_scale（0.05）
        [24:27)  base_ang_vel（机体系）× ang_vel_scale（0.2）
        [27:30)  projected_gravity（scale 1）
        [30:42)  last_action = 上一步 pre-scale 输出（无噪声）

    推理口径：无噪声（Unoise 训练面才施加）。``out`` 给定时就地写入
    （ActionTerm 复用 buffer 零分配）；返回 42 维张量。
    """
    n = joint_pos.shape[0]
    if out is None:
        out = torch.zeros(n, PROPRIO_DIM, device=joint_pos.device,
                          dtype=joint_pos.dtype)
    for name, t, shape in (
        ("joint_pos", joint_pos, (n, ACTION_DIM)),
        ("default_joint_pos", default_joint_pos, (n, ACTION_DIM)),
        ("joint_vel", joint_vel, (n, ACTION_DIM)),
        ("last_action", last_action, (n, ACTION_DIM)),
        ("base_ang_vel", base_ang_vel, (n, 3)),
        ("projected_gravity", projected_gravity, (n, 3)),
    ):
        if tuple(t.shape) != shape:
            raise ValueError(
                f"[upper] proprio42 组装形状不符: {name} "
                f"{tuple(t.shape)} != {shape}")
    out[:, OFF_JOINT_POS:OFF_JOINT_VEL] = joint_pos - default_joint_pos
    out[:, OFF_JOINT_VEL:OFF_BASE_ANG_VEL] = joint_vel * float(joint_vel_scale)
    out[:, OFF_BASE_ANG_VEL:OFF_PROJ_GRAVITY] = base_ang_vel * float(ang_vel_scale)
    out[:, OFF_PROJ_GRAVITY:OFF_LAST_ACTION] = projected_gravity
    out[:, OFF_LAST_ACTION:OFF_LAST_ACTION + ACTION_DIM] = last_action
    return out


def assemble_low_level_obs_legacy(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    last_action: torch.Tensor,
    base_lin_vel: torch.Tensor,
    command: torch.Tensor,
    *,
    joint_vel_scale: float = SCALE_JOINT_VEL,
    ang_vel_scale: float = SCALE_BASE_ANG_VEL,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """组装冻结低层 48 维 obs（**legacy U4b B 臂契约**；纯函数——单测对象）。

    ⚠️ legacy 分支（--low_level_convention legacy 才消费；缺省官方布局走
    :func:`assemble_low_level_obs`）。通道序 = 旧 contracts.py §1.1 适配表
    （偏移常量驱动）::

        [0:12)   joint_pos_rel = joint_pos − default_joint_pos   （scale 1）
        [12:24)  joint_vel × joint_vel_scale（0.05）
        [24:27)  base_ang_vel（机体系）× ang_vel_scale（0.2）
        [27:30)  projected_gravity（scale 1）
        [30:42)  last_action = 低层自己上一步 pre-scale 输出（无噪声）
        [42:45)  base_lin_vel（机体系；无缩放，与训练面同参）
        [45:48)  command = 刚写入的 [vx,vy,wz]（无噪声）

    推理口径：无噪声（Unoise 训练面才施加）；全部输入为 env 真值 buffer。

    参数形状（N = num_envs）：joint_pos/default_joint_pos/joint_vel/
    last_action (N,12)；base_ang_vel/projected_gravity/base_lin_vel/command
    (N,3)。``out`` 给定时就地写入（ActionTerm 复用 buffer 零分配）。
    """
    n = joint_pos.shape[0]
    if out is None:
        out = torch.zeros(n, LOW_LEVEL_OBS_DIM, device=joint_pos.device,
                          dtype=joint_pos.dtype)
    # [0:42) proprio 前缀 = assemble_proprio42 单源（就地写入 out 切片——
    # 通道公式零复制；形状校验在子函数内）
    assemble_proprio42(
        joint_pos, default_joint_pos, joint_vel, base_ang_vel,
        projected_gravity, last_action, joint_vel_scale=joint_vel_scale,
        ang_vel_scale=ang_vel_scale, out=out[:, :PROPRIO_DIM])
    # 形状 fail-fast（48 维扩展段；列号即错误消息指路）
    for name, t, shape in (
        ("base_lin_vel", base_lin_vel, (n, 3)),
        ("command", command, (n, CMD_DIM)),
    ):
        if tuple(t.shape) != shape:
            raise ValueError(
                f"[upper][armD] 低层 obs 组装形状不符: {name} "
                f"{tuple(t.shape)} != {shape}")
    # [42:45) base_lin_vel
    out[:, OFF_LAST_ACTION + ACTION_DIM:
        OFF_LAST_ACTION + ACTION_DIM + BASE_LIN_VEL_DIM] = base_lin_vel
    # [45:48) command
    out[:, LOW_LEVEL_OBS_DIM - CMD_DIM:] = command
    return out


def joint_target_from_low_action(
    default_joint_pos: torch.Tensor,
    scale_vec: torch.Tensor,
    low_action: torch.Tensor,
) -> torch.Tensor:
    """关节目标 = default_joint_pos + scale_vec ⊙ 低层 pre-scale 输出（纯函数）。

    = JointPositionAction 的 env 侧处理语义（raw × scale + offset，
    use_default_offset=True → offset = default_joint_pos）。scale_vec 按约定
    分派：official = 均匀 0.5（contracts.OFFICIAL_ACTION_SCALE）、legacy =
    core/scale_vec 单源逐关节表。返回**新张量**（不改输入 buffer）。
    """
    if low_action.shape != default_joint_pos.shape or \
            low_action.shape != scale_vec.shape:
        raise ValueError(
            f"[upper][armD] 关节目标公式形状不符: low_action "
            f"{tuple(low_action.shape)} default {tuple(default_joint_pos.shape)}"
            f" scale {tuple(scale_vec.shape)}")
    return default_joint_pos + scale_vec * low_action


if LOW_LEVEL_OBS_DIM != PROPRIO_DIM + BASE_LIN_VEL_DIM + CMD_DIM:
    raise ValueError("[upper][armD] 低层 obs 维契约漂移（contracts 单源）")

__all__ = [
    "LOW_LEVEL_OBS_DIM",
    "assemble_low_level_obs",
    "assemble_low_level_obs_legacy",
    "assemble_proprio42",
    "joint_target_from_low_action",
]
