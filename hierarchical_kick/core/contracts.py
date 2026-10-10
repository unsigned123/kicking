"""上层通道契约常量（适配表单一真相源）。

上层 obs 48 维 = decoder proprio 42（首 term，切片契约 ``[..., :42]``）
⊕ base_lin_vel 3 ⊕ velocity_commands 3；动作 12 维（pre-scale 关节偏移，
语义 scale 由 env 侧逐关节施加）。本模块零 isaaclab（torch 纯常量），
sim 面 ObsTerm 参数与离线单测共同消费本表。

⚠️ 官方 velocity policy 组自己的项参数（joint_vel U±1.5×scale0.01、
base_ang_vel scale1）**不得照抄**——以本表（decoder 契约）为准。
"""
from __future__ import annotations

import torch

# -- 维度契约 ---------------------------------------------------------------
Z_DIM = 32
"""z 段潜空间维（st-vae 段 latent，d_z=32；段 latent 动作面）。"""

LEGACY_Z_DIM = 64
"""旧线 z 维（decoder 106 = 42+64 面装载体消费；保留常量供旧线
装载体缺省面）。"""

SEG_RATE_HZ = 12.5
"""上层段发射率（段时钟 50Hz ÷ 4 = 12.5Hz）。"""

FRAMES_PER_SEG = 4
"""每段帧数（50Hz ÷ 12.5Hz = 4，段 s 覆盖帧 [4s, 4s+4)）。"""

SEG_LEN_S = 0.08
"""段时长（0.08s；f4 时钟 4 帧 @50Hz）。"""

SIGMA_INIT_DEFAULT = 0.5
"""段空间探索 σ init（绝对 z 单位；对应 st-vae decoder 训练的 z 扰动
课程起点 sigma.start=0.5，decoder 对该量级扰动已验证不变性）。"""

SIGMA_CAP_DEFAULT = 0.8
"""段空间探索 σ cap（绝对 z 单位；PPO update 后逐维 clamp 上界）。

依据：z 扰动不变性训练区上沿（stvae_decoder.yaml sigma.end=0.8）——
超出该量的探索步长落在 decoder 未训练的扰动区，且无界 std 会在 PPO
更新中经熵棘轮持续增长导致性能退化；机制参数化（sigma_cap=None 关闭），
缺省开帽。"""

PROPRIO_DIM = 42
"""decoder proprio 帧（单帧，无历史）：joint_pos+joint_vel+base_ang_vel+
projected_gravity+last_action。"""

BASE_LIN_VEL_DIM = 3
CMD_DIM = 3

OBS_DIM = PROPRIO_DIM + BASE_LIN_VEL_DIM + CMD_DIM
"""上层策略 obs 48。"""

ACTION_DIM = 12
"""解码动作维（12 关节 pre-scale 偏移）= env.num_actions。"""

# -- 42 维内偏移（§1.1 表列号） ----------------------------------------------
OFF_JOINT_POS = 0        # (0:12)
OFF_JOINT_VEL = 12       # (12:24)
OFF_BASE_ANG_VEL = 24    # (24:27)
OFF_PROJ_GRAVITY = 27    # (27:30)
OFF_LAST_ACTION = 30     # (30:42)

# -- 逐通道噪声/缩放（isaaclab ObsTerm 原生参数） ---------------------------
NOISE_JOINT_POS = 0.01        # U(±0.01)，scale 1.0
NOISE_JOINT_VEL = 0.5         # U(±0.5)；⚠️ 官方 U±1.5 不得照抄
SCALE_JOINT_VEL = 0.05        # ⚠️ 官方 0.01 不得照抄
NOISE_BASE_ANG_VEL = 0.2      # U(±0.2)
SCALE_BASE_ANG_VEL = 0.2      # ⚠️ 官方 scale=1 不得照抄
NOISE_PROJ_GRAVITY = 0.05     # U(±0.05)，scale 1.0
NOISE_BASE_LIN_VEL = 0.1      # U(±0.1)（策略面；官方同参）
# last_action / velocity_commands 无噪声

# -- 探索（§1.4） -------------------------------------------------------------
SIGMA_SEM = 0.2
"""语义空间探索 σ；pre-space init = σ_sem ⊘ scale_vec
（abad/hip ≈1.3503 / knee ≈0.7044）。std_param 逐维覆盖在 UpperActor 构造期。"""

# -- 官方 velocity-flat 口径（上游对齐；低层 official 分支） -----------------
# 上游 isaac-sim/IsaacLab main：policy obs 48 维单帧（顺序保留、无历史栈、
# 无 height_scan），joint_vel 无 scale（raw rad/s）、base_ang_vel 无 scale、
# 动作 JointPositionActionCfg(scale=0.5, use_default_offset=True) 均匀 0.5。
OFFICIAL_ACTION_SCALE = 0.5
"""官方动作语义 scale（均匀标量；default + 0.5⊙action，替换逐关节 scale_vec）。"""

OFFICIAL_OBS_LIN_VEL = 0        # [0:3)   base_lin_vel（raw）
OFFICIAL_OBS_ANG_VEL = 3        # [3:6)   base_ang_vel（raw，scale 1）
OFFICIAL_OBS_GRAVITY = 6        # [6:9)   projected_gravity
OFFICIAL_OBS_CMD = 9            # [9:12)  velocity_commands
OFFICIAL_OBS_JOINT_POS = 12     # [12:24) joint_pos_rel（scale 1）
OFFICIAL_OBS_JOINT_VEL = 24     # [24:36) joint_vel（raw，scale 1）
OFFICIAL_OBS_LAST_ACTION = 36   # [36:48) last_action（pre-scale）

SCALE_COEF = 0.25
"""scale_j = SCALE_COEF × effort_j / KP（config/robots/go2.yaml actions
语义；KP=actuator.stiffness=40）。"""

KP = 40.0
"""PD 刚度（implicit actuator；go2.yaml actuator.stiffness）。"""


def proprio_slice(obs: torch.Tensor) -> torch.Tensor:
    """48 维策略 obs → 前 42 维 decoder proprio（首 term 切片契约）。"""
    return obs[..., :PROPRIO_DIM]


__all__ = [
    "Z_DIM", "LEGACY_Z_DIM", "SEG_RATE_HZ", "FRAMES_PER_SEG", "SEG_LEN_S",
    "SIGMA_INIT_DEFAULT", "SIGMA_CAP_DEFAULT",
    "PROPRIO_DIM", "BASE_LIN_VEL_DIM", "CMD_DIM", "OBS_DIM",
    "ACTION_DIM",
    "OFF_JOINT_POS", "OFF_JOINT_VEL", "OFF_BASE_ANG_VEL",
    "OFF_PROJ_GRAVITY", "OFF_LAST_ACTION",
    "NOISE_JOINT_POS", "NOISE_JOINT_VEL", "SCALE_JOINT_VEL",
    "NOISE_BASE_ANG_VEL", "SCALE_BASE_ANG_VEL", "NOISE_PROJ_GRAVITY",
    "NOISE_BASE_LIN_VEL",
    "SIGMA_SEM", "SCALE_COEF", "KP",
    "OFFICIAL_ACTION_SCALE",
    "OFFICIAL_OBS_LIN_VEL", "OFFICIAL_OBS_ANG_VEL", "OFFICIAL_OBS_GRAVITY",
    "OFFICIAL_OBS_CMD", "OFFICIAL_OBS_JOINT_POS", "OFFICIAL_OBS_JOINT_VEL",
    "OFFICIAL_OBS_LAST_ACTION",
    "proprio_slice",
]
