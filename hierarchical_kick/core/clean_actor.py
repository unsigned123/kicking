# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.core.clean_actor
#
# [cmd-only 拆分裁剪] 本文件只保留 kick 面 obs 维契约常量（下游消费：
# cmd_actor 守卫 4 / agents.UpperCmdActorCfg.obs_dim_contract）。原
# UpperCleanActor 类（旧「hier」臂：vanilla tanh z → V3 clean decoder）
# 依赖冻结 decoder 栈，属 cmd-only 排除面，已随拆分移除——复现位留在
# UT 旧分支。
#
# KICK_POLICY_OBS_DIM=55 与 kick_metrics.KICK_POLICY_OBS_DIM mirror
# 交叉断言，见 kick_metrics 尾部；契约通道构成 = PROPRIO_DIM(42) +
# base_lin_vel 3 + ball_rel_b 3 + goal_rel_b 4 + push_geometry_b 3。
# ---------------------------------------------------------------------------
"""Kick-face obs-dimension contract constants (cmd-only split)."""
from __future__ import annotations

from .contracts import BASE_LIN_VEL_DIM, PROPRIO_DIM

KICK_LIN_VEL_DIM = BASE_LIN_VEL_DIM  # 3
KICK_BALL_REL_DIM = 3
KICK_GOAL_REL_DIM = 4
KICK_PUSH_GEOM_DIM = 3

KICK_POLICY_OBS_DIM = (PROPRIO_DIM + KICK_LIN_VEL_DIM + KICK_BALL_REL_DIM
                       + KICK_GOAL_REL_DIM + KICK_PUSH_GEOM_DIM)
"""kick 面 policy obs 总维 = 42 + 3 + 3 + 4 + 3 = 55（mirror
ball_mdp.KICK_OBS_DIM；kick_metrics import 期与之交叉断言防漂移）。"""


__all__ = ["KICK_POLICY_OBS_DIM", "KICK_LIN_VEL_DIM", "KICK_BALL_REL_DIM",
           "KICK_GOAL_REL_DIM", "KICK_PUSH_GEOM_DIM"]
