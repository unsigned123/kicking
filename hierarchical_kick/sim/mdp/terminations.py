# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.mdp.terminations
#
# U0_UPPER_DESIGN §3 行「height 终止 0.2→0.15 教训」：官方 flat 终止面 =
# time_out + base_contact（§0.2），无 height 项；dog fork 的 owner 私改
# height 终止不可作 upstream（§0.2/§3），确需的差异以本包新 func 显式表达：
#   root_height(threshold=0.15) — 对齐 tracking fall_height 0.15 语义
#   （config/robots/go2.yaml fall_height: 0.15；站高 0.227，余量 0.077）。
# dog 实测 0.2 阈值误杀步态（367 次），故 0.15。
# ---------------------------------------------------------------------------

"""Termination functions for the upper velocity-flat environment.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg`
object to enable the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate when the root link height falls below a threshold.

    对齐 tracking 线 fall 终止的高度语义（root z < 0.15 m，站高 0.227）；
    上层快死在经济学审计里被结构性排除（§2.3），本终止是快死的落地面——
    低质量 z 触发的瘫倒不挂正流（死时无 track/β2/β1 流）。
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2] < threshold


__all__ = ["root_height"]
