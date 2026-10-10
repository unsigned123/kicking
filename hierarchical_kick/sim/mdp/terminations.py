# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.mdp.terminations
#
# 官方 flat 终止面 = time_out + base_contact，无 height 项；确需的差异以
# 本包新 func 显式表达：
#   root_height(threshold=0.15) — 对齐 tracking fall_height 0.15 语义
#   （config/robots/go2.yaml fall_height: 0.15；站高 0.227，余量 0.077；
#   0.2 阈值会误杀步态质心波谷，故 0.15）。
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
    本终止即摔死的落地面——低质量 z 触发的瘫倒不挂正流
    （死时无 track/survival 流）。
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2] < threshold


__all__ = ["root_height"]
