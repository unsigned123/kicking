# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# VENDORED — hierarchical_kick.sim.mdp.curriculums
#
# 上游：isaac-sim/IsaacLab @ v2.3.0
#      （commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77）
# 原文：isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
#
# vendor 理由：vendored velocity_env_cfg.py 的 CurriculumCfg 在 import 期引用
# ``mdp.terrain_levels_vel``（类体求值），不 vendor 该符号则 base cfg 无法
# 导入（A100 stripped fork 无 locomotion 包可回退）。函数 verbatim 零 diff。
# 上层 flat 环境运行期不用地形课程（flat_env_cfg 置 terrain_levels=None）。
#
# 非 vendor：velocity mdp/terminations.py 的 terrain_out_of_bounds（flat/plane
# 不需要）——velocity 包 vendor 清单 = rewards(2) + curriculums(1)，见本包
# __init__ 头注与 DEVIATIONS.md U1 段。
# ---------------------------------------------------------------------------

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
