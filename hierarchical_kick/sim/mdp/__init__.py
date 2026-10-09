# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# VENDORED-STRUCTURE — hierarchical_kick.sim.mdp（包聚合 __init__）
#
# 上游模板：isaac-sim/IsaacLab @ v2.3.0
#      （commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77）
# 原文：isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py
#
# 与上游 __init__ 的 diff（逐处记录）：
#   1) 核心 mdp 源同构（``from isaaclab.envs.mdp import *``）——官方同款；
#   2) 不 re-export ``.curriculums``？——否：保留（terrain_levels_vel，verbatim）；
#   3) 不 re-export ``.terminations``（官方 re-export terrain_out_of_bounds）
#      ——flat/plane 不需要，且本包 terminations.py 是 upper 自建
#      root_height（非上游），显式 re-export 以便 cfg 引用；
#   4) ``.rewards`` re-export 面 = 上游两函数（feet_air_time/feet_slide
#      verbatim）+ upper 自建 alive_reward（上游无此函数）。
#   5) ``.ball_mdp`` re-export（UPPER 自建追加，U-Kick 球 env：
#      docs/UKICK_BALL_ENV_DESIGN.md §8-WP2 名称契约——obs 三函数
#      ball_rel_b/goal_rel_b/push_geometry_b、终止五函数 goal_scored/
#      ball_out/robot_fallen/robot_out_of_field/wall_stuck（接口备用）、
#      奖励七函数（2026-09-30 phase0 规则接口对齐 + survival 走本包
#      alive_reward）、任务几何常量；donor = soccer_absolute，
#      provenance 见该文件头）。
# ---------------------------------------------------------------------------

"""This sub-module contains the functions that are specific to the upper velocity-flat environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .ball_mdp import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
