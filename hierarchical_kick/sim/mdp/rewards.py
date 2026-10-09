# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# VENDORED — hierarchical_kick.sim.mdp.rewards
#
# 上游：isaac-sim/IsaacLab @ v2.3.0
#      （commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77）
# 原文：isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
#
# vendor 范围（U0_UPPER_DESIGN §0.2 移植裁定：velocity 包非核心 mdp 函数
# 仅两个）：
#   - feet_air_time   — 官方原文逐字符 verbatim，零 diff；
#   - feet_slide      — 官方原文逐字符 verbatim，零 diff。
# 其余 velocity 包函数（feet_air_time_positive_biped /
# track_lin_vel_xy_yaw_frame_exp / track_ang_vel_z_world_exp /
# stand_still_joint_deviation_l1）不 vendor（上层不用；track_* 与
# stand_still 在 isaaclab.envs.mdp 核心/dog 私改面）。
#
# UPPER 自建（非 vendor，与本上游无关）：
#   - alive_reward    — §1.5 survival 项 RewardTermCfg(func=..., weight=β1)
#                       的恒 +1.0 func（V4 latent_survival 同形态）。
# ---------------------------------------------------------------------------

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— U0_UPPER_DESIGN §1.5 survival 项
# ---------------------------------------------------------------------------
def alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """每步恒 +1.0（存活奖励 func；权值 β1 由 RewardTermCfg.weight 施加）。

    U0 §1.5 ``survival 项 = RewardTermCfg(func=<恒 +1.0>, weight=β1)``；
    V4 latent_survival 同形态。经济学审计（§2）把 β1 当保险丝
    （快死结构性排除后站桩-跟踪不等式才是主导项）。
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


__all__ = ["feet_air_time", "feet_slide", "alive_reward"]
