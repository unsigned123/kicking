"""策略闭环纯逻辑层（无 ROS2 依赖，可离线单测）。

持有全部输入缓冲；``tick()`` 做新鲜度检查 + obs 构造 + 策略推理，返回
(命令或 None, 阻止原因)。ROS2 粘合层（node.py / skill.py）只负责收发。
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from . import joint_spec
from .obs_builder import GOAL_CENTER_XY, build_obs
from .policy import PolicyRuntime

BAD_STATUS = {2, 3, 4}  # ESTOP / EDAMP / LIFTED（出厂运控 switch_status）


class KickController:
    def __init__(
        self,
        policy_ckpt: str,
        *,
        goal_xy: Sequence[float] = GOAL_CENTER_XY,
        vel_limit: Sequence[float] = (1.6, 0.55, 2.5),
        joint_timeout_s: float = 0.10,
        odom_timeout_s: float = 0.25,
        ball_timeout_s: float = 0.30,
        last_action_source: str = "joint_deviation",
    ) -> None:
        self.policy = PolicyRuntime(policy_ckpt)
        self.goal_xy = tuple(float(v) for v in goal_xy)
        self.vel_limit = np.asarray(vel_limit, dtype=np.float64)
        self.joint_timeout_s = float(joint_timeout_s)
        self.odom_timeout_s = float(odom_timeout_s)
        self.ball_timeout_s = float(ball_timeout_s)
        self.last_action_source = last_action_source
        self.default_q = np.asarray(joint_spec.TRAIN_DEFAULT_Q, dtype=np.float64)

        self.q: Optional[np.ndarray] = None
        self.qd: Optional[np.ndarray] = None
        self.quat = None
        self.gyro = None
        self.robot_xy = None
        self.base_lin_vel = None
        self.ball_xy = None
        self.t_joint = 0.0
        self.t_odom = 0.0
        self.t_ball = 0.0
        self.status: Optional[int] = None
        self.estop = False

    # ---- 输入 ----------------------------------------------------------- #
    def set_joints(self, q_train: Sequence[float], qd_train: Sequence[float]) -> None:
        self.q = np.asarray(q_train, dtype=np.float64)
        self.qd = np.asarray(qd_train, dtype=np.float64)
        self.t_joint = time.time()

    def set_imu(self, quat_xyzw: Sequence[float], gyro: Sequence[float]) -> None:
        self.quat = np.asarray(quat_xyzw, dtype=np.float64)
        self.gyro = np.asarray(gyro, dtype=np.float64)

    def set_odom(self, robot_xy: Sequence[float],
                 base_lin_vel: Sequence[float],
                 gyro: Optional[Sequence[float]] = None,
                 quat_xyzw: Optional[Sequence[float]] = None) -> None:
        self.robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]
        self.base_lin_vel = np.asarray(base_lin_vel, dtype=np.float64)
        if gyro is not None:
            self.gyro = np.asarray(gyro, dtype=np.float64)
        if quat_xyzw is not None:
            self.quat = np.asarray(quat_xyzw, dtype=np.float64)
        self.t_odom = time.time()

    def set_ball(self, ball_xy: Sequence[float]) -> None:
        self.ball_xy = np.asarray(ball_xy, dtype=np.float64)[:2]
        self.t_ball = time.time()

    def set_status(self, switch_status: int) -> None:
        self.status = int(switch_status)

    def set_estop(self, active: bool) -> None:
        self.estop = bool(active)

    # ---- 主循环 --------------------------------------------------------- #
    def block_reason(self) -> Optional[str]:
        now = time.time()
        if self.estop:
            return "estop"
        if self.q is None or now - self.t_joint > self.joint_timeout_s:
            return "joint_states stale/missing"
        if (self.quat is None or self.robot_xy is None
                or now - self.t_odom > self.odom_timeout_s):
            return "odom/imu stale/missing"
        if self.ball_xy is None or now - self.t_ball > self.ball_timeout_s:
            return "mocap ball stale/missing"
        if self.status is not None and self.status in BAD_STATUS:
            return f"factory switch_status={self.status}"
        return None

    def last_action(self) -> np.ndarray:
        if self.last_action_source == "zeros":
            return np.zeros(12)
        return (self.q - self.default_q) / joint_spec.OFFICIAL_ACTION_SCALE

    def tick(self) -> tuple[Optional[np.ndarray], Optional[str]]:
        reason = self.block_reason()
        if reason is not None:
            return None, reason
        obs = build_obs(
            joint_pos=self.q, joint_vel=self.qd,
            base_ang_vel=self.gyro, quat_xyzw=self.quat,
            last_action=self.last_action(), base_lin_vel=self.base_lin_vel,
            robot_xy=self.robot_xy, ball_xy=self.ball_xy,
            default_q=self.default_q, goal_xy=self.goal_xy)
        cmd = np.clip(self.policy.act(obs), -self.vel_limit, self.vel_limit)
        return cmd, None
