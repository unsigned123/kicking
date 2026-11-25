"""kick_sim2real 主节点：50Hz 策略闭环 + 安全 FSM。

数据流::

    /joint_states (q, qd) ─┐
    /imu/data (quat, gyro) ─┤
    /odom (pose, vBody)    ─┼→ 55 维 obs → 策略 → [vx,vy,wz] → 运输后端
    mocap /ball_pose       ─┘                        (Twist 或 LCM)

安全（缺省开启；任一触发 → 发零速保持心跳）：
  - /joint_states 超时 > joint_timeout_s
  - odom/动捕超时
  - 出厂运控 switch_status ∈ {2 ESTOP, 3 EDAMP, 4 LIFTED}（LCM 后端）
  - 急停话题 std_msgs/Bool True（可选）
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from . import joint_spec
from .obs_builder import (GOAL_CENTER_XY, build_obs)
from .policy import PolicyRuntime
from .transport import LcmTransport, make_transport

SAFE_STATUS = {0}       # Done
BAD_STATUS = {2, 3, 4}  # ESTOP / EDAMP / LIFTED


class KickSim2RealNode(Node):
    def __init__(self) -> None:
        super().__init__("kick_sim2real")

        # ---- parameters ------------------------------------------------- #
        self.declare_parameter("policy_ckpt", "")
        self.declare_parameter("backend", "twist")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("ball_pose_topic", "/mocap/ball_pose")
        self.declare_parameter("estop_topic", "/kick_sim2real/estop")
        self.declare_parameter("goal_center_xy", [0.0, 3.0])
        self.declare_parameter("mocap_frame_offset_xy", [0.0, 0.0])
        self.declare_parameter("mocap_yaw_offset", 0.0)
        self.declare_parameter("vel_limit", [1.6, 0.55, 2.5])
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("joint_timeout_s", 0.10)
        self.declare_parameter("odom_timeout_s", 0.25)
        self.declare_parameter("ball_timeout_s", 0.30)
        self.declare_parameter("last_action_source", "joint_deviation")
        self.declare_parameter("lcm_gait_id", 26)
        self.declare_parameter("lcm_mode", 11)
        self.declare_parameter("dry_run", False)

        gp = lambda k: self.get_parameter(k).value  # noqa: E731
        self._goal = tuple(float(v) for v in gp("goal_center_xy"))
        self._vel_limit = np.asarray(gp("vel_limit"), dtype=np.float64)
        self._joint_timeout = float(gp("joint_timeout_s"))
        self._odom_timeout = float(gp("odom_timeout_s"))
        self._ball_timeout = float(gp("ball_timeout_s"))
        self._last_action_source = str(gp("last_action_source"))
        self._dry_run = bool(gp("dry_run"))
        self._mocap_off = np.asarray(gp("mocap_frame_offset_xy"),
                                     dtype=np.float64)
        self._mocap_yaw = float(gp("mocap_yaw_offset"))
        self._default_q = np.asarray(joint_spec.TRAIN_DEFAULT_Q,
                                     dtype=np.float64)

        ckpt = str(gp("policy_ckpt"))
        if not ckpt:
            raise RuntimeError("参数 policy_ckpt 未设置（导出的 .pt 路径）")
        self.policy = PolicyRuntime(ckpt)
        self.get_logger().info(
            f"policy loaded: {ckpt} (obs={self.policy.obs_dim}, "
            f"hidden={self.policy.hidden}, md5={PolicyRuntime.md5(ckpt)[:8]})")

        # ---- state buffers ---------------------------------------------- #
        self._q: Optional[np.ndarray] = None
        self._qd: Optional[np.ndarray] = None
        self._quat = None
        self._gyro = None
        self._robot_xy = None
        self._base_lin_vel = None
        self._ball_xy = None
        self._t_joint = self._t_odom = self._t_ball = 0.0
        self._estop = False
        self._status: Optional[int] = None
        self._joint_index: Optional[list[int]] = None
        self._joint_warned = False

        # ---- transport --------------------------------------------------- #
        backend = str(gp("backend"))
        cfg = {"backend": backend,
               "cmd_vel_topic": str(gp("cmd_vel_topic")),
               "lcm_gait_id": int(gp("lcm_gait_id")),
               "lcm_mode": int(gp("lcm_mode")),
               "vel_limit": tuple(self._vel_limit)}
        self.transport = make_transport(self, cfg)
        if isinstance(self.transport, LcmTransport):
            self.transport._on_odom = self._on_lcm_odom
            self.transport._on_status = self._on_lcm_status
            self._lcm = True
        else:
            self._lcm = False

        # ---- ROS subscriptions ------------------------------------------ #
        from std_msgs.msg import Bool
        self.create_subscription(JointState, str(gp("joint_states_topic")),
                                 self._on_joint, 10)
        self.create_subscription(Imu, str(gp("imu_topic")), self._on_imu, 10)
        self.create_subscription(Odometry, str(gp("odom_topic")), self._on_odom,
                                 10)
        from geometry_msgs.msg import PoseStamped
        self.create_subscription(PoseStamped, str(gp("ball_pose_topic")),
                                 self._on_ball, 10)
        self.create_subscription(Bool, str(gp("estop_topic")), self._on_estop,
                                 10)

        hz = float(gp("control_hz"))
        self.create_timer(1.0 / hz, self._tick)
        self._warn_t = 0.0
        self.get_logger().info(
            f"kick_sim2real ready: backend={backend} hz={hz} "
            f"goal={self._goal} vel_limit={self._vel_limit.tolist()} "
            f"dry_run={self._dry_run}")

    # ---- callbacks ------------------------------------------------------ #
    def _on_joint(self, msg: JointState) -> None:
        if self._joint_index is None:
            try:
                self._joint_index = joint_spec.resolve_joint_indices(msg.name)
            except KeyError as e:
                if not self._joint_warned:
                    self.get_logger().error(str(e))
                    self._joint_warned = True
                return
        idx = self._joint_index
        q_sdk = [msg.position[i] for i in idx]
        qd_sdk = [msg.velocity[i] for i in idx] if msg.velocity else [0.0] * 12
        self._q = np.asarray(joint_spec.reorder_sdk_to_train(q_sdk))
        self._qd = np.asarray(joint_spec.reorder_sdk_to_train(qd_sdk))
        self._t_joint = time.time()

    def _on_imu(self, msg: Imu) -> None:
        self._quat = np.array([msg.orientation.x, msg.orientation.y,
                               msg.orientation.z, msg.orientation.w])
        self._gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                               msg.angular_velocity.z])

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._robot_xy = np.array([p.x, p.y])
        v = msg.twist.twist.linear
        self._base_lin_vel = np.array([v.x, v.y, v.z])
        w = msg.twist.twist.angular
        self._gyro = np.array([w.x, w.y, w.z])
        o = msg.pose.pose.orientation
        self._quat = np.array([o.x, o.y, o.z, o.w])
        self._t_odom = time.time()

    def _on_lcm_odom(self, xyz, vbody, omega, rpy, stamp) -> None:
        self._robot_xy = xyz[:2].copy()
        self._base_lin_vel = vbody.copy()
        self._gyro = omega.copy()
        # rpy → quat（xyz 欧拉，zyx 序）
        import math
        cy, cp, cr = (math.cos(rpy[2] / 2), math.cos(rpy[1] / 2),
                      math.cos(rpy[0] / 2))
        sy, sp, sr = (math.sin(rpy[2] / 2), math.sin(rpy[1] / 2),
                      math.sin(rpy[0] / 2))
        self._quat = np.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy])
        self._t_odom = stamp

    def _on_lcm_status(self, switch_status: int) -> None:
        self._status = switch_status

    def _on_ball(self, msg) -> None:
        p = msg.pose.position
        x = p.x + self._mocap_off[0]
        y = p.y + self._mocap_off[1]
        if abs(self._mocap_yaw) > 1e-9:
            c, s = np.cos(self._mocap_yaw), np.sin(self._mocap_yaw)
            x, y = c * x - s * y, s * x + c * y
        self._ball_xy = np.array([x, y])
        self._t_ball = time.time()

    def _on_estop(self, msg) -> None:
        self._estop = bool(msg.data)

    # ---- main loop ------------------------------------------------------ #
    def _safe_reason(self) -> Optional[str]:
        now = time.time()
        if self._estop:
            return "estop"
        if self._q is None or now - self._t_joint > self._joint_timeout:
            return "joint_states stale/missing"
        if (self._quat is None or self._robot_xy is None
                or now - self._t_odom > self._odom_timeout):
            return "odom/imu stale/missing"
        if self._ball_xy is None or now - self._t_ball > self._ball_timeout:
            return "mocap ball stale/missing"
        if self._status is not None and self._status in BAD_STATUS:
            return f"factory switch_status={self._status}"
        return None

    def _tick(self) -> None:
        if self._lcm:
            self.transport.spin_once(0.0)
        reason = self._safe_reason()
        if reason is not None:
            if self._dry_run:
                self.get_logger().warn(f"[dry_run] blocked: {reason}",
                                       throttle_duration_sec=2.0)
                return
            self.transport.send(np.zeros(3))
            now = time.time()
            if now - self._warn_t > 2.0:
                self.get_logger().warn(f"hold zero: {reason}")
                self._warn_t = now
            return

        last_action = self._last_action()
        obs = build_obs(
            joint_pos=self._q, joint_vel=self._qd,
            base_ang_vel=self._gyro, quat_xyzw=self._quat,
            last_action=last_action, base_lin_vel=self._base_lin_vel,
            robot_xy=self._robot_xy, ball_xy=self._ball_xy,
            default_q=self._default_q, goal_xy=self._goal)
        cmd = self.policy.act(obs)
        cmd = np.clip(cmd, -self._vel_limit, self._vel_limit)
        if self._dry_run:
            self.get_logger().info(
                f"[dry_run] cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]",
                throttle_duration_sec=1.0)
            return
        self.transport.send(cmd)

    def _last_action(self) -> np.ndarray:
        if self._last_action_source == "zeros":
            return np.zeros(12)
        # joint_deviation 代理：低层 target = default + 0.5·a（官方口径），
        # 实测 q 跟踪 target → a ≈ (q − default)/0.5
        return (self._q - self._default_q) / joint_spec.OFFICIAL_ACTION_SCALE

    def destroy_node(self) -> bool:
        try:
            if not self._dry_run:
                self.transport.send(np.zeros(3))
        finally:
            self.transport.close()
            return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = KickSim2RealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
