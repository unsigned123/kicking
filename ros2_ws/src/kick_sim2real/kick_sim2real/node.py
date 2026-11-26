"""kick_sim2real 独立节点（不依赖 skill_manager；也可被 skill.py 包装复用）。

数据源（与 Machine-Dog 模板同话题面）::

    /absolute_pose/<dog_name>   (robot_interfaces/AbsoluteObjectInfo) 狗世界位姿+朝向
    /absolute_pose/ball         (robot_interfaces/AbsoluteObjectInfo) 球世界位姿
    /<dog_name>/odom_out        (nav_msgs/Odometry) 机体速度（body 系）
    /joint_states               (sensor_msgs/JointState) 12 关节
    /imu/data                   (sensor_msgs/Imu) 陀螺/四元数（缺省回退）
    /kick_sim2real/estop        (std_msgs/Bool)

输出：``protocol.msg/MotionServoCmd`` → ``/<dog_name>/motion_servo_cmd``
（出厂速度控制器；gait 303/308/305 自动选择）。
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from .controller import KickController
from .policy import PolicyRuntime
from .transport import LcmTransport, make_transport


class KickSim2RealNode(Node):
    def __init__(self) -> None:
        super().__init__("kick_sim2real")
        d = self.declare_parameter

        d("policy_ckpt", "")
        d("backend", "motion_servo")
        d("dog_name", "dog0")
        d("motion_servo_topic", "")
        d("cmd_vel_topic", "/cmd_vel")
        d("joint_states_topic", "/joint_states")
        d("imu_topic", "/imu/data")
        d("odom_topic", "")
        d("dog_pose_topic", "")
        d("ball_pose_topic", "/absolute_pose/ball")
        d("pose_source", "vrpn")            # vrpn（方案A，直接原始动捕）| absolute
        d("dog_tracker", "dog0")            # 本机 VRPN tracker 名（如 GEXY2/Lizj_dog）
        d("ball_tracker", "ball")
        d("goal_source", "vrpn")            # vrpn | fixed
        d("goal_tracker", "goal_left")      # 进攻门（双门场地二选一）
        d("estop_topic", "/kick_sim2real/estop")
        d("goal_center_xy", [0.0, 3.0])
        d("mocap_frame_offset_xy", [0.0, 0.0])
        d("mocap_yaw_offset", 0.0)
        d("odom_to_world_xy", [0.0, 0.0])
        d("odom_to_world_yaw", 0.0)
        d("vel_limit", [1.6, 0.55, 2.5])
        d("control_hz", 50.0)
        d("joint_timeout_s", 0.10)
        d("odom_timeout_s", 0.25)
        d("ball_timeout_s", 0.30)
        d("last_action_source", "joint_deviation")
        d("joint_source", "ros")          # ros(/joint_states) | lcm_spi(只读)
        d("odom_source", "ros")           # ros(absolute_pose/odom_out) | lcm_localization
        d("lcm_gait_id", 26)
        d("lcm_mode", 11)
        d("dry_run", False)

        gp = lambda k: self.get_parameter(k).value  # noqa: E731
        dog = str(gp("dog_name"))
        self._dry_run = bool(gp("dry_run"))
        self._last_body_vel = np.zeros(3)
        self._last_gyro = np.zeros(3)
        self._mocap_off = np.asarray(gp("mocap_frame_offset_xy"),
                                     dtype=np.float64)
        self._mocap_yaw = float(gp("mocap_yaw_offset"))
        self._warn_t = 0.0

        ckpt = str(gp("policy_ckpt"))
        if not ckpt:
            raise RuntimeError("参数 policy_ckpt 未设置（export 产物路径）")
        self.ctrl = KickController(
            ckpt,
            goal_xy=tuple(gp("goal_center_xy")),
            vel_limit=tuple(gp("vel_limit")),
            joint_timeout_s=float(gp("joint_timeout_s")),
            odom_timeout_s=float(gp("odom_timeout_s")),
            ball_timeout_s=float(gp("ball_timeout_s")),
            last_action_source=str(gp("last_action_source")),
        )
        self.get_logger().info(
            f"policy loaded: {ckpt} (obs={self.ctrl.policy.obs_dim}, "
            f"hidden={self.ctrl.policy.hidden}, "
            f"md5={PolicyRuntime.md5(ckpt)[:8]})")

        # ---- transport --------------------------------------------------- #
        backend = str(gp("backend"))
        cfg = {"backend": backend, "dog_name": dog,
               "motion_servo_topic": str(gp("motion_servo_topic")) or None,
               "cmd_vel_topic": str(gp("cmd_vel_topic")),
               "lcm_gait_id": int(gp("lcm_gait_id")),
               "lcm_mode": int(gp("lcm_mode")),
               "vel_limit": tuple(gp("vel_limit"))}
        self.transport = make_transport(self, cfg)
        self._lcm = isinstance(self.transport, LcmTransport)
        if self._lcm:
            self.transport._on_odom = self._on_lcm_odom
            self.transport._on_status = self.ctrl.set_status

        # ---- sources (template topics) ----------------------------------- #
        from robot_interfaces.msg import AbsoluteObjectInfo
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, JointState

        dog_pose_topic = str(gp("dog_pose_topic")) or f"/absolute_pose/{dog}"
        odom_topic = str(gp("odom_topic")) or f"/{dog}/odom_out"

        pose_source = str(gp("pose_source"))
        goal_source = str(gp("goal_source"))
        if goal_source == "vrpn":
            from geometry_msgs.msg import PoseStamped
            self.create_subscription(
                PoseStamped, f"/vrpn/{gp('goal_tracker')}/pose",
                lambda m: self.ctrl.set_goal([m.pose.position.x,
                                              m.pose.position.y]), 10)
        joint_source = str(gp("joint_source"))
        odom_source = str(gp("odom_source"))
        if joint_source == "ros":
            self.create_subscription(JointState,
                                     str(gp("joint_states_topic")),
                                     self._on_joint, 10)
        elif joint_source == "lcm_spi":
            from .lcm_sources import SpiDataReader
            self._spi = SpiDataReader(self.ctrl.set_joints)
            self.get_logger().info(
                "joint_source=lcm_spi（只读订阅 spi_data，不进入电机模式）")
        else:
            raise ValueError(f"joint_source 非法: {joint_source}")
        self.create_subscription(Imu, str(gp("imu_topic")), self._on_imu, 10)
        if odom_source == "ros":
            self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        elif odom_source == "lcm_localization":
            from .lcm_sources import LocalizationReader
            self._loc = LocalizationReader(self.ctrl.set_odom)
            self.get_logger().info("odom_source=lcm_localization（只读）")
        else:
            raise ValueError(f"odom_source 非法: {odom_source}")
        if pose_source == "vrpn":
            from geometry_msgs.msg import PoseStamped
            self.create_subscription(
                PoseStamped, f"/vrpn/{gp('dog_tracker')}/pose",
                self._on_vrpn_dog_pose, 10)
            self.create_subscription(
                PoseStamped, f"/vrpn/{gp('ball_tracker')}/pose",
                self._on_vrpn_ball, 10)
        elif pose_source == "absolute":
            self.create_subscription(AbsoluteObjectInfo, dog_pose_topic,
                                     self._on_dog_pose, 10)
            self.create_subscription(
                AbsoluteObjectInfo, str(gp("ball_pose_topic")),
                self._on_ball, 10)
        else:
            raise ValueError(f"pose_source 非法: {pose_source}")
        self.create_subscription(Bool, str(gp("estop_topic")),
                                 lambda m: self.ctrl.set_estop(m.data), 10)

        self._joint_index = None
        self._joint_warned = False
        hz = float(gp("control_hz"))
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            f"kick_sim2real ready: backend={backend} dog={dog} hz={hz} "
            f"goal={self.ctrl.goal_xy} vel_limit={self.ctrl.vel_limit.tolist()} "
            f"dry_run={self._dry_run}")

    # ---- callbacks ------------------------------------------------------ #
    def _on_joint(self, msg) -> None:
        from . import joint_spec
        if self._joint_index is None:
            try:
                self._joint_index = joint_spec.resolve_joint_indices(msg.name)
            except KeyError as e:
                if not self._joint_warned:
                    self.get_logger().error(str(e))
                    self._joint_warned = True
                return
        idx = self._joint_index
        q = [msg.position[i] for i in idx]
        qd = [msg.velocity[i] for i in idx] if msg.velocity else [0.0] * 12
        self.ctrl.set_joints(joint_spec.reorder_sdk_to_train(q),
                             joint_spec.reorder_sdk_to_train(qd))

    def _on_imu(self, msg) -> None:
        if self.ctrl.quat is not None:
            return  # 世界朝向以 absolute_pose 为准
        self.ctrl.set_imu(
            [msg.orientation.x, msg.orientation.y,
             msg.orientation.z, msg.orientation.w],
            [msg.angular_velocity.x, msg.angular_velocity.y,
             msg.angular_velocity.z])

    def _on_odom(self, msg) -> None:
        """odom_out 提供机体速度/陀螺（body 系，与坐标系无关）。"""
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        self._last_body_vel = np.array([v.x, v.y, v.z])
        self._last_gyro = np.array([w.x, w.y, w.z])
        if self.ctrl.robot_xy is None:  # 位姿源缺席时退化用 odom 位置
            p = msg.pose.pose.position
            self.ctrl.robot_xy = np.array([p.x, p.y])
        self.ctrl.set_odom(self.ctrl.robot_xy, self._last_body_vel,
                           self._last_gyro)

    def _on_vrpn_dog_pose(self, msg) -> None:
        """方案A：狗世界位姿直接取 VRPN（与球/门同系，无需任何变换）。"""
        p, o = msg.pose.position, msg.pose.orientation
        self.ctrl.set_odom([p.x, p.y], self._last_body_vel,
                           self._last_gyro, [o.x, o.y, o.z, o.w])

    def _on_vrpn_ball(self, msg) -> None:
        p = msg.pose.position
        self.ctrl.set_ball([p.x, p.y])

    def _on_dog_pose(self, msg) -> None:
        p = msg.position
        o = msg.orientation
        v = msg.linear_velocity
        w = msg.angular_velocity
        self.ctrl.set_odom([p.x, p.y], [v.x, v.y, v.z],
                           [w.x, w.y, w.z], [o.x, o.y, o.z, o.w])

    def _on_ball(self, msg) -> None:
        p = msg.position
        x, y = p.x + self._mocap_off[0], p.y + self._mocap_off[1]
        if abs(self._mocap_yaw) > 1e-9:
            c, s = np.cos(self._mocap_yaw), np.sin(self._mocap_yaw)
            x, y = c * x - s * y, s * x + c * y
        self.ctrl.set_ball([x, y])

    def _on_lcm_odom(self, xyz, vbody, omega, rpy, stamp) -> None:
        import math
        cy, cp, cr = (math.cos(rpy[2] / 2), math.cos(rpy[1] / 2),
                      math.cos(rpy[0] / 2))
        sy, sp, sr = (math.sin(rpy[2] / 2), math.sin(rpy[1] / 2),
                      math.sin(rpy[0] / 2))
        self.ctrl.set_odom(
            xyz[:2], vbody, omega,
            [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
             cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])

    # ---- loop ----------------------------------------------------------- #
    def _tick(self) -> None:
        if self._lcm:
            self.transport.spin_once(0.0)
        cmd, reason = self.ctrl.tick()
        if cmd is None:
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
        if self._dry_run:
            self.get_logger().info(
                f"[dry_run] cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]",
                throttle_duration_sec=1.0)
            return
        self.transport.send(cmd)

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
