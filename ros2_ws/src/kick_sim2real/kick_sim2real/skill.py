"""模板兼容的 Skill 封装：可挂进 Machine-Dog 的 skill_manager。

在模板工程中使用::

    from kick_sim2real.skill import KickPolicySkill
    ...
    self.skills = {
        5: KickPolicySkill(self, move_local_skill=move_skill,
                           policy_ckpt="/home/mi/kick_policy.pt"),
    }

Skill 自身只做「读缓存 + 推理 + 通过 move_local 发布」；输入缓存由
``KickPolicySkill.bind_sources`` 订阅的模板话题填充（absolute_pose / vrpn /
joint_states / imu）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # 模板环境内可 import；独立环境用内置兜底基类
    from robot_skills.skill_base import Skill  # type: ignore
except Exception:  # pragma: no cover
    class Skill:  # type: ignore
        def __init__(self) -> None:
            self._running = False

        def start(self, params):
            self._running = True

        def update(self, dt):
            return self._running

        def stop(self):
            self._running = False


from .controller import KickController


class KickPolicySkill(Skill):
    """SKILL 语义：持续型；``start([vx_max, vy_max, wz_max])`` 可覆盖限速。"""

    def __init__(self, node, move_local_skill, policy_ckpt: str,
                 dog_name: str = "dog0",
                 goal_xy=(0.0, 3.0), vel_limit=(1.6, 0.55, 2.5)) -> None:
        super().__init__()
        self.node = node
        self.move_local = move_local_skill
        self.ctrl = KickController(policy_ckpt, goal_xy=goal_xy,
                                   vel_limit=vel_limit)
        self.active = False
        self.dog_name = dog_name
        self._bind_sources_template()
        node.get_logger().info(
            f"KickPolicySkill ready (policy obs={self.ctrl.policy.obs_dim}, "
            f"dog={dog_name})")

    # ---- 模板话题面 ----------------------------------------------------- #
    def _bind_sources_template(self) -> None:
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, JointState

        from . import joint_spec

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        node = self.node
        node.create_subscription(JointState, "/joint_states",
                                 self._on_joint, qos)
        node.create_subscription(Imu, "/imu/data", self._on_imu, qos)
        node.create_subscription(Odometry, f"/{self.dog_name}/odom_out",
                                 self._on_odom, qos)
        try:
            from robot_interfaces.msg import AbsoluteObjectInfo
            node.create_subscription(
                AbsoluteObjectInfo, f"/absolute_pose/{self.dog_name}",
                self._on_dog_pose, 10)
            node.create_subscription(
                AbsoluteObjectInfo, "/absolute_pose/ball", self._on_ball, 10)
        except Exception:  # pragma: no cover - 无模板接口包时降级
            node.get_logger().warn(
                "robot_interfaces 不可用：请另行提供球/狗位姿输入")

        self._joint_index = None

    # ---- 回调 ----------------------------------------------------------- #
    def _on_joint(self, msg) -> None:
        from . import joint_spec
        if self._joint_index is None:
            try:
                self._joint_index = joint_spec.resolve_joint_indices(msg.name)
            except KeyError as e:
                self.node.get_logger().error(str(e))
                return
        idx = self._joint_index
        self.ctrl.set_joints(
            joint_spec.reorder_sdk_to_train(
                [msg.position[i] for i in idx]),
            joint_spec.reorder_sdk_to_train(
                [msg.velocity[i] for i in idx] if msg.velocity else [0.0] * 12))

    def _on_imu(self, msg) -> None:
        if self.ctrl.quat is not None:
            return
        self.ctrl.set_imu(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z,
             msg.orientation.w],
            [msg.angular_velocity.x, msg.angular_velocity.y,
             msg.angular_velocity.z])

    def _on_odom(self, msg) -> None:
        p, v, w = (msg.pose.pose.position, msg.twist.twist.linear,
                   msg.twist.twist.angular)
        self.ctrl.set_odom([p.x, p.y], [v.x, v.y, v.z], [w.x, w.y, w.z])

    def _on_dog_pose(self, msg) -> None:
        p, o = msg.position, msg.orientation
        v, w = msg.linear_velocity, msg.angular_velocity
        self.ctrl.set_odom([p.x, p.y], [v.x, v.y, v.z], [w.x, w.y, w.z],
                           [o.x, o.y, o.z, o.w])

    def _on_ball(self, msg) -> None:
        self.ctrl.set_ball([msg.position.x, msg.position.y])

    # ---- Skill 接口 ----------------------------------------------------- #
    def start(self, params) -> None:
        super().start(params)
        self.active = True
        if params and len(params) >= 3:
            self.ctrl.vel_limit = np.asarray(params[:3], dtype=np.float64)

    def update(self, dt: float) -> bool:
        if not self.active:
            self.move_local.set_emergency(False)
            return True
        cmd, reason = self.ctrl.tick()
        if cmd is None:
            self.move_local._publish_motion(0.0, 0.0, 0.0)
            self.node.get_logger().warn(f"kick policy hold zero: {reason}",
                                        throttle_duration_sec=2.0)
            return True
        self.move_local.target_vx = float(cmd[0])
        self.move_local.target_vy = float(cmd[1])
        self.move_local.target_wz = float(cmd[2])
        self.move_local.active = True   # 由 move_local 统一限幅+选步态+发布
        return True

    def stop(self) -> None:
        super().stop()
        self.active = False
        self.move_local._publish_motion(0.0, 0.0, 0.0)
