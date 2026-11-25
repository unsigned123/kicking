"""速度命令输出后端。

后端 A — ROS2 Twist（缺省）::

    发布 geometry_msgs/Twist 到 ``cmd_vel_topic``（linear.x=vx, linear.y=vy,
    angular.z=wz）。适用于机器人侧已有 ROS2 桥（cyberdog_ros2 的
    motion_control / 自建桥）把 Twist 转给出厂运控的场景。

后端 B — LCM 直连出厂运控（文档接口）::

    发布 ``robot_control_cmd``（udpm://239.255.76.67:7671）::

        mode = 11 (LOCOMOTION)
        gait_id = 26 (TROT 自变频；可选 10=FAST / 3=MEDIUM / 27=SLOW)
        vel_des = [vx, vy(≤1.6), yaw(≤2.5)]
        duration = 0            # 持续到下一帧
        life_count += 1 (mod 256)  # 心跳；500ms 无帧出厂控制器趴下保护

    同时订阅 ``global_to_robot``（7667, 50Hz 里程计：xyz/vBody/omegaBody/rpy）
    与 ``robot_control_response``（7670, 50Hz：switch_status 安全态）。

LCM 消息类需先生成：``bash scripts/fetch_lcm_msgs.sh``（从 MiRoboticsLab
loco_hl_example 拷贝 lcm-gen 产物到 ``kick_sim2real/lcm_msgs/``）。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np


class TwistTransport:
    def __init__(self, node, topic: str) -> None:
        from geometry_msgs.msg import Twist  # noqa: F401
        self._msg_cls = Twist
        self._pub = node.create_publisher(Twist, topic, 10)
        self._last = None

    def send(self, cmd: np.ndarray) -> None:
        msg = self._msg_cls()
        msg.linear.x = float(cmd[0])
        msg.linear.y = float(cmd[1])
        msg.angular.z = float(cmd[2])
        self._pub.publish(msg)

    def close(self) -> None:
        pass


class LcmTransport:
    """出厂高层运控直连（mode=11 / gait_id=26 速度跟踪）。"""

    CMD_URL = "udpm://239.255.76.67:7671?ttl=255"
    STATE_URL = "udpm://239.255.76.67:7670?ttl=255"
    ODOM_URL = "udpm://239.255.76.67:7667?ttl=255"

    def __init__(self, gait_id: int = 26, mode: int = 11,
                 vel_limit: tuple[float, float, float] = (1.6, 0.55, 2.5),
                 on_odom: Optional[Callable] = None,
                 on_status: Optional[Callable] = None) -> None:
        import lcm  # optional dependency
        from kick_sim2real import lcm_msgs  # generated files

        self._lcm = lcm.LCM(self.CMD_URL)
        self._state_lcm = lcm.LCM(self.STATE_URL)
        self._odom_lcm = lcm.LCM(self.ODOM_URL)
        self._cmd_cls = lcm_msgs.robot_control_cmd_lcmt
        self._resp_cls = lcm_msgs.robot_control_response_lcmt
        self._odom_cls = lcm_msgs.localization_lcmt
        self._gait_id = int(gait_id)
        self._mode = int(mode)
        self._limit = np.asarray(vel_limit, dtype=np.float64)
        self._life = 0
        self._on_odom = on_odom
        self._on_status = on_status
        self._last_send = 0.0

        self._state_lcm.subscribe("robot_control_response", self._resp_cb)
        self._odom_lcm.subscribe("global_to_robot", self._odom_cb)
        self._msg = self._cmd_cls()
        self._msg.mode = self._mode
        self._msg.gait_id = self._gait_id
        self._msg.duration = 0
        self._msg.contact = 0x0F
        self._msg.step_height = [0.05, 0.05]

    # ---- subscriptions -------------------------------------------------- #
    def _resp_cb(self, channel, data) -> None:
        msg = self._resp_cls.decode(data)
        if self._on_status is not None:
            self._on_status(int(msg.switch_status))

    def _odom_cb(self, channel, data) -> None:
        msg = self._odom_cls.decode(data)
        if self._on_odom is not None:
            self._on_odom(
                np.array(msg.xyz, dtype=np.float64),
                np.array(msg.vBody, dtype=np.float64),
                np.array(msg.omegaBody, dtype=np.float64),
                np.array(msg.rpy, dtype=np.float64),
                time.time(),
            )

    # ---- command -------------------------------------------------------- #
    def send(self, cmd: np.ndarray) -> None:
        clamped = np.clip(np.asarray(cmd, dtype=np.float64),
                          -self._limit, self._limit)
        self._msg.vel_des = [float(clamped[0]), float(clamped[1]),
                             float(clamped[2])]
        self._life = (self._life + 1) % 256
        self._msg.life_count = self._life
        self._lcm.publish("robot_control_cmd",
                          self._msg.encode())
        self._last_send = time.time()

    def spin_once(self, timeout_s: float = 0.0) -> None:
        self._state_lcm.handle_timeout(int(timeout_s * 1000))
        self._odom_lcm.handle_timeout(0)

    def close(self) -> None:
        try:
            self.send(np.zeros(3))
        except Exception:  # pragma: no cover
            pass


def make_transport(node, cfg: dict):
    backend = cfg.get("backend", "twist")
    if backend == "twist":
        return TwistTransport(node, cfg.get("cmd_vel_topic", "/cmd_vel"))
    if backend == "lcm":
        return LcmTransport(
            gait_id=int(cfg.get("lcm_gait_id", 26)),
            mode=int(cfg.get("lcm_mode", 11)),
            vel_limit=tuple(cfg.get("vel_limit", (1.6, 0.55, 2.5))),
        )
    raise ValueError(f"未知 backend: {backend}（∈ twist|lcm）")
