"""只读 LCM 数据源（不改运控模式）：关节反馈 + 里程计/IMU。

高层接口章节（cyberdog_loco_cn）的 5 个通道清单里**没有关节反馈**；关节 q/qd
在底层通道 ``spi_data``（udpm 7667）里——底层 SDK 的 CustomInterface 会订阅它，
只有向 ``motor_ctrl`` *发布* 才进入电机模式。**本模块只订阅、不发布**，因此在
出厂高层控制器驱动机器人的同时可以安全读取关节状态。

``spi_data`` 字段序（lcm-gen hpp 权威，payload 偏移相对消息体起点）::

    q_abad[4] @0   q_hip[4] @16  q_knee[4] @32
    qd_abad[4] @48 qd_hip[4] @64 qd_knee[4] @80
    flags[12] @96  spi_driver_status @144
    tau_* @148/164/180   tmp_* @196/204/212   reserve[40] @220
    总 payload = 260 B；LCM 包头 8 B（magic+fingerprint）→ 从 data[8:] 解析

符号：底层 SDK 每腿 ``sig = (+1, −1, −1)`` 作用于 abad/hip/knee；腿序 =
FR, FL, RR, RL（SDK 序），随后由 joint_spec 重排到训练序。
"""

from __future__ import annotations

import math
import struct
import threading
import time
from typing import Callable, Optional

import numpy as np

from . import joint_spec

URL_MAIN = "udpm://239.255.76.67:7667?ttl=255"
"""主场通道（spi_data / global_to_robot 同总线）。"""

_LEG_SIGN = (1.0, -1.0, -1.0) * 4
"""每腿 abad(+)/hip(−)/knee(−) 的符号修正（底层 SDK 同表）。"""

SPI_PAYLOAD_OFFSET = 8


def decode_spi_data(data: bytes) -> dict[str, tuple[float, ...]]:
    """解码 spi_data 消息体 → {'q': 12, 'qd': 12, 'tau': 12}（SDK 序，已符号修正）。"""
    if len(data) < SPI_PAYLOAD_OFFSET + 260:
        raise ValueError(f"spi_data 长度异常: {len(data)}")
    off = SPI_PAYLOAD_OFFSET
    q_abad = struct.unpack_from("<4f", data, off + 0)
    q_hip = struct.unpack_from("<4f", data, off + 16)
    q_knee = struct.unpack_from("<4f", data, off + 32)
    qd_abad = struct.unpack_from("<4f", data, off + 48)
    qd_hip = struct.unpack_from("<4f", data, off + 64)
    qd_knee = struct.unpack_from("<4f", data, off + 80)
    tau_abad = struct.unpack_from("<4f", data, off + 148)
    tau_hip = struct.unpack_from("<4f", data, off + 164)
    tau_knee = struct.unpack_from("<4f", data, off + 180)

    # 腿序 FR,FL,RR,RL × abad/hip/knee，逐腿符号修正
    q = [0.0] * 12
    qd = [0.0] * 12
    tau = [0.0] * 12
    for i in range(4):
        q[i * 3 + 0] = q_abad[i] * _LEG_SIGN[i * 3 + 0]
        q[i * 3 + 1] = q_hip[i] * _LEG_SIGN[i * 3 + 1]
        q[i * 3 + 2] = q_knee[i] * _LEG_SIGN[i * 3 + 2]
        qd[i * 3 + 0] = qd_abad[i] * _LEG_SIGN[i * 3 + 0]
        qd[i * 3 + 1] = qd_hip[i] * _LEG_SIGN[i * 3 + 1]
        qd[i * 3 + 2] = qd_knee[i] * _LEG_SIGN[i * 3 + 2]
        tau[i * 3 + 0] = tau_abad[i] * _LEG_SIGN[i * 3 + 0]
        tau[i * 3 + 1] = tau_hip[i] * _LEG_SIGN[i * 3 + 1]
        tau[i * 3 + 2] = tau_knee[i] * _LEG_SIGN[i * 3 + 2]
    return {"q": tuple(q), "qd": tuple(qd), "tau": tuple(tau)}


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    """欧拉(固定轴 xyz) → 四元数 xyzw（与 absolute_coordination 同序）。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class SpiDataReader:
    """spi_data 只读订阅 → on_joints(q_train, qd_train)。"""

    def __init__(self, on_joints: Callable[[list[float], list[float]], None],
                 url: str = URL_MAIN, channel: str = "spi_data") -> None:
        import lcm
        self._cb = on_joints
        self._lcm = lcm.LCM(url)
        self._lcm.subscribe(channel, self._on_msg)
        self._t = threading.Thread(target=self._spin, daemon=True)
        self._t.start()

    def _spin(self) -> None:
        while True:
            try:
                self._lcm.handle()
            except Exception:  # pragma: no cover - 通道异常不拖垮主进程
                time.sleep(0.1)

    def _on_msg(self, channel: str, data: bytes) -> None:
        try:
            dec = decode_spi_data(data)
        except ValueError:
            return
        self._cb(joint_spec.reorder_sdk_to_train(dec["q"]),
                 joint_spec.reorder_sdk_to_train(dec["qd"]))


class LocalizationReader:
    """global_to_robot 只读订阅 → on_odom(xy, v_body, omega_body, quat_xyzw)。"""

    def __init__(self, on_odom: Callable, url: str = URL_MAIN,
                 channel: str = "global_to_robot") -> None:
        import lcm
        from kick_sim2real.lcm_msgs import localization_lcmt
        self._cls = localization_lcmt
        self._cb = on_odom
        self._lcm = lcm.LCM(url)
        self._lcm.subscribe(channel, self._on_msg)
        self._t = threading.Thread(target=self._spin, daemon=True)
        self._t.start()

    def _spin(self) -> None:
        while True:
            try:
                self._lcm.handle()
            except Exception:  # pragma: no cover
                time.sleep(0.1)

    def _on_msg(self, channel: str, data: bytes) -> None:
        msg = self._cls.decode(data)
        quat = rpy_to_quat_xyzw(*[float(v) for v in msg.rpy])
        self._cb(np.array(msg.xyz[:2], dtype=np.float64),
                 np.array(msg.vBody, dtype=np.float64),
                 np.array(msg.omegaBody, dtype=np.float64),
                 np.array(quat, dtype=np.float64))
