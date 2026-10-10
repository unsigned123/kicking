"""Robot profiles（cmd-only 闭包：go2 常量 + profile dataclass，零 isaaclab）。

robots 常量面（go2 本体 + 通用 profile）
（多本体装配）不进本包闭包；仅保留 go2 单本体消费面。
"""
from __future__ import annotations

from .go2 import (  # noqa: F401
    GO2_BODY_PARTS,
    GO2_DEFAULT_Q,
    GO2_FOOT_BODIES,
    GO2_JOINT_ORDER_IL,
    make_go2_track_profile,
)
from .robot_profile import RobotProfile  # noqa: F401

__all__ = [
    "GO2_JOINT_ORDER_IL",
    "GO2_FOOT_BODIES",
    "GO2_DEFAULT_Q",
    "GO2_BODY_PARTS",
    "make_go2_track_profile",
    "RobotProfile",
]
