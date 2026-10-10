"""Robot profiles（cmd-only 闭包：go2/cyberdog2 常量 + profile dataclass，零 isaaclab）。

robots 常量面（各本体常量 + 通用 profile）+ 注册表分派（
``make_robot_profile``：name ∈ {go2, cyberdog2}，Unknown fail-fast——
工厂内惰性 import isaaclab，常量/分派面离线可用）。
"""
from __future__ import annotations

from typing import Dict, Optional

from .cyberdog2 import (  # noqa: F401
    CYBERDOG2_BODY_PARTS,
    CYBERDOG2_DEFAULT_Q,
    CYBERDOG2_FOOT_BODIES,
    CYBERDOG2_JOINT_ORDER_IL,
    CYBERDOG2_JOINT_ORDER_SDK,
    make_cyberdog2_track_profile,
)
from .go2 import (  # noqa: F401
    GO2_BODY_PARTS,
    GO2_DEFAULT_Q,
    GO2_FOOT_BODIES,
    GO2_JOINT_ORDER_IL,
    make_go2_track_profile,
)
from .robot_profile import RobotProfile  # noqa: F401

# 注册表（name → profile 工厂；分派单源，velocity cfg / scripts 共用）
_PROFILE_FACTORIES = {
    "go2": make_go2_track_profile,
    "cyberdog2": make_cyberdog2_track_profile,
}

# 注册表名面（离线契约锚：tasks/train_velocity --robot choices 同源）
ROBOT_PROFILE_NAMES = tuple(sorted(_PROFILE_FACTORIES))


def make_robot_profile(
    name: str,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    body_parts: Optional[Dict] = None,
) -> RobotProfile:
    """按注册名构建 RobotProfile（Unknown fail-fast；工厂内惰性拉 isaaclab）。"""
    if name not in _PROFILE_FACTORIES:
        raise ValueError(
            f"[hkick][robots] Unknown robot profile: {name!r}"
            f"（registered: {list(ROBOT_PROFILE_NAMES)}）")
    return _PROFILE_FACTORIES[name](prim_path=prim_path, body_parts=body_parts)


__all__ = [
    "GO2_JOINT_ORDER_IL",
    "GO2_FOOT_BODIES",
    "GO2_DEFAULT_Q",
    "GO2_BODY_PARTS",
    "make_go2_track_profile",
    "CYBERDOG2_JOINT_ORDER_IL",
    "CYBERDOG2_JOINT_ORDER_SDK",
    "CYBERDOG2_FOOT_BODIES",
    "CYBERDOG2_DEFAULT_Q",
    "CYBERDOG2_BODY_PARTS",
    "make_cyberdog2_track_profile",
    "ROBOT_PROFILE_NAMES",
    "make_robot_profile",
    "RobotProfile",
]
