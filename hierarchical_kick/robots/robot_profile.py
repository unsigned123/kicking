"""RobotProfile dataclass + registry 装配规范（universal tracking 版）。

接口照抄 data_collect/robots/base.py 的 RobotProfile 范式，差异：
  - 顶层**不** import isaaclab（ArticulationCfg 用惰性类型标注，isalloclab 导入
    放在具体机器人的工厂函数内），使本模块（及 registry/go2.py 的常量）可离线
    import——离线 pytest 与 scripts/print_obs_dims.py 直接复用；
  - 增加 ``name``（registry 名）与 ``joint_order``=Isaac Lab 实际关节序（probed；
    用于与 clip/motion 序的按名置换校验，对应 TrackCommand 的 R5 逻辑）。

多本体注册扩展（仅 embodiments 路径填充；单本体 go2 路径保持
None = 无规范层，语义不变）：
  - ``canonical_joint_order``: 规范关节序（腿分组三连 FL/FR/RL/RR ×
    {abad,hip,knee}，如 ``FL_abad``）；obs 规范序 / 参考关节对齐 /
    action 分发三处共用的锚点序；
  - ``joint_to_canonical`` / ``canonical_to_joint``: 本体关节名 ↔ 规范关节名
    的双向映射（ embodiments yaml 的 joint_map 展开产物；按名置换，方向
    断言见 tests/test_m23_embodiments.py，R5 教训）。

字段语义（与 data_collect 完全一致）：
  - joint_order: Isaac Lab 实际关节序（PhysX 按关节类型分组）
  - default_joint_pos: joint 名 -> 默认站姿角（policy q0，按名）
  - foot_bodies: 足端 body 名（policy 腿序 FL,FR,RL,RR）
  - body_parts: feet/base/joints/hips[/ankles] 可配 dict（配置覆写入口）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg


@dataclass
class RobotProfile:
    """开箱即用的机器人静态描述。"""

    name: str
    articulation_cfg: Optional["ArticulationCfg"] = None
    joint_order: Optional[List[str]] = None
    default_joint_pos: Optional[Dict[str, float]] = None
    foot_bodies: Optional[List[str]] = None
    body_parts: Optional[Dict[str, Any]] = None
    # 规范关节层（embodiments 路径填充；单本体路径 None）
    canonical_joint_order: Optional[List[str]] = None
    joint_to_canonical: Optional[Dict[str, str]] = None
    canonical_to_joint: Optional[Dict[str, str]] = None
    # 存档用元数据（可选；写入训练 meta/日志）
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_order or [])

    @property
    def num_feet(self) -> int:
        return len(self.foot_bodies or [])


__all__ = ["RobotProfile"]