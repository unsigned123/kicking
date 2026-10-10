"""逐关节 action scale_vec（env 侧语义 scale 的规范值源）。

go2：scale_j = SCALE_COEF(0.25) × effort_j / KP(40)，effort = URDF per-joint
声明值（implicit 无显式 effort_limit 的回退真值，go2.yaml
``urdf_effort: {abad: 23.7, hip: 23.7, knee: 45.43}``）→ abad/hip 0.148125、
knee 0.2839375。
cyberdog2：**xiaomi deploy.yaml 快照 uniform 0.25**（"以快照为准"——公式
0.25×effort/KP = 0.25×12/25 = 0.12 为备选口径，未被快照采用；快照 =
vf_500_benchmark params/deploy.yaml:29 全 12 关节 0.25）。

槽位序 = PhysX 序，单一真相源 ``robots/go2.py::GO2_JOINT_ORDER_IL`` /
``robots/cyberdog2.py::CYBERDOG2_JOINT_ORDER_IL``（按关节类型分组——关节序
漂移的防线）。
canonical 名 → URDF 名映射恒同 go2.yaml ``joint_map``
（abad→hip_joint / hip→thigh_joint / knee→calf_joint；cyberdog2 同名恒等）。
零 isaaclab：robots 常量 + yaml 读值均可离线。
"""
from __future__ import annotations

import os

import torch
import yaml

from ..robots.cyberdog2 import CYBERDOG2_JOINT_ORDER_IL
from ..robots.go2 import GO2_JOINT_ORDER_IL
from .contracts import KP, SCALE_COEF

# go2.yaml / cyberdog2.yaml 相对包根（hierarchical_kick/config/robots/）
_CONFIG_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "robots"))
_GO2_YAML = os.path.join(_CONFIG_ROOT, "go2.yaml")
_CYBERDOG2_YAML = os.path.join(_CONFIG_ROOT, "cyberdog2.yaml")

# cyberdog2 PD 刚度（xiaomi CYBERDOG2_CFG DCMotor stiffness；cyberdog2.yaml
# actuator.stiffness 一致性校验面——单一真相源本常量，yaml 漂移即报错）
CYBERDOG2_KP = 25.0
"""cyberdog2 DCMotor KP（xiaomi 快照；contracts.KP=40 为 go2 专属，不混用）。"""

CYBERDOG2_ACTION_SCALE_SNAPSHOT = 0.25
"""cyberdog2 逐关节 action scale（xiaomi deploy.yaml 快照 uniform 值；
cyberdog2.yaml ``action_scale_value`` 一致性校验面）。"""

# URDF 关节后缀 → canonical part 名（= go2.yaml joint_map 恒等改名关系：
# abad→hip_joint / hip→thigh_joint / knee→calf_joint 的逆向查询面；
# cyberdog2 三段同名 *_abad/_hip/_knee_joint，后缀直接是 part 名）
_SUFFIX_TO_PART = {"hip": "abad", "thigh": "hip", "calf": "knee"}


def load_go2_yaml(path: str | None = None) -> dict:
    """读 go2.yaml 静态元数据（urdf_effort / actuator.stiffness 校验面）。"""
    with open(path or _GO2_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def go2_action_scale_table(yaml_path: str | None = None) -> dict[str, float]:
    """URDF 关节名（PhysX 槽位序）→ scale_j 的有序表（插入序 = 槽位序）。

    读 go2.yaml ``urdf_effort`` 与 ``actuator.stiffness``（KP 缺省恒 40，
    yaml 值仅作一致性校验——contracts.KP 是唯一真相源，yaml 漂移即报错）。
    """
    raw = load_go2_yaml(yaml_path)
    effort = {str(k): float(v) for k, v in raw["urdf_effort"].items()}
    kp = float(raw["actuator"]["stiffness"])
    if kp != KP:
        raise ValueError(f"[upper] go2.yaml actuator.stiffness={kp} != "
                         f"contracts.KP={KP}（契约漂移，fail-fast）")
    table: dict[str, float] = {}
    for name in GO2_JOINT_ORDER_IL:          # PhysX 槽位序（类型分组）
        suffix = name.removesuffix("_joint")
        part = _SUFFIX_TO_PART[suffix.split("_", 1)[1]]
        if part not in effort:
            raise KeyError(f"[upper] go2.yaml urdf_effort 缺 {part}")
        table[name] = SCALE_COEF * effort[part] / kp
    if len(table) != len(GO2_JOINT_ORDER_IL):
        raise ValueError("[upper] scale 表覆盖不全（关节名解析漂移，fail-fast）")
    return table


def go2_action_scale_vec(yaml_path: str | None = None) -> torch.Tensor:
    """PhysX 槽位序 scale_vec (12,) f32：abad/hip 0.148125 ×8、knee 0.2839375 ×4。"""
    table = go2_action_scale_table(yaml_path)
    names = list(table)  # dict 插入序 = 槽位序
    return torch.tensor([table[n] for n in names], dtype=torch.float32)


def go2_joint_names_slot_order(yaml_path: str | None = None) -> list[str]:
    """与 scale_vec 同序的 URDF 关节名（PhysX 槽位序；构造期断言消费）。"""
    return list(go2_action_scale_table(yaml_path))


def cyberdog2_action_scale_table(
        yaml_path: str | None = None) -> dict[str, float]:
    """URDF 关节名（PhysX 槽位序）→ scale_j 的有序表（cyberdog2）。

    **xiaomi deploy 快照 uniform 0.25**（CYBERDOG2_ACTION_SCALE_SNAPSHOT
    单源；yaml ``action_scale_value``/``actuator.stiffness`` 仅一致性校验
    ——漂移即报错）。插入序 = CYBERDOG2_JOINT_ORDER_IL 槽位序。
    """
    raw = load_yaml_file(yaml_path or _CYBERDOG2_YAML)
    kp = float(raw["actuator"]["stiffness"])
    if kp != CYBERDOG2_KP:
        raise ValueError(
            f"[upper] cyberdog2.yaml actuator.stiffness={kp} != "
            f"CYBERDOG2_KP={CYBERDOG2_KP}（契约漂移，fail-fast）")
    scale = float(raw.get("action_scale_value",
                          CYBERDOG2_ACTION_SCALE_SNAPSHOT))
    if scale != CYBERDOG2_ACTION_SCALE_SNAPSHOT:
        raise ValueError(
            f"[upper] cyberdog2.yaml action_scale_value={scale} != 快照 "
            f"{CYBERDOG2_ACTION_SCALE_SNAPSHOT}（契约漂移，fail-fast）")
    return {name: scale for name in CYBERDOG2_JOINT_ORDER_IL}


def cyberdog2_action_scale_vec(
        yaml_path: str | None = None) -> torch.Tensor:
    """PhysX 槽位序 scale_vec (12,) f32：uniform 0.25 ×12（xiaomi 快照）。"""
    table = cyberdog2_action_scale_table(yaml_path)
    return torch.tensor([table[n] for n in CYBERDOG2_JOINT_ORDER_IL],
                        dtype=torch.float32)


def action_scale_table_for(robot: str) -> dict[str, float]:
    """注册名分派（kick cfg 面 go2/cyberdog2 双本体共用的单一入口）。

    name ∈ make_robot_profile 注册表同名（robots.ROBOT_PROFILE_NAMES）；
    Unknown fail-fast。
    """
    if robot == "go2":
        return go2_action_scale_table()
    if robot == "cyberdog2":
        return cyberdog2_action_scale_table()
    raise ValueError(f"[upper] Unknown robot for action scale: {robot!r}"
                     f"（registered: go2, cyberdog2）")


def load_yaml_file(path: str) -> dict:
    """yaml 直读（cyberdog2 表校验面；go2 面保持 load_go2_yaml 别名）。"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_std_vec(sigma_sem: float = 0.2,
                 yaml_path: str | None = None) -> torch.Tensor:
    """pre-space 逐维探索 σ init = sigma_sem ⊘ scale_vec。"""
    return sigma_sem / go2_action_scale_vec(yaml_path)


__all__ = [
    "CYBERDOG2_ACTION_SCALE_SNAPSHOT", "CYBERDOG2_KP",
    "load_go2_yaml", "load_yaml_file", "go2_action_scale_table",
    "go2_action_scale_vec", "go2_joint_names_slot_order",
    "cyberdog2_action_scale_table", "cyberdog2_action_scale_vec",
    "action_scale_table_for", "init_std_vec",
]
