"""逐关节 action scale_vec（env 侧语义 scale 的规范值源）。

scale_j = SCALE_COEF(0.25) × effort_j / KP(40)，effort = URDF per-joint
声明值（implicit 无显式 effort_limit 的回退真值，go2.yaml
``urdf_effort: {abad: 23.7, hip: 23.7, knee: 45.43}``）→ abad/hip 0.148125、
knee 0.2839375。

槽位序 = PhysX 序，单一真相源 ``robots/go2.py::GO2_JOINT_ORDER_IL``
（按关节类型分组：hip×4, thigh×4, calf×4——关节序漂移的防线）。
canonical 名 → URDF 名映射恒同 go2.yaml ``joint_map``
（abad→hip_joint / hip→thigh_joint / knee→calf_joint）。
零 isaaclab：robots.go2 常量 + yaml 读值均可离线。
"""
from __future__ import annotations

import os

import torch
import yaml

from ..robots.go2 import GO2_JOINT_ORDER_IL
from .contracts import KP, SCALE_COEF

# go2.yaml 相对包根（hierarchical_kick/config/robots/go2.yaml）
_GO2_YAML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "config", "robots", "go2.yaml"))

# URDF 关节后缀 → canonical part 名（= go2.yaml joint_map 恒等改名关系：
# abad→hip_joint / hip→thigh_joint / knee→calf_joint 的逆向查询面）
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


def init_std_vec(sigma_sem: float = 0.2,
                 yaml_path: str | None = None) -> torch.Tensor:
    """pre-space 逐维探索 σ init = sigma_sem ⊘ scale_vec。"""
    return sigma_sem / go2_action_scale_vec(yaml_path)


__all__ = [
    "load_go2_yaml", "go2_action_scale_table", "go2_action_scale_vec",
    "go2_joint_names_slot_order", "init_std_vec",
]
