"""关节命名 / 顺序 / 默认角 —— 训练侧与部署侧的单一映射真相源。

训练侧（isaaclab 3.0rc1 PhysX，BFS 分层序）::

    [FR_abad, FL_abad, RR_abad, RL_abad,
     FR_hip,  FL_hip,  RR_hip,  RL_hip,
     FR_knee, FL_knee, RR_knee, RL_knee]

部署侧（CyberDog2 URDF / SDK 序，腿分组三连）::

    [FR_abad, FR_hip, FR_knee,
     FL_abad, FL_hip, FL_knee,
     RR_abad, RR_hip, RR_knee,
     RL_abad, RL_hip, RL_knee]

本模块生成 `sdk_to_train` 置换（train[i] = sdk[sdk_to_train[i]]）与训练序
default_q。数值与 hierarchical_kick/robots/cyberdog2.py 的训练装配同源
（xb7 快照：abad ∓0.10 / hip F0.80 R1.00 / knee −1.50）。
"""

from __future__ import annotations

from typing import Sequence

TRAIN_JOINT_ORDER: tuple[str, ...] = (
    "FR_abad", "FL_abad", "RR_abad", "RL_abad",
    "FR_hip", "FL_hip", "RR_hip", "RL_hip",
    "FR_knee", "FL_knee", "RR_knee", "RL_knee",
)

SDK_JOINT_ORDER: tuple[str, ...] = (
    "FR_abad", "FR_hip", "FR_knee",
    "FL_abad", "FL_hip", "FL_knee",
    "RR_abad", "RR_hip", "RR_knee",
    "RL_abad", "RL_hip", "RL_knee",
)

# 训练序默认关节角（rad），与 xiaomi 快照 / 训练装配一致
TRAIN_DEFAULT_Q: tuple[float, ...] = (
    -0.10, 0.10, -0.10, 0.10,
    0.80, 0.80, 1.00, 1.00,
    -1.50, -1.50, -1.50, -1.50,
)

# 官方口径低层动作 → 关节目标的均匀 scale（default + 0.5·a）
OFFICIAL_ACTION_SCALE = 0.5

NUM_JOINTS = 12


def sdk_to_train_permutation() -> list[int]:
    """返回长度 12 的索引表：train[i] = sdk[perm[i]]。"""
    index = {name: i for i, name in enumerate(SDK_JOINT_ORDER)}
    return [index[name] for name in TRAIN_JOINT_ORDER]


def reorder_sdk_to_train(values: Sequence[float]) -> list[float]:
    """把 SDK 序的 12 维量重排为训练序。"""
    if len(values) != NUM_JOINTS:
        raise ValueError(f"期望 12 维关节量，实际 {len(values)}")
    perm = sdk_to_train_permutation()
    return [float(values[i]) for i in perm]


def resolve_joint_indices(joint_names: Sequence[str]) -> list[int]:
    """从 /joint_states 的 name 列表解出 SDK 序索引（支持带后缀命名）。

    兼容 ``FR_abad``、``FR_abad_joint``、``FR_ABAD_JOINT`` 三种写法。
    """
    normalized = []
    for name in joint_names:
        key = name.strip()
        if key.endswith("_joint"):
            key = key[: -len("_joint")]
        normalized.append(key.lower())
    index = {}
    for i, name in enumerate(normalized):
        index[name] = i
    order: list[int] = []
    for expected in SDK_JOINT_ORDER:
        if expected.lower() not in index:
            raise KeyError(
                f"/joint_states 缺关节 {expected}（实际: {list(joint_names)}）")
        order.append(index[expected.lower()])
    return order
