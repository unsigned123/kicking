"""冻结低层 ckpt 缺省路径（按 low_level_convention 分派；零 torch/零
isaaclab/零 rsl_rl——**pre-launch 可 import**）。

常量落位顶层零依赖模块：train.py 的 argparse 段在 AppLauncher 之前必须
零 torch（Kit 对 preloaded torch 不容忍——pxr 预加载冲突 + OpenBLAS×fork
segfault）；``core/weights`` 保持单源 re-export 兼容旧消费面。

路径常量单源（scripts.train 缺省分派 / core.weights re-export 共同消费）：
  - official/go2       → weights/velocity_flat_official/model_walk.pt
    （HK-Velocity-Flat-Unitree-Go2-v0 训练产物，官方 agents cfg 结构
    MLP[128,128,128] ELU）
  - official/cyberdog2 → weights/velocity_flat_official/cyberdog2/model_walk.pt
    （HK-Velocity-Flat-CyberDog2-v0 训练产物，同结构官方口径）
  - legacy             → weights/model_2999.pt（旧契约冻结资产）
"""
from __future__ import annotations

import os

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOW_LEVEL_CKPT_OFFICIAL = os.path.join(
    _PACKAGE_ROOT, "weights", "velocity_flat_official", "model_walk.pt")
"""官方口径低层 ckpt 缺省路径（go2；velocity-flat 官方口径训练产物）。"""

DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2 = os.path.join(
    _PACKAGE_ROOT, "weights", "velocity_flat_official", "cyberdog2",
    "model_walk.pt")
"""官方口径低层 ckpt 缺省路径（cyberdog2；HK-Velocity-Flat-CyberDog2-v0
训练产物——行走门过门后落位）。"""

DEFAULT_LOW_LEVEL_CKPT = os.path.join(
    _PACKAGE_ROOT, "weights", "model_2999.pt")
"""legacy 低层 ckpt 缺省路径（model_2999.pt；旧契约资产）。"""


def default_low_level_ckpt_official(robot: str = "go2") -> str:
    """按注册名分派 official 约定缺省 ckpt（Unknown fail-fast）。"""
    if robot == "go2":
        return DEFAULT_LOW_LEVEL_CKPT_OFFICIAL
    if robot == "cyberdog2":
        return DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2
    raise ValueError(f"[hkick][weights] Unknown robot: {robot!r}"
                     "（registered: go2, cyberdog2）")


__all__ = [
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL",
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2",
    "DEFAULT_LOW_LEVEL_CKPT",
    "default_low_level_ckpt_official",
]
