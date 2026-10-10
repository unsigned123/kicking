"""冻结低层 ckpt 缺省路径（按 low_level_convention 分派；零 torch/零
isaaclab/零 rsl_rl——**pre-launch 可 import**）。

常量落位顶层零依赖模块：train.py 的 argparse 段在 AppLauncher 之前必须
零 torch（Kit 对 preloaded torch 不容忍——pxr 预加载冲突 + OpenBLAS×fork
segfault）；``core/weights`` 保持单源 re-export 兼容旧消费面。

路径常量单源（scripts.train 缺省分派 / core.weights re-export 共同消费）：
  - official → weights/velocity_flat_official/model_walk.pt
    （HK-Velocity-Flat-Unitree-Go2-v0 训练产物，官方 agents cfg 结构
    MLP[128,128,128] ELU）
  - legacy   → weights/model_2999.pt（旧契约冻结资产）
"""
from __future__ import annotations

import os

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOW_LEVEL_CKPT_OFFICIAL = os.path.join(
    _PACKAGE_ROOT, "weights", "velocity_flat_official", "model_walk.pt")
"""官方口径低层 ckpt 缺省路径（velocity-flat 官方口径训练产物）。"""

DEFAULT_LOW_LEVEL_CKPT = os.path.join(
    _PACKAGE_ROOT, "weights", "model_2999.pt")
"""legacy 低层 ckpt 缺省路径（model_2999.pt；旧契约资产）。"""


__all__ = [
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL",
    "DEFAULT_LOW_LEVEL_CKPT",
]
