"""冻结低层 ckpt 缺省路径（按 low_level_convention 分派；零 isaaclab）。

常量实现单源在 ``hierarchical_kick.frozen_weights``（顶层零依赖模块，
零 torch/零 rsl_rl——train.py 的 argparse 段在 AppLauncher 之前必须零
torch）：本模块保留 re-export 兼容 post-launch 消费面
（frozen_loco_action / 测试锚）与既有 import 路径。

路径常量单源（frozen_loco_action / scripts.train 共同消费）：
  - official → weights/velocity_flat_official/model_walk.pt
    （HK-Velocity-Flat-Unitree-Go2-v0 训练产物）
  - legacy   → weights/model_2999.pt（旧契约冻结资产）
"""
from __future__ import annotations

from hierarchical_kick.frozen_weights import (  # noqa: F401
    DEFAULT_LOW_LEVEL_CKPT,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
)

__all__ = [
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL",
    "DEFAULT_LOW_LEVEL_CKPT",
]
