"""冻结低层 ckpt 缺省路径（按 low_level_convention 分派；零 isaaclab）。

路径常量单源（frozen_loco_action / scripts.train 共同消费；argparse 缺省
解析发生在 AppLauncher 之前，故必须零 isaaclab 可 import）：
  - official → weights/velocity_flat_official/model_walk.pt
    （HK-Velocity-Flat-Unitree-Go2-v0 训练产物；2026-10-09 rc1 runtime
    修复后重训 1000it@4096env——Mean reward 32.87、行走门全过（v_body
    0.37m/s / 步态振荡 99 翻转 / 推球 1.32m），替代不会走的 model_299.pt）
  - legacy   → weights/model_2999.pt（U4b B 臂旧契约冻结资产）

[isaaclab 3.0 迁移] 常量实现上移至 ``hierarchical_kick.frozen_weights``
（顶层零依赖模块，零 torch/零 rsl_rl）：本模块位于 ``core`` 包内，import
即经 ``core/__init__`` eager 拉起 torch/rsl_rl——旧 isaacsim 4.x 容忍
pre-launch torch，Kit 110（isaacsim 6.1）不容忍（pxr 预加载冲突 +
OpenBLAS×fork segfault）。scripts.train 的 pre-launch 段改消费
``frozen_weights``；本模块保留 re-export 兼容 post-launch 消费面
（frozen_loco_action / 测试锚）与既有 import 路径。
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
