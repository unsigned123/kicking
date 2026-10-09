"""冻结低层 ckpt 缺省路径（按 low_level_convention 分派；零 torch/零
isaaclab/零 rsl_rl——**pre-launch 可 import**）。

[isaaclab 3.0 迁移] 常量自 ``core/weights`` 上移至本模块：0.54 栈容忍
pre-launch 经 ``core/__init__`` eager 拉起 torch/rsl_rl，isaacsim 6.1
（Kit 110）不容忍——preloaded torch 触发 AppLauncher
``_initialize_preloaded_torch_cuda`` 特殊路径 → pxr 预加载冲突 + OpenBLAS
× fork 竞态 segfault。train.py 的 argparse 段必须零 torch，故常量落位
顶层零依赖模块；``core/weights`` 保持单源 re-export 兼容旧消费面。

路径常量单源（scripts.train 缺省分派 / core.weights re-export 共同消费）：
  - official → weights/velocity_flat_official/model_walk.pt
    （HK-Velocity-Flat-Unitree-Go2-v0 训练产物；2026-10-09 rc1 runtime 修复后
    重训 1000it@4096env——Mean reward 32.87、探针 v_body 0.37m/s/步态振荡
    99 翻转/推球 1.32m 过行走门，替代不会走的 model_299.pt（md5 d14f4e6a，
    站桩 2.74 留档）；官方 agents cfg 结构 MLP[128,128,128] ELU 不变）
  - legacy   → weights/model_2999.pt（U4b B 臂旧契约冻结资产）
"""
from __future__ import annotations

import os

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LOW_LEVEL_CKPT_OFFICIAL = os.path.join(
    _PACKAGE_ROOT, "weights", "velocity_flat_official", "model_walk.pt")
"""官方口径低层 ckpt 缺省路径（velocity-flat 官方口径训练产物）。"""

DEFAULT_LOW_LEVEL_CKPT = os.path.join(
    _PACKAGE_ROOT, "weights", "model_2999.pt")
"""legacy 低层 ckpt 缺省路径（U4b B 臂 model_2999.pt；旧契约资产）。"""


__all__ = [
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL",
    "DEFAULT_LOW_LEVEL_CKPT",
]
