"""hierarchical_kick — U-Kick 真分层面（3 维速度命令高层 + 冻结低层）cmd-only 独立包。

零副作用红线：本 ``__init__`` 只含版本常量——**import 本包不得拉起
isaaclab/gym**（离线测试红线）。gym 注册面在 ``hierarchical_kick.tasks``
（显式 import 才注册）；isaaclab 面在 ``sim/``（entry point 字符串懒加载）。

谱系：自 UT 源树（分支 vae）upper/ 子树按 cmd-only 闭包拆分，
task id ``Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0`` 冻结不变。
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
