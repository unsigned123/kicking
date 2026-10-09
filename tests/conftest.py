"""pytest fixtures：离线（无 Isaac Sim）测试（cmd-only 拆分版）。

sys.path 注入仓库根（``hierarchical_kick`` 可作为顶层包导入；其
``__init__`` 零副作用，gym 注册面 ``hierarchical_kick.tasks`` 在无
gymnasium 时自动跳过）。

rsl_rl 桩：本地无真实 rsl_rl 时把 5.1.0 的纯 torch 桩覆盖包插到 sys.path
最前（服务器有真实 rsl_rl 时本包不会被用——与 UT conftest 同款断言）。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)                    # hierarchical_kick 仓库根
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 离线 needs：本地无 rsl_rl → 纯 torch 桩插最前（真实库在场时本包不生效）
_FAKES = os.path.join(_HERE, "fakes")
if _FAKES not in sys.path:
    sys.path.insert(0, _FAKES)
import importlib as _impl  # noqa: E402
import rsl_rl  # noqa: E402

print(f"[conftest] rsl_rl 来源: {rsl_rl.__file__}  (fake={os.path.realpath(_FAKES)})")
assert os.path.realpath(_FAKES) in os.path.realpath(rsl_rl.__file__), (
    f"本地测试必须用 rsl_rl 桩（{os.path.realpath(_FAKES)}）而非真实库: {rsl_rl.__file__}")
