# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.core（包聚合 __init__）
#
# [cmd-only 拆分] 原 UT upper/core 的 decoder/ledger/u31/
# upper_actor/z_pca/z_stats/oracle_teacher/play_actor/direct_actor 消费面
# 已随 ours/e2e 臂拆分移除；冻结低层构造/装载面收拢在 low_level_actor
# （direct_actor + play_actor 合并，闭包新增）。
# ---------------------------------------------------------------------------
"""Command-space hierarchy core: contracts, cmd actor, frozen low-level face."""

from . import (clean_actor, cmd_actor, contracts, hold_math,
               kick_metrics, low_level_actor, scale_vec)  # noqa: F401

__all__ = ["contracts", "scale_vec", "cmd_actor", "clean_actor",
           "kick_metrics", "hold_math", "low_level_actor"]
