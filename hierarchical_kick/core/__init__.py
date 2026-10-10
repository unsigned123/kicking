# ---------------------------------------------------------------------------
# hierarchical_kick.core（包聚合 __init__）— 上层策略核心：通道契约、cmd
# actor、冻结低层构造/装载（low_level_actor）、kick 指标、关节 scale 表。
# ---------------------------------------------------------------------------
"""Command-space hierarchy core: contracts, cmd actor, frozen low-level face."""

from . import (clean_actor, cmd_actor, contracts, kick_metrics,
               low_level_actor, scale_vec)  # noqa: F401

__all__ = ["contracts", "scale_vec", "cmd_actor", "clean_actor",
           "kick_metrics", "low_level_actor"]
