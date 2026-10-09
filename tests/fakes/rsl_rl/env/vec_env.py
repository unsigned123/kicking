"""fake rsl_rl.env —— 极简 VecEnv 占位（仅接口名对齐）。"""
from __future__ import annotations


class VecEnv:
    """VecEnv 占位（本工程离线测试不建真 env）。"""

    num_envs: int = 1
    device: str = "cpu"