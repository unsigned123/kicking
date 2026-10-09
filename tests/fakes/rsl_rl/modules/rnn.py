"""fake rsl_rl.modules.rnn —— 仅提供 HiddenState 类型别名。"""
from __future__ import annotations

HiddenState = None  # 本工程只用非循环模型：hidden_state 恒为 None