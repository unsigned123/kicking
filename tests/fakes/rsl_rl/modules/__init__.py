"""fake rsl_rl.modules（5.1.0 纯 torch 复刻）：MLP / 归一化 / 分布 / HiddenState。"""
from __future__ import annotations

from .distribution import (
    Distribution,
    GaussianDistribution,
    HeteroscedasticGaussianDistribution,
)
from .mlp import MLP
from .normalization import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
)
from .rnn import HiddenState

__all__ = [
    "MLP",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
]