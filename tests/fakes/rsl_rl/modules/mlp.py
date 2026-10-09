"""fake rsl_rl.modules.mlp —— rsl_rl 5.1.0 modules/mlp.py 逐字复刻（纯 torch）。"""
from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-Layer Perceptron（5.1.0 复刻：-1 隐层推断 / tuple 输出 reshape）。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)