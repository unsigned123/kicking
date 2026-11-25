"""训练 ckpt → 纯 torch 策略（无 rsl_rl / isaaclab 依赖）。

rsl_rl 5.x 的 MLPModel 存面：``mlp.0/2/4/6``（Linear+ELU 交替）+ 输出头
``distribution.std_param``。本模块按 hidden 结构重建 nn.Sequential，取确定性
均值输出（部署不采样）。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import torch
from torch import nn

ACTIVATION = nn.ELU


def _mlp_layers(state: dict[str, torch.Tensor]) -> list[str]:
    keys = [k for k in state if k.startswith("mlp.") and k.endswith(".weight")]
    return sorted(keys, key=lambda k: int(k.split(".")[1]))


def infer_hidden_dims(state: dict[str, torch.Tensor]) -> tuple[int, list[int]]:
    """从 state_dict 反推 (obs_dim, hidden_dims)。"""
    weight_keys = _mlp_layers(state)
    if not weight_keys:
        raise KeyError("ckpt 缺 mlp.* 权重（非 MLPModel 存面？）")
    dims = [int(state[weight_keys[0]].shape[1])]
    for k in weight_keys:
        dims.append(int(state[k].shape[0]))
    return dims[0], dims[1:-1]


def build_actor(state: dict[str, torch.Tensor]) -> nn.Sequential:
    obs_dim, hidden = infer_hidden_dims(state)
    weight_keys = _mlp_layers(state)
    out_dim = int(state[weight_keys[-1]].shape[0])
    layers: list[nn.Module] = []
    prev = obs_dim
    for i, key in enumerate(weight_keys):
        idx = int(key.split(".")[1])
        w = state[key]
        b = state[f"mlp.{idx}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(b)
        layers.append(lin)
        if i < len(weight_keys) - 1:
            layers.append(ACTIVATION())
        prev = w.shape[0]
    assert prev == out_dim
    model = nn.Sequential(*layers)
    model.eval()
    return model


class PolicyRuntime:
    """确定性均值策略（[55] → [3] 机体系速度命令）。"""

    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = (blob.get("model_state_dict")
                 or (blob.get("mlp") if isinstance(blob.get("mlp"), dict) else None)
                 or blob)
        if not isinstance(state, dict):
            raise TypeError(f"无法解析 ckpt: {type(state)}")
        self.model = build_actor({k: v.float() for k, v in state.items()
                                  if torch.is_tensor(v)})
        self.obs_dim, self.hidden = infer_hidden_dims(
            {k: v for k, v in state.items() if torch.is_tensor(v)})
        self.device = device
        self.model.to(device)
        self.last_action_cmd = np.zeros(3, dtype=np.float64)

    @staticmethod
    def md5(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @torch.inference_mode()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"obs 维 {obs.shape} != 策略期望 {self.obs_dim}")
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device)
        cmd = self.model(x.unsqueeze(0))[0].cpu().numpy().astype(np.float64)
        self.last_action_cmd = cmd
        return cmd


def export_ckpt(src: str, dst: str) -> dict[str, Any]:
    """训练 ckpt → 纯权重 .pt + meta（部署机无需 rsl_rl）。"""
    blob = torch.load(src, map_location="cpu", weights_only=False)
    state = blob.get("model_state_dict") or blob
    keep = {k: v for k, v in state.items()
            if torch.is_tensor(v) and k.startswith("mlp.")}
    with open(dst, "wb") as f:
        torch.save({"mlp": keep}, f)
    obs_dim, hidden = infer_hidden_dims(keep)
    meta = {"source": os.path.basename(src), "md5": PolicyRuntime.md5(src),
            "obs_dim": obs_dim, "hidden_dims": hidden,
            "iter": blob.get("iter")}
    with open(os.path.splitext(dst)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta
