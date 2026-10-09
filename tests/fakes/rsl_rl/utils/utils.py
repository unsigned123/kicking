"""fake rsl_rl.utils —— 5.1.0 utils/utils.py 纯 torch 复刻（剥 tensordict）。"""
from __future__ import annotations

import importlib
import pkgutil
import torch
import warnings
from typing import Any, Callable

import rsl_rl


def get_param(param: Any, idx: int) -> Any:
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    act_dict = {
        "elu": torch.nn.ELU(), "selu": torch.nn.SELU(), "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(), "lrelu": torch.nn.LeakyReLU(), "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(), "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(), "swish": torch.nn.SiLU(), "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }
    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """5.1.0 复刻：callable 直返 / ``module.path:Attr`` / ``module.path.Attr`` / 简单名。"""
    if callable(callable_or_name):
        return callable_or_name
    if not isinstance(callable_or_name, str):
        raise TypeError(f"Expected callable or string, got {type(callable_or_name)}")

    if ":" in callable_or_name:
        module_path, attr_path = callable_or_name.rsplit(":", 1)
        module = importlib.import_module(module_path)
        obj = module
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj

    if "." in callable_or_name:
        parts = callable_or_name.split(".")
        module_found = False
        for i in range(len(parts) - 1, 0, -1):
            module_path = ".".join(parts[:i])
            attr_parts = parts[i:]
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                continue
            module_found = True
            try:
                obj = module
                for attr in attr_parts:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        if module_found:
            raise AttributeError(f"Could not resolve '{callable_or_name}': attribute not found in module")
        raise ImportError(f"Could not resolve '{callable_or_name}': no valid module.attr split found")

    for _, module_name, _ in pkgutil.iter_modules(rsl_rl.__path__, "rsl_rl."):
        module = importlib.import_module(module_name)
        if hasattr(module, callable_or_name):
            return getattr(module, callable_or_name)
    raise ValueError(
        f"Could not resolve '{callable_or_name}'. Use qualified name like 'module.path:ClassName' "
        f"or pass the class directly."
    )


def resolve_obs_groups(obs, obs_groups: dict[str, list[str]], default_sets: list[str]) -> dict[str, list[str]]:
    """5.1.0 复刻：校验 + 缺省观测集推导（obs 为 dict 即可）。"""
    if len(obs_groups) == 0:
        warnings.warn("The observation configuration dictionary 'obs_groups' is empty and thus likely not configured.")
    else:
        for set_name, groups in obs_groups.items():
            if len(groups) == 0:
                raise ValueError(f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list.")
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                        f" environment. Available observations from the environment: {list(obs.keys())}"
                    )
    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the "
                    f"'{default_set_name}' key. As an observation group with the name '{default_set_name}' was found, "
                    f"this is assumed to be the appropriate observation."
                )
            elif "policy" in obs:
                obs_groups[default_set_name] = ["policy"]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' does not contain the "
                    f"'{default_set_name}' key. As an observation group with the name 'policy' was found, this is "
                    f"assumed to be the appropriate observation."
                )
            else:
                raise ValueError(
                    f"The observation configuration dictionary 'obs_groups' does not contain the "
                    f"'{default_set_name}' key and no suitable observation could be found."
                )
    return obs_groups


def unpad_trajectories(trajectories, masks: torch.Tensor):
    """5.1.0 复刻：dict 版本 split_and_pad 的逆操作。"""
    valid_steps = trajectories.transpose(1, 0)[masks.transpose(1, 0)]
    if isinstance(trajectories, dict):
        return {k: v.reshape(-1, trajectories.shape[0]).transpose(1, 0)
                for k, v in valid_steps.items()}
    return valid_steps.view(-1, trajectories.shape[0], *trajectories.shape[2:]).transpose(1, 0)


def split_and_pad_trajectories(tensor, dones: torch.Tensor):
    """5.1.0 复刻（dict 版；本工程非循环模型不用，供接口对齐）。"""
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    if isinstance(tensor, dict):
        padded_trajectories = {}
        for k, v in tensor.items():
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)[:, :-1]
        out = {k: v for k, v in padded_trajectories.items()}
    else:
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        out = torch.nn.utils.rnn.pad_sequence(trajectories)[:, :-1]
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return out, trajectory_masks


def check_nan(obs, rewards, dones):
    """5.1.0 复刻（runner rollout 的 NaN 哨兵）：任一输入含 NaN 即 ValueError。"""
    import math

    def _has_nan(x):
        if isinstance(x, dict):
            return any(_has_nan(v) for v in x.values())
        if isinstance(x, torch.Tensor):
            return bool(torch.isnan(x).any())
        if isinstance(x, (int, float)):
            return math.isnan(float(x))
        return False

    if _has_nan(obs) or _has_nan(rewards) or _has_nan(dones):
        raise ValueError("NaN encountered in rollout (obs/rewards/dones)")