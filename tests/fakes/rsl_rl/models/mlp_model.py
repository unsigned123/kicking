"""fake rsl_rl.models.mlp_model —— 5.1.0 复刻（剥 tensordict 依赖，obs 用 dict）。"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model（5.1.0 接口复刻；obs 为 dict-like，``obs["group"]`` 取张量）。"""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: dict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(self, obs, masks=None, hidden_state=HiddenState, stochastic_output=False) -> torch.Tensor:
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(self, obs, masks=None, hidden_state=HiddenState) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        latent = self.obs_normalizer(latent)
        return latent

    def reset(self, dones=None, hidden_state=HiddenState) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: dict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)

    def _get_obs_dim(self, obs, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT（本工程导出路径未用，留接口对齐）。"""

    def __init__(self, model: MLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX（本工程导出路径未用，留接口对齐）。"""

    is_recurrent: bool = False

    def __init__(self, model: MLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)