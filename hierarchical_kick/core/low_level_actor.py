# ---------------------------------------------------------------------------
# hierarchical_kick.core.low_level_actor — 冻结低层 actor 单源构造与 ckpt
# 装载（零 isaaclab）。唯一消费方 = env 侧冻结动作 term
# （sim/mdp/frozen_loco_action.py）。
#
# 契约：构造期守卫 fail-fast / σ_sem⊘scale_vec 逐维 std / ckpt 键面兼容
# （rsl_rl 5.1.0 ``actor_state_dict`` 分离存面 + 旧 ``model_state_dict``
# 合一面）。rsl_rl 模型面离线经 tests/fakes/rsl_rl 桩直测。
# ---------------------------------------------------------------------------
"""Frozen low-level actor: single-source construction + ckpt load.

零 isaaclab：rsl_rl 模型面（离线经 tests/fakes/rsl_rl 桩）。
"""
from __future__ import annotations

import os

import torch

from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from .contracts import ACTION_DIM, OBS_DIM, SIGMA_SEM
from .scale_vec import go2_action_scale_vec

UPPER_DIRECT_ACTOR_CLASS_PATH = \
    "hierarchical_kick.core.low_level_actor:UpperDirectActor"
"""agents cfg ``class_name`` 点分路径（resolve_callable 冒号格式）。"""

OFFICIAL_ACTOR_HIDDEN_DIMS = [128, 128, 128]
"""官方 velocity-flat 口径低层 MLP 结构（官方 go2 agents cfg
UnitreeGo2FlatPPORunnerCfg.policy.actor_hidden_dims；ckpt 侧从 state_dict
反推后与本表对账——结构漂移 fail-fast）。"""


def resolve_actor_state_dict(ckpt, *, device=None) -> tuple[dict, dict]:
    """ckpt（路径或已 load 的 dict）→ (完整 dict, actor state_dict)。

    键面兼容：rsl_rl 5.1.0 分离存面（``actor_state_dict``，官方 train
    存面）与旧合一面（``model_state_dict``）；其余键面
    fail-fast KeyError。零装载（纯解析面——结构校验/装载分步走）。
    """
    if isinstance(ckpt, (str, os.PathLike)):
        ckpt = torch.load(ckpt, map_location=device or "cpu",
                          weights_only=False)
    if "actor_state_dict" in ckpt:
        return ckpt, ckpt["actor_state_dict"]
    if "model_state_dict" in ckpt:
        return ckpt, ckpt["model_state_dict"]
    raise KeyError(f"[upper] ckpt 键面不识别: {sorted(ckpt.keys())[:8]}")


def infer_actor_hidden_dims(actor_sd: dict) -> list[int]:
    """从 actor state_dict 反推 MLP hidden dims（``mlp.<i>.weight`` 形状）。

    rsl_rl 5.1.0 MLP = nn.Sequential（Linear/ELU 交错，Linear 位于偶数
    下标）：hidden dims = 除末层外各 Linear 的 out_features；末层 out =
    动作维（一并校验 = contracts.ACTION_DIM）。非 MLP 键面 fail-fast。
    """
    idxs = sorted(
        int(k.split(".")[1]) for k in actor_sd
        if k.startswith("mlp.") and k.endswith(".weight"))
    if not idxs or idxs != list(range(0, 2 * len(idxs), 2)):
        raise KeyError(f"[upper] ckpt 键面非 rsl_rl MLP 形状: "
                       f"{sorted(k for k in actor_sd if k.startswith('mlp.'))}")
    dims = [tuple(actor_sd[f"mlp.{i}.weight"].shape) for i in idxs]
    hidden = [d[0] for d in dims[:-1]]
    if dims[-1][0] != ACTION_DIM:
        raise ValueError(f"[upper] ckpt 末层输出 {dims[-1][0]} != 动作维 "
                         f"{ACTION_DIM}")
    for d in dims[1:-1]:
        if d[1] != hidden[0] and d[1] not in hidden:
            raise ValueError(f"[upper] ckpt MLP 层间维不闭合: {dims}")
    return hidden


def validate_official_actor_structure(actor_sd: dict,
                                      *, expect_hidden=None) -> list[int]:
    """官方口径 ckpt 结构校验：sd 反推 hidden dims == 官方 agents cfg 表。

    expect_hidden 缺省 = OFFICIAL_ACTOR_HIDDEN_DIMS（官方 go2 flat
    [128,128,128]）；漂移即 ValueError（防止把异构 ckpt 静默装进官方契约）。
    """
    hidden = infer_actor_hidden_dims(actor_sd)
    expect = list(OFFICIAL_ACTOR_HIDDEN_DIMS if expect_hidden is None
                  else expect_hidden)
    if hidden != expect:
        raise ValueError(
            f"[upper] 官方口径 ckpt 结构漂移: sd 反推 {hidden} != 官方 "
            f"agents cfg {expect}")
    return hidden


class UpperDirectActor(MLPModel):
    """冻结低层 π_direct：policy obs（契约参数化）── MLP[128,128,128] ELU
    ──► mean12 ──► Gaussian(12)。

    MLP 直出 12 关节 pre-scale 偏移（velocity 面 obs 48 / kick 面 55）。
    本包内仅作**冻结低层**消费（frozen_loco_action 构造期
    装载 + 冻结三件套），不作为可训上层 actor。
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(128, 128, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        *,
        sigma_sem: float = SIGMA_SEM,
        obs_dim_contract: int = OBS_DIM,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.sigma_sem = float(sigma_sem)
        # obs 维契约参数化（velocity 面缺省 48 行为零变化；kick 面由
        # cfg 显式传 ball_mdp.KICK_OBS_DIM=55）
        self.obs_dim_contract = int(obs_dim_contract)
        self._guards(obs_set, obs_normalization, output_dim)
        # 逐维探索 σ init = σ_sem ⊘ scale_vec（与上层 actor 构造期
        # 同一表达式——PhysX 槽位序，abad/hip ≈1.3503 / knee ≈0.7044）
        scale_vec = go2_action_scale_vec()
        self.register_buffer("action_scale_vec", scale_vec,
                             persistent=False)
        with torch.no_grad():
            self.distribution.std_param.data.copy_(
                float(sigma_sem) / scale_vec
                .to(self.distribution.std_param.device))

    # ------------------------------------------------------------------ #
    def _guards(self, obs_set: str, obs_normalization: bool,
                output_dim: int) -> None:
        """构造期守卫（fail-fast）。"""
        # 1) 仅 actor 面 + 单 policy 组（obs 与上层同源的结构前提）
        if obs_set != "actor":
            raise ValueError(f"[upper] UpperDirectActor 仅限 actor 面: "
                             f"{obs_set!r}")
        if list(self.obs_groups) != ["policy"]:
            raise ValueError(
                f"[upper] actor obs_groups 须 ['policy']: "
                f"{self.obs_groups!r}")
        # 2) obs_normalization 恒 False（同 obs 面，无双归一）
        if obs_normalization:
            raise ValueError(
                "[upper] obs_normalization 必须 False（同 obs 面）")
        # 3) 分布必须 12 维 Gaussian 且 output_dim == 动作维
        if not isinstance(self.distribution, GaussianDistribution):
            raise ValueError(
                f"[upper] 分布必须 Gaussian: {type(self.distribution)!r}")
        if output_dim != ACTION_DIM or self.distribution.output_dim \
                != ACTION_DIM:
            raise ValueError(
                f"[upper] output_dim 必须 == {ACTION_DIM}: {output_dim} / "
                f"{self.distribution.output_dim}")
        # 4) obs 维契约（参数化）：velocity 面缺省 48 = proprio 42 +
        #    base_lin_vel 3 + cmd 3；kick 面由 cfg 显式传 55。
        if self.obs_dim != self.obs_dim_contract:
            raise ValueError(
                f"[upper] obs 维契约不符: obs_dim={self.obs_dim} != "
                f"obs_dim_contract={self.obs_dim_contract}"
                f"（velocity 面契约 {OBS_DIM}=proprio42+lin3+cmd3；kick 面"
                f"应显式传 55=ball_mdp.KICK_OBS_DIM）")

    # ------------------------------------------------------------------ #
    def forward(self, obs, masks=None, hidden_state=None,
                stochastic_output=True) -> torch.Tensor:
        x = obs["policy"]
        if masks is not None and not self.is_recurrent:
            x = unpad_trajectories(x, masks)
        # obs_normalization 恒 False → Identity，显式直过（守卫 2 锁定）
        x = self.obs_normalizer(x)
        mean = self.mlp(x)                    # (N,12) pre-scale 偏移直出
        if self.distribution is not None:
            self.distribution.update(mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean


def upper_direct_actor_kwargs(actor_cfg) -> dict:
    """runner cfg 的 actor 面 → :class:`UpperDirectActor` ctor kwargs。

    单源推导面（与官方链 handle_deprecated 后 ``to_dict()`` → PPO
    ``**cfg["actor"]`` 逐键逐值相等）。actor_cfg duck-typing：读
    hidden_dims / activation / obs_normalization / distribution_cfg
    (.class_name/.init_std/.std_type) / sigma_sem / obs_dim_contract 属性。
    """
    d = actor_cfg.distribution_cfg
    return dict(
        hidden_dims=list(actor_cfg.hidden_dims),
        activation=actor_cfg.activation,
        obs_normalization=bool(actor_cfg.obs_normalization),
        distribution_cfg={"class_name": d.class_name,
                          "init_std": float(d.init_std),
                          "std_type": getattr(d, "std_type", "scalar")},
        sigma_sem=float(actor_cfg.sigma_sem),
        obs_dim_contract=int(getattr(actor_cfg, "obs_dim_contract", OBS_DIM)),
    )


def build_upper_direct_actor(dummy_obs, actor_cfg=None, *, device=None,
                             klass=UpperDirectActor,
                             obs_groups=None, obs_set="actor",
                             output_dim=ACTION_DIM, **kw) -> UpperDirectActor:
    """从 cfg 构造 UpperDirectActor（kwargs 全经
    :func:`upper_direct_actor_kwargs`；``**kw`` 透传 ctor）。"""
    kwargs = (upper_direct_actor_kwargs(actor_cfg)
              if actor_cfg is not None else {})
    kwargs.update(kw)
    actor = klass(dummy_obs, obs_groups or {"actor": ["policy"]}, obs_set,
                  output_dim, **kwargs)
    if device is not None:
        actor = actor.to(device)
    return actor


def load_upper_actor_ckpt(actor, ckpt, *, device=None,
                          strict=True) -> dict:
    """ckpt（路径或已 load 的 dict）→ actor strict 装载；返回已 load 的 dict。

    键面兼容经 :func:`resolve_actor_state_dict`（分离存面/合一面）。
    """
    ckpt, sd = resolve_actor_state_dict(ckpt, device=device)
    actor.load_state_dict(sd, strict=strict)
    return ckpt


__all__ = [
    "UpperDirectActor", "UPPER_DIRECT_ACTOR_CLASS_PATH",
    "OFFICIAL_ACTOR_HIDDEN_DIMS",
    "upper_direct_actor_kwargs", "build_upper_direct_actor",
    "resolve_actor_state_dict", "infer_actor_hidden_dims",
    "validate_official_actor_structure", "load_upper_actor_ckpt",
]
