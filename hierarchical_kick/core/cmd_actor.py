# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.core.cmd_actor
#
# U-Kick 对照臂「真分层」（arm D，docs/UKICK_BALL_ENV_DESIGN.md §0 谱系
# 扩展）：上层 RL 动作空间 = 3 维机体系速度命令 [vx,vy,wz]。
# 本文件新增独立分支——三臂（e2e/ours/cmd）与 velocity 面**零行为变化**
# （arm D 全部走新增文件/新增分支纪律）。
#
# [st-vae 迁移 P3 面注] 三臂换代后（Q7：e2e/ours/cmd）：ours 臂动作面 =
# 32 维 z 段 @5Hz（env 侧 hold+插值+冻结 st-vae decoder），本臂保持 50Hz
# 逐步命令接口（冻结低层每子步重观测——时间抽象差异 = 动作接口本体的
# 一部分，frozen_loco_action 头注同款裁定；"ours 5Hz vs cmd 50Hz"的频率
# 非对称是否需要 arm D hold 变体 = 待 owner 裁定项，见 P3 报告）。
#
# 与 :class:`UpperDirectActor`（e2e 臂直出）的公平性锚（逐项同源）：
#   - 同 obs 面：policy 组契约守卫同款（obs_set=actor / 单 policy 组 /
#     obs_normalization=False / ``obs_dim_contract`` 参数化——arm D 恒 kick
#     面 55 维，缺省即 KICK_POLICY_OBS_DIM=55）；
#   - 同 trunk：hidden_dims=[128,128,128] ELU（cfg 层公平性测试钉）；
#   - 同探索：Gaussian 输出分布，init std = σ_sem 同值（SIGMA_SEM=0.2；
#     3 维命令空间**无 scale_vec**——12 维关节面 σ=σ_sem⊘scale_vec 的逐维
#     公式不适用，命令通道无量纲缩放 → 标量 σ_sem 逐维填充）；
#     [P3] σ cap 机制参数化在位（``sigma_cap`` + :meth:`cap_sigma_`——与
#     ours 段空间臂同机制；缺省 None 关闭 = 命令臂 PPO 面逐位现状，开帽
#     与否属实验动作）。
# 唯一差异（动作接口本体）：output 3 维（命令空间）vs 32 维（z 段）vs
# 12 维（pre-scale 关节偏移）——arm D 动作不进关节，由 env 侧
# FrozenLocoJointPositionAction 消费。
#
# ckpt 键面承诺：stock MLPModel 面（``mlp.*`` + ``distribution.std_param``）；
# 零 decoder/z/scale_vec 面（arm D 上层无冻结资产消费）。
# 零 isaaclab：rsl_rl 模型面（离线经 tests/fakes/rsl_rl 桩）。
# ---------------------------------------------------------------------------
"""UpperCmdActor：U-Kick arm D「真分层」上层命令策略（MLP 55 → 3 维速度命令）。

对照本体：动作 = [vx,vy,wz] 机体系命令（Gaussian(3)，σ init=σ_sem 同值），
与三臂同 obs/奖励/终止/PPO——唯一变量是动作接口（真分层 vs 12 关节直出/
z64 解码）。
"""
from __future__ import annotations

import torch

from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

from .clean_actor import KICK_POLICY_OBS_DIM
from .contracts import SIGMA_SEM

UPPER_CMD_ACTOR_CLASS_PATH = \
    "hierarchical_kick.core.cmd_actor:UpperCmdActor"
"""agents cfg ``class_name`` 点分路径（resolve_callable 冒号格式）。"""

CMD_ACTION_DIM = 3
"""arm D 上层动作维 = 机体系速度命令 [vx,vy,wz]。"""


class UpperCmdActor(MLPModel):
    """π_cmd：policy obs 55 ── MLP[128,128,128] ELU ──► mean3 ──► Gaussian(3)。

    与 UpperDirectActor 同构减 scale_vec 段（3 维命令空间无逐关节缩放）；
    ``stochastic_output=False`` 路径 = deterministic mean（runner/PPO 面
    同款语义）。对照变量 = 动作接口（真分层命令 vs 12 关节直出 vs 32 维
    z 段 @5Hz）。
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
        sigma_cap: float | None = None,
        obs_dim_contract: int = KICK_POLICY_OBS_DIM,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims=hidden_dims, activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.sigma_sem = float(sigma_sem)
        self.sigma_cap = (None if sigma_cap is None else float(sigma_cap))
        # obs 维契约参数化（arm D 恒 kick 面 55；velocity 面不支持——守卫指路）
        self.obs_dim_contract = int(obs_dim_contract)
        self._guards(obs_set, obs_normalization, output_dim)
        # 探索 σ init = σ_sem 标量填充（3 维命令空间；Gaussian(3) init_std
        # 同值——与三臂 σ_sem 同源公平性锚，见模块头）
        with torch.no_grad():
            self.distribution.std_param.data.fill_(float(sigma_sem))

    # ------------------------------------------------------------------ #
    def cap_sigma_(self, cap: float | None = None) -> int:
        """逐维 clamp 探索 σ ≤ cap（与 UpperActor 同机制；缺省 = 构造期帽）。

        缺省 None = 关闭（命令臂 PPO 面逐位现状）；返回被截断维数。"""
        c = self.sigma_cap if cap is None else (None if cap is None
                                               else float(cap))
        if c is None:
            return 0
        if not (float(c) > 0.0):
            raise ValueError(f"[upper] sigma cap 须正: {c!r}")
        with torch.no_grad():
            std = self.distribution.std_param.data
            exceeded = int((std > float(c)).sum())
            std.clamp_(max=float(c))
        return exceeded

    # ------------------------------------------------------------------ #
    def _guards(self, obs_set: str, obs_normalization: bool,
                output_dim: int) -> None:
        """构造期守卫（direct_actor 同款契约换 3 维命令面；fail-fast）。"""
        # 1) 仅 actor 面 + 单 policy 组（obs 与三臂同源的结构前提）
        if obs_set != "actor":
            raise ValueError(f"[upper] UpperCmdActor 仅限 actor 面: "
                             f"{obs_set!r}")
        if list(self.obs_groups) != ["policy"]:
            raise ValueError(
                f"[upper] actor obs_groups 须 ['policy']: "
                f"{self.obs_groups!r}")
        # 2) obs_normalization 恒 False（三臂同 obs 面，无双归一）
        if obs_normalization:
            raise ValueError(
                "[upper] obs_normalization 必须 False（arm D 与三臂同 obs 面）")
        # 3) 分布必须 Gaussian 且 output_dim == 命令维 3
        if not isinstance(self.distribution, GaussianDistribution):
            raise ValueError(
                f"[upper] 分布必须 Gaussian: {type(self.distribution)!r}")
        if output_dim != CMD_ACTION_DIM or self.distribution.output_dim \
                != CMD_ACTION_DIM:
            raise ValueError(
                f"[upper] output_dim 必须 == {CMD_ACTION_DIM}（arm D 命令空间"
                f" [vx,vy,wz]）: {output_dim} / "
                f"{self.distribution.output_dim}")
        # 4) obs 维契约：恒 kick 面 55（42-prefix+lin3+ball3+goal4+push3）；
        #    velocity 面 48 契约不支持（错误消息带两面数字 + 指路）
        if self.obs_dim != self.obs_dim_contract:
            raise ValueError(
                f"[upper] obs 维契约不符: obs_dim={self.obs_dim} != "
                f"obs_dim_contract={self.obs_dim_contract}"
                f"（arm D 仅 kick 面，契约 "
                f"{KICK_POLICY_OBS_DIM}=proprio42+lin3+ball3+goal4+push3）")

    # ------------------------------------------------------------------ #
    # 前向（direct_actor 同构——命令直出）
    # ------------------------------------------------------------------ #
    def forward(self, obs, masks=None, hidden_state=None,
                stochastic_output=True) -> torch.Tensor:
        x = obs["policy"]
        if masks is not None and not self.is_recurrent:
            x = unpad_trajectories(x, masks)
        # obs_normalization 恒 False → Identity，显式直过（守卫 2 锁定）
        x = self.obs_normalizer(x)
        mean = self.mlp(x)                    # (N,3) [vx,vy,wz] 命令直出
        if self.distribution is not None:
            self.distribution.update(mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean


__all__ = ["UpperCmdActor", "UPPER_CMD_ACTOR_CLASS_PATH", "CMD_ACTION_DIM"]
