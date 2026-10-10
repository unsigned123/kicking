# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.agents_velocity
#
# 官方口径 velocity-flat 训练面的 rsl_rl cfg（官方 go2 agents cfg 复制适
# 配；isaaclab 面，仅 sim 侧解析）。源 = dog 源树 isaaclab_tasks fork 的
# ``UnitreeGo2RoughPPORunnerCfg`` / ``UnitreeGo2FlatPPORunnerCfg``
# （locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py，官方
# Isaac-Velocity-Flat-Unitree-Go2-v0 注册的同一 cfg）。
#
# 复制适配 diff（相对官方原文，逐处）：
#   A1) experiment_name：官方 "unitree_go2_flat" →
#       "hk_velocity_flat_official"（logs/rsl_rl/ 隔离，不与官方产物混写）；
#   A2) run_name = "velocity_official"（日志目录后缀面）。
# 其余逐字同源：num_steps_per_env=24 / max_iterations=300 / save_interval=50 /
# policy（RslRlPpoActorCriticCfg：init_noise_std=1.0、actor/critic hidden
# [128,128,128] ELU、obs_normalization False）/ algorithm（lr 1e-3 adaptive、
# entropy 0.01、clip 0.2、epochs 5、minibatch 4、γ0.99、λ0.95、kl0.01、
# grad 1.0）。rsl_rl >= 4.0 的 actor/critic 模型 cfg 演绎由
# handle_deprecated_rsl_rl_cfg 完成（train_velocity.py 调用）——旧 policy 面
# 保持官方原文形态。
#
# 零 eager 消费：本模块只被注册 entry point 字符串 / train_velocity.py 拉起。
# ---------------------------------------------------------------------------

"""Official-convention velocity-flat rsl_rl runner cfg (copy-adapted)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HKVelocityOfficialGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """官方 go2 rough PPO cfg 逐字复制（仅 experiment_name/run_name 适配）。"""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hk_velocity_official_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HKVelocityOfficialGo2FlatPPORunnerCfg(HKVelocityOfficialGo2RoughPPORunnerCfg):
    """官方 go2 flat PPO cfg 逐字复制（任务 HK-Velocity-Flat-Unitree-Go2-v0）。

    max_iterations=300 / hidden [128,128,128]——官方 agents cfg 原值（时间
    盒 20 分钟内；低层结构契约 = MLP[128,128,128] ELU，core/
    low_level_actor.OFFICIAL_ACTOR_HIDDEN_DIMS 对账表）。
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "hk_velocity_flat_official"
        self.run_name = "velocity_official"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class HKVelocityOfficialCyberDog2FlatPPORunnerCfg(
        HKVelocityOfficialGo2FlatPPORunnerCfg):
    """cyberdog2 flat PPO cfg（任务 HK-Velocity-Flat-CyberDog2-v0）。

    超参/policy 结构与 go2 flat 完全同参（官方口径低层契约不变：
    MLP[128,128,128] ELU、官方 48 维单帧 obs）；仅 experiment_name 换
    cyberdog2 面隔离（logs/rsl_rl/ 不与 go2 产物混写）。
    """

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "hk_velocity_flat_cyberdog2"
        # 探索降噪（round-5；训练超参，非冻结低层契约）：官方 init_noise_std
        # 1.0 × 动作 scale 0.5 下 cyberdog2（kp25 弱执行器）被初始探索直接
        # 踢倒（13 步团灭，PPO 全缓冲无长存活样本，站桩/行走盆地不可达
        # ——round-1~4 递进诊断记录于 sim/velocity_official_cfg.py）；
        # std 0.5 = 初始目标抖动 ±0.25 rad，站桩盆地可达。
        self.policy.init_noise_std = 0.5


__all__ = [
    "HKVelocityOfficialGo2RoughPPORunnerCfg",
    "HKVelocityOfficialGo2FlatPPORunnerCfg",
    "HKVelocityOfficialCyberDog2FlatPPORunnerCfg",
]
