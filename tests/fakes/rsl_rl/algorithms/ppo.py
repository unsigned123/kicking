"""fake rsl_rl.algorithms.ppo —— 5.1.0 PPO.__init__ 复刻（属性面与 update 依赖齐）。

本工程 SonicPPO 只继承 __init__ 并整体覆写 update()；离线测试经此桩驱动真实
SonicPPO.update() 复制体（含重建项 / ramp / 码本熵）。rollout 循环
（act/process_env_step/compute_returns）本工程测试不走，留占位。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain

from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class PPO:
    """PPO（5.1.0 构造签名复刻；RND / Symmetry 本桩恒 None）。"""

    actor: MLPModel
    critic: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.symmetry = None

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate)

        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    # -- rollout 循环（本工程测试不走，占位） ---------------------------------- #
    def act(self, obs):  # pragma: no cover
        raise NotImplementedError("fake PPO: act() 仅用于 rollout，离线测试不涉及")

    def process_env_step(self, obs, rewards, dones, extras):  # pragma: no cover
        raise NotImplementedError("fake PPO: process_env_step() 仅用于 rollout")

    def compute_returns(self, obs):  # pragma: no cover
        raise NotImplementedError("fake PPO: compute_returns() 仅用于 rollout")

    # -- 训练模式（供 runner 对齐；测试不直接用） ------------------------------ #
    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_policy(self) -> MLPModel:
        return self._raw_actor

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """5.1.0 PPO.load 复刻：按 load_cfg 恢复模型/优化器，返回是否恢复迭代。

        SonicPPO.save()/load() 覆写依赖本方法与真实 5.1.0 同名接口
        对齐（OnPolicyRunner.load 经 ``alg.load(loaded_dict, load_cfg, strict)``
        调用，return==load_cfg["iteration"] 时才用 ``loaded_dict["iter"]`` 覆写
        runner 的 current_learning_iteration）。
        """
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)