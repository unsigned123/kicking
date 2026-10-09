"""上层 RL 的 rsl_rl runner/模型 cfg（isaaclab 面；仅 sim 侧解析）。

[cmd-only 拆分] 原 UT upper/agents 的 UpperActorCfg
（ours 段空间）/ UpperDirectActorCfg（e2e 直出）/ UpperCleanActorCfg
（旧 hier 复现位）及 e2e/hier/ours 三个 kick runner cfg 已随拆分移除；
仅保留 arm D「真分层」消费面。PPO 超参 = 设计 §1.5 表（与 UT 三臂
逐字同源——公平性纪律：除 actor/experiment_name 外逐字相等）。

零 eager 消费：本模块只被 register/launch 的 entry point 字符串解析拉起。
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from .core.clean_actor import KICK_POLICY_OBS_DIM
from .core.cmd_actor import UPPER_CMD_ACTOR_CLASS_PATH
from .core.contracts import SIGMA_SEM


@configclass
class UpperCmdActorCfg(RslRlMLPModelCfg):
    """U-Kick arm D「真分层」上层命令 actor cfg（3 维 [vx,vy,wz] 输出）。

    与 UT 三臂 actor cfg 公平性同源（arm D 与三臂同 trunk/同探索/同 obs
    面）：hidden_dims/activation/obs_normalization/distribution_cfg/
    sigma_sem 全同。差异仅 class_name（命令 MLP）+ obs 契约缺省面（本 cfg
    仅 kick 面消费——缺省即 55）；无 decoder/z/k_env 面（上层无冻结资产
    消费——冻结低层在 env 侧动作 term）。
    distribution init_std = SIGMA_SEM 同值（3 维命令空间无 scale_vec →
    标量 σ_sem 逐维填充，ctor 内执行）。
    """

    class_name: str = UPPER_CMD_ACTOR_CLASS_PATH
    hidden_dims: list[int] = [128, 128, 128]
    activation: str = "elu"
    obs_normalization: bool = False   # 守卫 2：arm D 与三臂同 obs 面（无归一）
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=SIGMA_SEM)           # 标量名义值；3 维命令空间 σ init = σ_sem
    # --- upper 自定义字段（经 to_dict 透传 UpperCmdActor 构造） ---
    sigma_sem: float = SIGMA_SEM      # 与三臂同值——命令空间探索 σ 同源
    obs_dim_contract: int = KICK_POLICY_OBS_DIM   # arm D 恒 kick 面 55


@configclass
class UpperGo2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """上层 go2 kick-flat runner cfg 基类（experiment 隔离 upper_v0）。

    PPO 超参（U0 §1.5 表）：lr 1e-4 adaptive / γ 0.99 / λ 0.95 / clip 0.2 /
    epochs 5 / minibatch 4 / entropy 0.01（§1.4 v2 实测健康值）/ grad 1.0。
    ⚠️ kick 面训练入口（train.py）以 config/upper_ukick.yaml 同源的
    entropy_coef=0.001 运行期覆写本缺省（UT 三臂同配方，公平面在 PPO
    超参——[entropy 定值 0.001 2026-09-30 三拍]）。

    [cmd-only 拆分] UT 原版本基类 actor = UpperActorCfg（ours 段空间）；
    本包仅存 cmd 面，actor 缺省 = UpperCmdActorCfg()（字段面除 class_name
    与 obs 契约缺省外逐字同源）。
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "upper_v0"      # logs/rsl_rl/upper_v0/ 隔离（§4）
    empirical_normalization = False   # deprecated 字段；归一化走模型侧
    clip_actions: float | None = 10.0  # WTW 坑表：防 action 自反馈爆炸（house 惯例）
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],          # 无特权组：critic 同 55 维 obs（kick env 单组）
    }

    actor = UpperCmdActorCfg()
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,        # 缺省（kick 面训练入口覆写 0.001，见类注）
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,     # adaptive 基准（v2/V3 谱系实跑值）
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UpperKickCmdPPORunnerCfg(UpperGo2FlatPPORunnerCfg):
    """U-Kick arm D「真分层」runner cfg（3 维命令 + 冻结低层）。

    与 UT 三 kick runner cfg 公平性同源：最小子类——除 experiment_name/
    actor 外逐字继承基类（PPO 超参/obs_groups/clip_actions/critic 逐字
    同 UpperGo2FlatPPORunnerCfg）。
    task id = KICK_CMD_TASK_ID（Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0）。
    """

    experiment_name = "upper_ukick_cmd"   # logs/rsl_rl/upper_ukick_cmd/
    actor = UpperCmdActorCfg()            # 3 维命令 actor（obs 契约缺省 kick 55）


__all__ = [
    "UpperCmdActorCfg",
    "UpperGo2FlatPPORunnerCfg",
    "UpperKickCmdPPORunnerCfg",
]
