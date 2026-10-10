"""上层 RL 的 rsl_rl runner/模型 cfg（isaaclab 面；仅 sim 侧解析）。

零 eager 消费：本模块只被 register/launch 的 entry point 字符串解析拉起。
PPO 超参在基类 UpperGo2FlatPPORunnerCfg 单源（子类除 actor/experiment_name
外逐字继承——公平性纪律）。
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
    """cmd「真分层」上层命令 actor cfg（3 维 [vx,vy,wz] 输出）。

    同 trunk/同探索/同 obs 面：hidden_dims/activation/obs_normalization/
    distribution_cfg/sigma_sem 全同基线值。无 decoder/z/k_env 面（上层无
    冻结资产消费——冻结低层在 env 侧动作 term）。
    distribution init_std = SIGMA_SEM 同值（3 维命令空间无 scale_vec →
    标量 σ_sem 逐维填充，ctor 内执行）。
    """

    class_name: str = UPPER_CMD_ACTOR_CLASS_PATH
    hidden_dims: list[int] = [128, 128, 128]
    activation: str = "elu"
    obs_normalization: bool = False   # 守卫 2：同 obs 面（无归一）
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=SIGMA_SEM)           # 标量名义值；3 维命令空间 σ init = σ_sem
    # --- upper 自定义字段（经 to_dict 透传 UpperCmdActor 构造） ---
    sigma_sem: float = SIGMA_SEM      # 命令空间探索 σ 同源
    obs_dim_contract: int = KICK_POLICY_OBS_DIM   # cmd 面恒 kick 面 55


@configclass
class UpperGo2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """上层 go2 kick-flat runner cfg 基类（experiment 隔离 upper_v0）。

    PPO 超参：lr 1e-4 adaptive / γ 0.99 / λ 0.95 / clip 0.2 /
    epochs 5 / minibatch 4 / entropy 0.01 / grad 1.0。
    ⚠️ kick 面训练入口（train.py）以 ENTROPY_COEF=0.001 运行期覆写本缺省
    （0.003 时 std 无界增长、0.0 探索消亡，取 0.001）。

    actor 缺省 = UpperCmdActorCfg()（cmd 面）。
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "upper_v0"      # logs/rsl_rl/upper_v0/ 隔离
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
        learning_rate=1.0e-4,     # adaptive 基准
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UpperKickCmdPPORunnerCfg(UpperGo2FlatPPORunnerCfg):
    """cmd「真分层」runner cfg（3 维命令 + 冻结低层）。

    最小子类——除 experiment_name/
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
