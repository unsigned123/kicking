"""上层任务 gym 注册（官方字符串懒加载模板；U0 §4 工程结构）。

[cmd-only 拆分] 原 UT upper/sim/register 的五个 task id
（velocity 面 / kick 三臂原面 / oracle 面 / ours Seg 面）随 ours/e2e 臂
拆分移除；仅保留 arm D「真分层」cmd 面，task id 字符串**冻结不变**
（与 UT 同环境共存会有重复注册警告，可接受）。

注册模块零 eager 导入：entry point 全为字符串，cfg 类只在 entry point
解析时拉起。训练入口 ``hierarchical_kick/scripts/train.py`` 顶部显式
``import hierarchical_kick.tasks`` 拉起。

无 gymnasium/isaaclab 的离线环境：静默跳过（try/except ImportError；
注册字符串面经模块常量离线消费）。
"""
from __future__ import annotations

# 注册字符串面（模块常量 = 离线测试的契约锚；勿内联进 gym.register 调用）
KICK_CMD_TASK_ID = "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"
"""kick arm D 任务 id（真分层面 Go2KickFlatEnvCfgCommand）。"""

KICK_CMD_TASK_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
"""env entry point（官方 ManagerBasedRLEnv 面，与官方 velocity 任务同）。"""

KICK_CMD_ENV_CFG_ENTRY_POINT = (
    "hierarchical_kick.sim.ball_env_cfg:Go2KickFlatEnvCfgCommand")
"""kick arm D 面 env cfg entry point（字符串懒加载）。"""

KICK_CMD_RSL_RL_CFG_ENTRY_POINT = (
    "hierarchical_kick.agents:UpperKickCmdPPORunnerCfg")
"""kick arm D 面 rsl_rl cfg entry point（UpperKickCmdPPORunnerCfg，3 维命令 actor）。"""

# -- 官方 velocity-flat 训练面（官方口径低层 ckpt 的产出面；2026-10-09） --
HK_VELOCITY_TASK_ID = "HK-Velocity-Flat-Unitree-Go2-v0"
"""官方口径 velocity-flat 任务 id（fork UnitreeGo2FlatEnvCfg 覆写回上游口径）。"""

HK_VELOCITY_TASK_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
"""env entry point（官方 ManagerBasedRLEnv 面，与官方 velocity 任务同）。"""

HK_VELOCITY_ENV_CFG_ENTRY_POINT = (
    "hierarchical_kick.sim.velocity_official_cfg:VelocityOfficialGo2FlatEnvCfg")
"""官方口径 velocity-flat 面 env cfg entry point（字符串懒加载；基类 =
isaaclab_tasks fork 的 UnitreeGo2FlatEnvCfg，__post_init__ 覆写回上游口径）。"""

HK_VELOCITY_RSL_RL_CFG_ENTRY_POINT = (
    "hierarchical_kick.agents_velocity:HKVelocityOfficialGo2FlatPPORunnerCfg")
"""官方口径 rsl_rl cfg entry point（官方 go2 flat agents cfg 复制适配）。"""

try:
    import gymnasium as gym

    gym.register(
        id=KICK_CMD_TASK_ID,
        entry_point=KICK_CMD_TASK_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": KICK_CMD_ENV_CFG_ENTRY_POINT,
            "rsl_rl_cfg_entry_point": KICK_CMD_RSL_RL_CFG_ENTRY_POINT,
        },
    )

    gym.register(
        id=HK_VELOCITY_TASK_ID,
        entry_point=HK_VELOCITY_TASK_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": HK_VELOCITY_ENV_CFG_ENTRY_POINT,
            "rsl_rl_cfg_entry_point": HK_VELOCITY_RSL_RL_CFG_ENTRY_POINT,
        },
    )
except (ImportError, ModuleNotFoundError):
    # 离线（无 gymnasium）静默跳过：注册字符串面常量仍可消费（U1-⑥）
    pass

__all__ = [
    "KICK_CMD_TASK_ID",
    "KICK_CMD_TASK_ENTRY_POINT",
    "KICK_CMD_ENV_CFG_ENTRY_POINT",
    "KICK_CMD_RSL_RL_CFG_ENTRY_POINT",
    "HK_VELOCITY_TASK_ID",
    "HK_VELOCITY_TASK_ENTRY_POINT",
    "HK_VELOCITY_ENV_CFG_ENTRY_POINT",
    "HK_VELOCITY_RSL_RL_CFG_ENTRY_POINT",
]
