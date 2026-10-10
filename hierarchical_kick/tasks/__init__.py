"""上层任务 gym 注册（官方字符串懒加载模板）。

task id 字符串**冻结不变**。注册模块零 eager 导入：entry point 全为
字符串，cfg 类只在 entry point 解析时拉起。训练入口
``hierarchical_kick/scripts/train.py`` 顶部显式
``import hierarchical_kick.tasks`` 拉起。

无 gymnasium/isaaclab 的离线环境：静默跳过（try/except ImportError；
注册字符串面经模块常量离线消费）。
"""
from __future__ import annotations

# 注册字符串面（模块常量 = 离线测试的契约锚；勿内联进 gym.register 调用）
KICK_CMD_TASK_ID = "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"
"""kick cmd 任务 id（真分层面 Go2KickFlatEnvCfgCommand）。"""

KICK_CMD_TASK_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
"""env entry point（官方 ManagerBasedRLEnv 面，与官方 velocity 任务同）。"""

KICK_CMD_ENV_CFG_ENTRY_POINT = (
    "hierarchical_kick.sim.ball_env_cfg:Go2KickFlatEnvCfgCommand")
"""kick cmd 面 env cfg entry point（字符串懒加载）。"""

KICK_CMD_RSL_RL_CFG_ENTRY_POINT = (
    "hierarchical_kick.agents:UpperKickCmdPPORunnerCfg")
"""kick cmd 面 rsl_rl cfg entry point（UpperKickCmdPPORunnerCfg，3 维命令 actor）。"""

# -- kick cmd 面 rsl_rl cfg entry point（cyberdog2；experiment 隔离子类） --
KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT = (
    "hierarchical_kick.agents:UpperCyberDog2KickCmdPPORunnerCfg")
"""cyberdog2 kick cmd 面 rsl_rl cfg entry point（PPO 面逐字同 go2 版，
仅 experiment_name = upper_ukick_cmd_cyberdog2）。"""

# -- cyberdog2 kick cmd 面（本体置换 C1-C7 见 ball_env_cfg 尾段表） --
KICK_CMD_CYBERDOG2_TASK_ID = "Isaac-Kick-Flat-CyberDog2-Upper-Cmd-v0"
"""cyberdog2 kick cmd 任务 id（真分层面 CyberDog2KickFlatEnvCfgCommand）。"""

KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT = (
    "hierarchical_kick.sim.ball_env_cfg:CyberDog2KickFlatEnvCfgCommand")
"""cyberdog2 kick cmd 面 env cfg entry point（字符串懒加载）。"""

# -- 官方 velocity-flat 训练面（官方口径低层 ckpt 的产出面） --
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

# -- 官方 velocity-flat 训练面（cyberdog2；上游通用基座 + 本体参数快照） --
HK_VELOCITY_CYBERDOG2_TASK_ID = "HK-Velocity-Flat-CyberDog2-v0"
"""cyberdog2 官方口径 velocity-flat 任务 id（通用基座 LocomotionVelocityRoughEnvCfg fork）。"""

HK_VELOCITY_CYBERDOG2_ENV_CFG_ENTRY_POINT = (
    "hierarchical_kick.sim.velocity_official_cfg:"
    "VelocityOfficialCyberDog2FlatEnvCfg")
"""cyberdog2 官方口径 velocity-flat 面 env cfg entry point（字符串懒加载；
基座 = 上游通用 LocomotionVelocityRoughEnvCfg，__post_init__ 修复 + 覆写）。"""

HK_VELOCITY_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT = (
    "hierarchical_kick.agents_velocity:"
    "HKVelocityOfficialCyberDog2FlatPPORunnerCfg")
"""cyberdog2 官方口径 rsl_rl cfg entry point（官方 go2 flat agents cfg 同参换名）。"""

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
        id=KICK_CMD_CYBERDOG2_TASK_ID,
        entry_point=KICK_CMD_TASK_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT,
            "rsl_rl_cfg_entry_point": KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT,
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

    gym.register(
        id=HK_VELOCITY_CYBERDOG2_TASK_ID,
        entry_point=HK_VELOCITY_TASK_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": HK_VELOCITY_CYBERDOG2_ENV_CFG_ENTRY_POINT,
            "rsl_rl_cfg_entry_point": (
                HK_VELOCITY_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT),
        },
    )
except (ImportError, ModuleNotFoundError):
    # 离线（无 gymnasium）静默跳过：注册字符串面常量仍可消费
    pass

__all__ = [
    "KICK_CMD_TASK_ID",
    "KICK_CMD_TASK_ENTRY_POINT",
    "KICK_CMD_ENV_CFG_ENTRY_POINT",
    "KICK_CMD_RSL_RL_CFG_ENTRY_POINT",
    "KICK_CMD_CYBERDOG2_TASK_ID",
    "KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT",
    "KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT",
    "HK_VELOCITY_TASK_ID",
    "HK_VELOCITY_TASK_ENTRY_POINT",
    "HK_VELOCITY_ENV_CFG_ENTRY_POINT",
    "HK_VELOCITY_RSL_RL_CFG_ENTRY_POINT",
    "HK_VELOCITY_CYBERDOG2_TASK_ID",
    "HK_VELOCITY_CYBERDOG2_ENV_CFG_ENTRY_POINT",
    "HK_VELOCITY_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT",
]
