"""官方低层 ckpt 能力判别探针（X-HK3 接触鸿沟根因二分：ckpt vs 组装）。

**问题**：probe_ball_contact 官方约定（model_299.pt）下狗不走路（v_body=0、
act 静态），legacy（model_2999.pt）下走路 0.42m/s、球位移 0.327m（物理可行）。
二分：官方 ckpt 本身不会走（A：欠训练/fork 奖励面站桩偏置） vs arm-D 运行时
组装错配（B：frozen_loco_math.assemble_low_level_obs 与训练面 obs 漂移）。

**方法**：在官方 ckpt 的**产出面本身**（HK-Velocity-Flat-Unitree-Go2-v0，
obs manager 面观测 + JointPositionAction(scale=0.5) 动作路径——与训练逐位
同管线）直驱 model_299.pt deterministic：每步把命令 buffer 覆写为常量
[0.4, 0, 0] 后重算 obs → policy → env.step。
  - v_x ≈ 0.4 → ckpt 会走 → arm-D 组装错配（假设 B）
  - v_x ≈ 0   → ckpt 不会走 → ckpt 质量问题（假设 A）

用法（dog env_isaaclab3，包根目录）::

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=$PWD python hierarchical_kick/scripts/probe_official_velocity.py
"""
from __future__ import annotations

import argparse
import os

import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import HK_VELOCITY_TASK_ID

parser = argparse.ArgumentParser(
    description="官方低层 ckpt 速度跟踪判别（自家训练面直驱）")
parser.add_argument("--task", type=str, default=HK_VELOCITY_TASK_ID)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--print_every", type=int, default=25)
parser.add_argument("--cmd_vx", type=float, default=0.4)
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

from hierarchical_kick.frozen_weights import (  # noqa: E402
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
)
_low_path = os.path.abspath(DEFAULT_LOW_LEVEL_CKPT_OFFICIAL)
if not os.path.isfile(_low_path):
    raise SystemExit(f"[probe2] 低层 ckpt 不存在: {_low_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402

from hierarchical_kick.core.contracts import OBS_DIM, SIGMA_SEM  # noqa: E402
from hierarchical_kick.core.low_level_actor import (  # noqa: E402
    build_upper_direct_actor,
    resolve_actor_state_dict,
    validate_official_actor_structure,
)
from hierarchical_kick.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    # [rc1 validator 兼容] 官方 velocity 面 sim.physics = RoughPhysicsCfg
    # （PresetCfg 子类——rc1 SimulationContext 拒收非 concrete PhysicsCfg）。
    # 官方 flat __post_init__ 本就把 default 选为 newton_mjwarp——直接取
    # preset.default（concrete NewtonCfg(PhysicsCfg)）= 训练面同后端。
    phys = env_cfg.sim.physics
    if not type(phys).__name__ == "PhysicsCfg" and hasattr(phys, "default"):
        env_cfg.sim.physics = phys.default
        print(f"[probe2] physics preset → concrete: "
              f"{type(env_cfg.sim.physics).__name__}")
    # [rc1 兼容 #2] events 里 _Preset 包裹的 term（base_com）解包为 concrete
    # EventTermCfg（rc1 EventManager 拒收 _Preset；官方 face 从未在 rc1 跑过）
    for name in list(getattr(env_cfg.events, "__dict__", {})):
        term = getattr(env_cfg.events, name, None)
        if type(term).__name__ == "_Preset" and hasattr(term, "default"):
            setattr(env_cfg.events, name, term.default)
            print(f"[probe2] events preset 解包: events.{name} → "
                  f"{type(term.default).__name__}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    sim_env = env.unwrapped
    robot = sim_env.scene["robot"]
    cmd_term = sim_env.command_manager.get_term("base_velocity")
    print(f"[probe2] task={args_cli.task} obs_dim="
          f"{sim_env.observation_manager.group_obs_dim}")
    print(f"[probe2] 低层 ckpt: {_low_path}")

    ckpt_full, sd = resolve_actor_state_dict(_low_path, device=sim_env.device)
    hidden_dims = validate_official_actor_structure(sd)
    dummy = {"policy": torch.zeros(args_cli.num_envs, OBS_DIM,
                                   device=sim_env.device)}
    policy = build_upper_direct_actor(
        dummy, actor_cfg=None, device=sim_env.device,
        hidden_dims=hidden_dims, activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution",
                          "init_std": SIGMA_SEM},
        sigma_sem=SIGMA_SEM, obs_dim_contract=OBS_DIM)
    policy.load_state_dict(sd, strict=True)
    policy.eval()
    print(f"[probe2] policy 装载完成 hidden={hidden_dims} iter="
          f"{ckpt_full.get('iter', '?')}")

    const_cmd = torch.tensor([args_cli.cmd_vx, 0.0, 0.0],
                             device=sim_env.device).repeat(
        args_cli.num_envs, 1)
    base_start = robot.data.root_pos_w.detach()[:, :2].clone()
    print("\n[probe2] step | obs_cmd | act|mean| | v_body | base_xy | Δbase_xy")
    for step in range(1, int(args_cli.steps) + 1):
        # 命令覆写 → 重算 obs（与训练面 obs manager 同管线） → policy → step
        cmd_term.command[:, :3] = const_cmd
        with torch.inference_mode():
            obs = sim_env.observation_manager.compute()
            obs_cmd = obs["policy"][:, 9:12].mean(dim=0)
            act = policy(obs, stochastic_output=False)
            _, _, _, _ = env.step(act)
        v_body = robot.data.root_lin_vel_b.detach().mean(dim=0)
        if step % int(args_cli.print_every) == 0 or step == 1:
            base_xy = robot.data.root_pos_w.detach()[:, :2].mean(dim=0)
            disp = torch.linalg.vector_norm(
                robot.data.root_pos_w.detach()[:, :2] - base_start,
                dim=-1).mean()
            print(f"[probe2] {step:>4d} | cmd=({float(obs_cmd[0]):+.2f},"
                  f"{float(obs_cmd[1]):+.2f},{float(obs_cmd[2]):+.2f}) | "
                  f"act|mean|={float(act.abs().mean()):.3f} | "
                  f"v=({float(v_body[0]):+.3f},{float(v_body[1]):+.3f},"
                  f"{float(v_body[2]):+.3f}) | "
                  f"base=({base_xy[0]:+.3f},{base_xy[1]:+.3f}) | "
                  f"Δ={float(disp):.3f}m")

    v_mean = float(robot.data.root_lin_vel_b[:, 0].mean())
    print("\n==================== 判别 ====================")
    print(f"末段 v_x = {v_mean:.4f} m/s（目标 {args_cli.cmd_vx}）")
    if abs(v_mean - args_cli.cmd_vx) < 0.1:
        print(">>> ckpt 会跟踪速度命令 → arm-D 组装错配（假设 B）")
    else:
        print(">>> ckpt 不跟踪速度命令 → ckpt 质量问题（假设 A：欠训练/"
              "fork 奖励面站桩偏置）")
    print("==============================================")
    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
