"""官方口径 velocity-flat 训练入口（HK-Velocity-Flat-Unitree-Go2-v0）。

冻结低层（arm D）官方口径 ckpt 的产出面：官方 go2 flat agents cfg
（MLP[128,128,128] ELU、lr 1e-3 adaptive、300 it）+ 官方 48 维单帧 policy
obs（sim/velocity_official_cfg.py 覆写回上游口径：joint_vel raw、单帧、
动作均匀 0.5）。结构 = 官方 runner 模板（parse_env_cfg +
RslRlVecEnvWrapper + OnPolicyRunner；kick 面 scripts/train.py 同款）。

用法（dog env_isaaclab，包根目录）::

    python hierarchical_kick/scripts/train_velocity.py \
        --num_envs 4096 --max_iterations 300

产物：logs/rsl_rl/hk_velocity_flat_official/<date>_<run_name>/
model_*.pt（终点 model_(N-1).pt → 拷入 hierarchical_kick/weights/
velocity_flat_official/ 后为官方口径低层缺省资产；2026-10-09 起缺省
资产 = model_walk.pt——rc1 runtime 修复后 1000it@4096env 重训，
Mean reward 32.87 过行走门，替代站桩的 model_299.pt）。
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

# ① 显式注册拉起（字符串懒加载，零 eager 导入；离线无 gymnasium 静默跳过）
import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import HK_VELOCITY_TASK_ID

parser = argparse.ArgumentParser(
    description="官方口径 velocity-flat train（官方 48 维单帧 obs + 均匀 "
                "0.5 动作 scale；冻结低层官方 ckpt 产出面）")
parser.add_argument("--task", type=str, default=HK_VELOCITY_TASK_ID,
                    help="Gym task id（hierarchical_kick.tasks 注册）")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--run_name", type=str, default="velocity_official")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--distributed", action="store_true", default=False,
                    help="torchrun 多卡：设备逐 rank = cuda:LOCAL_RANK")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- launch 缺省分流 + 守卫（AppLauncher 拉起前 fail-fast） ---- #
# 单 task 面：本入口仅服务官方口径 velocity 任务（kick cmd 面走 train.py）
if args_cli.task != HK_VELOCITY_TASK_ID:
    raise SystemExit(
        f"[hkick][velocity] 仅支持任务 id: {HK_VELOCITY_TASK_ID}"
        f"（收到 --task {args_cli.task}；kick cmd 面请走 "
        "hierarchical_kick/scripts/train.py）")

# 训练一律 headless（服务器无显示）
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 逐 rank 设备（torchrun 注入 LOCAL_RANK）
if getattr(args_cli, "distributed", False):
    args_cli.device = f"cuda:{app_launcher.local_rank}"

# --- isaaclab / rsl_rl 可用区 ---

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hierarchical_kick.robots.go2 import GO2_JOINT_ORDER_IL  # noqa: E402
from hierarchical_kick.utils.parse_cfg import (  # noqa: E402
    get_checkpoint_path,
    parse_env_cfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _make_log_dir(agent_cfg, base: str = "logs/rsl_rl") -> str:
    """logs/rsl_rl/<experiment>/<date>[_<run_name>]（官方 train.py 原逻辑）。"""
    log_root = os.path.abspath(os.path.join(base, agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if getattr(agent_cfg, "run_name", None):
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    return log_dir


def main() -> None:
    # 1) env cfg（注册表 entry point 字符串解析 + CLI 覆盖） -------------- #
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)

    # 官方口径装配自检（__post_init__ 覆写面；漂移 fail-fast） ------------- #
    pol = env_cfg.observations.policy
    jv_scale = getattr(pol.joint_vel, "scale", None)
    hist = getattr(pol, "history_length", None)
    act_scale = env_cfg.actions.joint_pos.scale
    if jv_scale not in (1.0, None) or hist is not None \
            or act_scale != 0.5:
        raise SystemExit(
            f"[hkick][velocity] 官方口径装配漂移: joint_vel.scale={jv_scale}"
            f"（须 1.0/None）history_length={hist}（须 None=单帧）"
            f" action.scale={act_scale}（须 0.5）")
    print(f"[hkick][velocity] 官方口径装配自检通过: joint_vel.scale="
          f"{jv_scale} history_length={hist}（单帧）action.scale={act_scale}")

    # 2) agent cfg（官方 go2 flat agents cfg；handle_deprecated 演绎新面） - #
    from hierarchical_kick.agents_velocity import (
        HKVelocityOfficialGo2FlatPPORunnerCfg as _RunnerCfg,
    )
    agent_cfg = _RunnerCfg()
    if args_cli.max_iterations:
        agent_cfg.max_iterations = int(args_cli.max_iterations)
    if args_cli.save_interval:
        agent_cfg.save_interval = int(args_cli.save_interval)
    if args_cli.experiment_name:
        agent_cfg.experiment_name = str(args_cli.experiment_name)
    agent_cfg.run_name = str(args_cli.run_name)
    # parse_env_cfg 缺省不设 seed（None）——launch 惯例缺省 42
    agent_cfg.seed = int(env_cfg.seed) if env_cfg.seed is not None else 42
    agent_cfg.device = str(env_cfg.sim.device)
    from importlib.metadata import version as _pkg_version
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, _pkg_version("rsl-rl-lib"))

    # 3) env + 关节序打印（与 kick env 逐位比对锚） ----------------------- #
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()  # scene spawn 完成（joint_names / group_obs_dim 可读面）
    sim_env = env.unwrapped
    joint_names = list(sim_env.scene["robot"].joint_names)
    print(f"[hkick][velocity] velocity env joint order ({len(joint_names)}): "
          f"{joint_names}")
    if set(joint_names) != set(GO2_JOINT_ORDER_IL):
        raise SystemExit(
            f"[hkick][velocity] env 关节集合漂移: {joint_names} != "
            f"{list(GO2_JOINT_ORDER_IL)}（fail-fast；832854c 教训）")
    if joint_names != list(GO2_JOINT_ORDER_IL):
        print(f"[hkick][velocity][WARN] 关节序与 GO2_JOINT_ORDER_IL 有序差 "
              f"{joint_names} vs {list(GO2_JOINT_ORDER_IL)}")
    gdim = {str(k): int(v[-1]) for k, v in
            sim_env.observation_manager.group_obs_dim.items()}
    print("[hkick][velocity] runtime group_obs_dim:", gdim)
    # obs 维契约：官方 velocity policy 48 维单帧（7 项：3+3+3+3+12+12+12）
    if gdim.get("policy") != 48:
        raise SystemExit(
            f"[hkick][velocity] policy obs 维 {gdim.get('policy')} != 48"
            "（官方口径单帧装配漂移，fail-fast）")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 4) 日志目录 + 配置落盘 ------------------------------------------------ #
    log_dir = _make_log_dir(agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print_dict({"num_envs": env_cfg.scene.num_envs,
                "device": agent_cfg.device,
                "experiment_name": agent_cfg.experiment_name})

    # 5) runner + 训练（官方 OnPolicyRunner 面） --------------------------- #
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    num_iters = int(agent_cfg.max_iterations)
    if args_cli.resume:
        if args_cli.checkpoint and os.path.isfile(
                os.path.abspath(args_cli.checkpoint)):
            resume_path = os.path.abspath(args_cli.checkpoint)
            print(f"[INFO] Loading model checkpoint from: {resume_path} "
                  "(explicit path)")
        else:
            log_root = os.path.abspath(
                os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            resume_path = get_checkpoint_path(
                log_root, args_cli.load_run, args_cli.checkpoint)
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        start_it = int(getattr(runner, "current_learning_iteration", 0) or 0)
        num_iters = max(0, num_iters - start_it)
        print(f"[hkick][velocity] resume @ iter {start_it} -> remaining "
              f"iterations {num_iters} (max_iterations 总预算不变)")

    runner.learn(num_learning_iterations=num_iters,
                 init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
