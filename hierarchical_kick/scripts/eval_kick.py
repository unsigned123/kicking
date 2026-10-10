"""U-Kick eval 仪器（上层 ckpt 行为审计 + kick 指标面）。

与 train.py 同构造链（env cfg + 冻结低层 official 口径 + 上层 policy eval
模式 **deterministic mean**），rollout 收集 per-episode 行为指标：
  - kick_count   踢击事件 = 球侧接触力范数 > 0.1N 的步数（包内
                 core/kick_metrics.KickMetricsTracker 单源）
  - goal         episode 结束帧 goal_scored 判据真值（终止面优先，与
                 DoneTerm 同函数同参数）
  - ball_speed   每集球速帧均值 |v_ball|（另附 per-ep max_ball_speed——
                 tracker 同名列）
  - behind_frac  cos_behind > 0 帧占比（ball_mdp._push_geometry_b_impl
                 第 0 列单一实现源）
  - falls        robot_fallen | base_contact 终止的集（真摔口径；
                 time_out/goal_scored/ball_out/robot_out_of_field 不计）
  - ep_len       episode 帧数（500 = 满存活 time_out）
  - end_term     终止归因（goal_scored > ball_out > robot_fallen >
                 base_contact > robot_out_of_field > time_out 优先级）

输出：逐集表 + 汇总均值行（stdout；--out json 全量逐集 + 汇总）。

用法（包根目录）::

    python hierarchical_kick/scripts/eval_kick.py \
        --checkpoint logs/rsl_rl/upper_ukick_cmd/<run>/model_1999.pt \
        --num_envs 64 --episodes 10 --out /tmp/eval_kick.json

    # 总步数预算模式（二选一；--steps > 0 时按精确步数 rollout）
    ... --steps 5000
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# ① 显式注册拉起（字符串懒加载；与 train.py 同款）
import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import (
    KICK_CMD_CYBERDOG2_TASK_ID,
    KICK_CMD_TASK_ID,
)

parser = argparse.ArgumentParser(
    description="U-Kick cmd ckpt 行为 eval（kick_count/goal/ball_speed/"
                "behind_frac/falls/ep_len；deterministic mean）")
parser.add_argument("--robot", type=str, default="go2",
                    choices=["go2", "cyberdog2"],
                    help="本体 profile（分派 task/低层 ckpt 缺省/关节序契约）")
parser.add_argument("--task", type=str, default=None,
                    help="Gym task id（缺省按 --robot 分派；显式给定时须与"
                         " --robot 一致）")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="上层 policy ckpt（model_*.pt；train.py 产物）")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=10,
                    help="每 env 目标完成集数（总目标 = episodes × num_envs；"
                         "--steps > 0 时本项忽略）")
parser.add_argument("--steps", type=int, default=0,
                    help="精确总 rollout 步数（>0 时启用并忽略 --episodes "
                         "停止判据）")
parser.add_argument("--seed", type=int, default=42)
# --stochastic：policy 走随机采样（训练期口径；缺省 False
# = deterministic mean 既有口径）。确定性口径下接触时序不同 kick_count
# 读数会不同——两种口径都评才可比。
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="policy 随机采样口径（训练期 stochastic；"
                         "缺省 deterministic mean。固定 --seed 保证臂间可比）")
# eval-only 球距覆写：球出生沿 +y 固定距（跳过距离采样；横向仍按
# kick_randomization 采样）——球距响应扫描（0.1/0.5/1.0/2.0/2.7m 课程
# 有效性曲线）单一口。缺省 None = 训练同款随机化分布。
parser.add_argument("--ball_dist", type=float, default=None,
                    help="球距覆写 m（球出生 = 狗前方沿 +y 恰该距离；"
                         "缺省 None = 训练同款 dist 采样）")
parser.add_argument("--out", type=str, default=None,
                    help="结果 json 输出路径（缺省仅 stdout）")
parser.add_argument("--low_level_convention", type=str,
                    choices=["official", "legacy"], default="official",
                    help="低层约定（与 train.py 同款）")
parser.add_argument("--low_level_ckpt", type=str, default=None,
                    help="冻结低层 ckpt（缺省按 --low_level_convention"
                         " 分派）")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- launch 缺省分流 + 守卫（与 train.py 逐款同源） ---------------------- #
_TASK_BY_ROBOT = {"go2": KICK_CMD_TASK_ID,
                  "cyberdog2": KICK_CMD_CYBERDOG2_TASK_ID}
if args_cli.task is None:
    args_cli.task = _TASK_BY_ROBOT[args_cli.robot]
elif args_cli.task != _TASK_BY_ROBOT[args_cli.robot]:
    raise SystemExit(
        f"[hkick][eval] --task 与 --robot 不一致: robot={args_cli.robot} "
        f"须 task {_TASK_BY_ROBOT[args_cli.robot]}"
        f"（收到 --task {args_cli.task}）")
from hierarchical_kick.frozen_weights import (  # noqa: E402
    DEFAULT_LOW_LEVEL_CKPT,
    default_low_level_ckpt_official,
)
if args_cli.low_level_ckpt is None:
    args_cli.low_level_ckpt = (
        default_low_level_ckpt_official(args_cli.robot)
        if args_cli.low_level_convention == "official"
        else DEFAULT_LOW_LEVEL_CKPT)
_low_path = os.path.abspath(args_cli.low_level_ckpt)
if not os.path.isfile(_low_path):
    raise SystemExit(f"[hkick][eval] 低层 ckpt 不存在: {_low_path}")
_ckpt_path = os.path.abspath(args_cli.checkpoint)
if not os.path.isfile(_ckpt_path):
    raise SystemExit(f"[hkick][eval] 上层 ckpt 不存在: {_ckpt_path}")
print(f"[hkick][eval] 资产守卫通过: upper={_ckpt_path}  low={_low_path} "
      f"(convention={args_cli.low_level_convention})")

args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- isaaclab / rsl_rl 可用区 ---

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hierarchical_kick.core.kick_metrics import KickMetricsTracker  # noqa: E402
from hierarchical_kick.robots import (  # noqa: E402
    CYBERDOG2_JOINT_ORDER_IL,
    GO2_JOINT_ORDER_IL,
)
from hierarchical_kick.utils.parse_cfg import parse_env_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 关节序契约面（robot 分派；train.py 同源）
_JOINT_ORDER_BY_ROBOT = {"go2": GO2_JOINT_ORDER_IL,
                         "cyberdog2": CYBERDOG2_JOINT_ORDER_IL}

# 终止归因优先级（进球/出球优先于摔/超时；fall 口径 = robot_fallen |
# base_contact——真摔两支，time_out / 场界不算）
_TERM_PRIORITY = ("goal_scored", "ball_out", "robot_fallen", "base_contact",
                  "robot_out_of_field", "time_out")
_FALL_TERMS = ("robot_fallen", "base_contact")


def _term_flags(sim_env) -> dict[str, torch.Tensor | None]:
    tm = sim_env.termination_manager
    flags: dict[str, torch.Tensor | None] = {}
    for name in _TERM_PRIORITY:
        try:
            flags[name] = tm.get_term(name).detach().to(torch.bool)
        except Exception:
            flags[name] = None
    return flags


def main() -> None:
    # 1) env cfg（与 train.py 同构造链） ----------------------------------- #
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.actions.joint_pos.low_level_ckpt_path = _low_path
    env_cfg.actions.joint_pos.low_level_convention = \
        str(args_cli.low_level_convention)
    # eval-only 球距覆写（reset_ball 采样口 dist 轴取常量）
    if args_cli.ball_dist is not None:
        env_cfg.kick_randomization.ball_dist_override = float(args_cli.ball_dist)
        print(f"[hkick][eval] ball_dist 覆写: "
              f"{float(args_cli.ball_dist)} m（球出生 = 狗前方沿 +y 恰该距，"
              "横向仍采样；跳过距离课程采样）")
    else:
        print("[hkick][eval] ball_dist 未覆写 → 训练同款随机化分布"
              "（eval 时段课程计数从 0 起，d_max≈起点 0.5m 域）")

    # 2) agent cfg（仅作 runner 构造面；训练超参不消费） ------------------- #
    if args_cli.robot == "go2":
        from hierarchical_kick.agents import (
            UpperKickCmdPPORunnerCfg as _RunnerCfg)
    else:
        from hierarchical_kick.agents import (
            UpperCyberDog2KickCmdPPORunnerCfg as _RunnerCfg)
    agent_cfg = _RunnerCfg()
    agent_cfg.run_name = "eval_kick"
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.device = str(env_cfg.sim.device)
    from importlib.metadata import version as _pkg_version
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, _pkg_version("rsl-rl-lib"))

    # 3) env + 契约断言（与 train.py 同款） ------------------------------- #
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    sim_env = env.unwrapped
    joint_names = list(sim_env.scene["robot"].joint_names)
    expected_joints = _JOINT_ORDER_BY_ROBOT[args_cli.robot]
    if set(joint_names) != set(expected_joints):
        raise SystemExit(f"[hkick][eval] env 关节集合漂移: {joint_names} != "
                         f"{list(expected_joints)}")
    gdim = {str(k): int(v[-1]) for k, v in
            sim_env.observation_manager.group_obs_dim.items()}
    from hierarchical_kick.core.kick_metrics import \
        KICK_POLICY_OBS_DIM as _KICK_OBS_DIM
    if gdim.get("policy") != _KICK_OBS_DIM:
        raise SystemExit(f"[hkick][eval] policy obs 维 {gdim.get('policy')} "
                         f"!= {_KICK_OBS_DIM}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    num_envs = int(env.num_envs)
    device = agent_cfg.device

    # 4) runner + ckpt 载入（load 恢复 policy 权重 + obs normalizer 态） --- #
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=device)
    runner.load(_ckpt_path)
    policy = runner.get_inference_policy(device=device)  # eval 模式
    stochastic = bool(args_cli.stochastic)
    # 随机采样口径的臂间可比性：固定 torch RNG（env seed 已设；采样流同序）
    torch.manual_seed(int(args_cli.seed))
    print(f"[hkick][eval] ckpt 载入完成: {_ckpt_path}（"
          f"{'stochastic sampling' if stochastic else 'deterministic mean'}"
          f" 口径, seed={args_cli.seed}）")

    # 5) 指标仪器（kick_count/goal/behind/frames 单源 = KickMetricsTracker） #
    tracker = KickMetricsTracker(num_envs=num_envs, device=device)
    ball_speed_sum = torch.zeros(num_envs, device=device)
    ball_frames = torch.zeros(num_envs, device=device)
    ep_idx = [0] * num_envs

    if args_cli.steps > 0:
        total_steps = int(args_cli.steps)
        target_episodes = None
    else:
        target_episodes = int(args_cli.episodes) * num_envs
        total_steps = target_episodes * 1100  # 安全上限 ≈ 2.2× nominal 500 步
    rows: list[dict] = []

    obs_dict, _ = env.reset()
    step = 0
    while step < total_steps:
        if target_episodes is not None and len(rows) >= target_episodes:
            break
        with torch.inference_mode():
            actions = policy(obs_dict, stochastic_output=stochastic)
        obs_dict, _rew, dones, _extras = env.step(actions)
        tracker.update(sim_env)
        speed = torch.linalg.vector_norm(
            sim_env.scene["ball"].data.root_lin_vel_w.detach(), dim=-1)
        ball_speed_sum += speed
        ball_frames += 1.0
        step += 1
        done_any = bool(dones.any())
        if done_any:
            flags = _term_flags(sim_env)
            done_mask = dones.detach().to(torch.bool).reshape(-1)
            done_idx = done_mask.nonzero(as_tuple=False).flatten()
            spd_mean = (ball_speed_sum[done_idx] /
                        ball_frames[done_idx].clamp(min=1.0))
            new_rows = tracker.pop_done(sim_env, dones)
            spd_l = spd_mean.tolist()
            idxs_l = done_idx.tolist()
            by_env = {r["env"]: r for r in new_rows}
            for env_id, sm in zip(idxs_l, spd_l):
                row = by_env.get(env_id)
                if row is None:
                    continue
                end_term = "unknown"
                for name in _TERM_PRIORITY:
                    f = flags.get(name)
                    if f is not None and bool(f[env_id]):
                        end_term = name
                        break
                fall = any(flags.get(n) is not None
                           and bool(flags[n][env_id]) for n in _FALL_TERMS)
                frames = float(row["frames"])
                row.update({
                    "ep": ep_idx[env_id],
                    "end_term": end_term,
                    "fall": bool(fall),
                    "ball_speed_mean": float(sm),
                    "behind_frac": (float(row["behind_frames"]) / frames
                                    if frames > 0 else 0.0),
                })
                rows.append(row)
                ep_idx[env_id] += 1
            ball_speed_sum[done_idx] = 0.0
            ball_frames[done_idx] = 0.0
        if step % 200 == 0:
            print(f"[hkick][eval] step {step}/{total_steps}  "
                  f"episodes={len(rows)}"
                  + (f"/{target_episodes}" if target_episodes else ""))

    # 6) 汇总 + 输出 ------------------------------------------------------- #
    if not rows:
        raise SystemExit("[hkick][eval] 零 episode 完成——检查 ckpt/env")
    n = len(rows)
    total_frames = sum(r["frames"] for r in rows)
    summary = {
        "mode": ("stochastic" if stochastic else "deterministic"),
        "episodes": n,
        "kick_count": sum(r["kick_count"] for r in rows) / n,
        "goal_rate": sum(r["goal"] for r in rows) / n,
        "ball_speed_mean": sum(r["ball_speed_mean"] for r in rows) / n,
        "max_ball_speed": sum(r["max_ball_speed"] for r in rows) / n,
        "behind_frac": (sum(r["behind_frames"] for r in rows) / total_frames
                        if total_frames > 0 else 0.0),
        "falls_per_ep": sum(r["fall"] for r in rows) / n,
        "ep_len": total_frames / n,
        "end_term_counts": dict(Counter(r["end_term"] for r in rows)),
    }
    print(f"\n[hkick][eval] 逐集表（{n} 集; ckpt={os.path.basename(_ckpt_path)}"
          f" num_envs={num_envs} "
          f"mode={'stochastic' if stochastic else 'deterministic'}）")
    print("env  ep  ep_len  kick_count  goal  ball_speed_mean  "
          "max_ball_speed  behind_frac  fall  end_term")
    for r in rows:
        print(f"{r['env']:>3d}  {r['ep']:>2d}  {int(r['frames']):>5d}  "
              f"{r['kick_count']:>9.0f}  {str(r['goal']):>4s}  "
              f"{r['ball_speed_mean']:>14.3f}  {r['max_ball_speed']:>13.3f}  "
              f"{r['behind_frac']:>10.4f}  {str(r['fall']):>4s}  "
              f"{r['end_term']}")
    print(f"\n[hkick][eval] SUMMARY (mean over {n} episodes)")
    for k, v in summary.items():
        print(f"    {k:<18s} = {v}")

    if args_cli.out:
        out = {
            "checkpoint": _ckpt_path,
            "task": args_cli.task,
            "num_envs": num_envs,
            "seed": int(args_cli.seed),
            "stochastic": stochastic,
            "ball_dist": args_cli.ball_dist,
            "steps_run": step,
            "episodes_collected": n,
            "low_level_convention": str(args_cli.low_level_convention),
            "summary": summary,
            "episodes": rows,
        }
        out_path = os.path.abspath(args_cli.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[hkick][eval] 结果 json 已写: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
