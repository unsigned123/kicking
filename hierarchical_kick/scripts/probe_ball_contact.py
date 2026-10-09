"""U-Kick 接触物理正探针（X-HK2 下一刀 #2：脚本正探针判物理）。

**问题**：isaaclab3/Kit 110 新 PhysX 下，球 collider 是否真被狗撞动？
（UT 旧栈 12.44 kick/集 vs HK3 6400 eval 集 ball_speed 逐位 0 的范畴级异常，
最后未验证判据 = 接触物理可行性。）

**方法**：构造链与 eval_kick.py 同源（parse_env_cfg + gym.make + wrapper），
**不载任何策略**——直接向 action buffer 写常量 [0.4, 0, 0]（机体系直行命令
0.4 m/s），冻结低层转关节目标，跑 5 秒（250 步 @50Hz）。逐 25 步打印：
球 root_pos_xy、球 root_lin_vel 范数、狗 base_pos_xy、足端接触力范数 max
（球侧传感器 filter 面——与 kick_event/kick_count 同一检测源）。

**判定**：球位移 Δ‖ball_xy‖（相对本次 episode 段锚点，done 重锚）：
  > 0.2m  = 物理可行（接触鸿沟不在物理层 → 归因行为/课程）
  ≈ 0     = 接触物理有问题（collider/材质/解算回归 → 归因物理层）

**二分定位（球不动时人工读表）**：
  ① 狗是否真在走：base_pos 位移 >0.3m？（低层是否执行直行命令）
  ② 狗端点（base+四足）到球心最小距离：是否根本没走到（球半径 0.11 +
     足端半径，接触线 ≈0.15m 内）？
  ③ 旧栈对照：env_isaaclab（0.54.3）跑同脚本（import 面差异见模块尾注）；
     旧栈不适配则以 dog UT cmd 臂 ball_speed 1.263 / 12.44 kick/集 作旁证。

用法（dog env_isaaclab3，包根目录）::

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=$PWD python hierarchical_kick/scripts/probe_ball_contact.py

可选：--steps 250 --cmd_vx 0.4 --ball_x 1.0（覆写球 spawn x 做 spawn 扫描）
"""
from __future__ import annotations

import argparse
import os

# ① 显式注册拉起（字符串懒加载；与 eval_kick.py 同款）
import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import KICK_CMD_TASK_ID

parser = argparse.ArgumentParser(
    description="U-Kick 接触物理正探针（常量直行命令驱狗撞球，判物理可行性）")
parser.add_argument("--task", type=str, default=KICK_CMD_TASK_ID,
                    help="Gym task id（hierarchical_kick.tasks 注册）")
parser.add_argument("--num_envs", type=int, default=1,
                    help="env 数（探针单 env 足够；>1 时打印批内均值/最小）")
parser.add_argument("--steps", type=int, default=250,
                    help="rollout 步数（50Hz；250 步 = 5 秒）")
parser.add_argument("--print_every", type=int, default=25,
                    help="打印间隔（步）")
parser.add_argument("--cmd_vx", type=float, default=0.4,
                    help="常量直行命令幅值 [cmd_vx, 0, 0]（m/s 机体系）")
parser.add_argument("--ball_x", type=float, default=None,
                    help="球 spawn x 覆写（None = env cfg 单源 1.0； spawn "
                         "扫描用，y/z 不动）")
parser.add_argument("--low_level_convention", type=str,
                    choices=["official", "legacy"], default="official",
                    help="冻结低层约定（与 train.py 同款；legacy = 旧栈旁证"
                         " ckpt model_2999）")
parser.add_argument("--low_level_ckpt", type=str, default=None,
                    help="冻结低层 ckpt 显式路径（None = 按 "
                         "--low_level_convention 分派包内缺省）")
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# 资产守卫（与 eval_kick.py 同款：探针不载 ckpt 但动作 term 构造期要低层）
from hierarchical_kick.frozen_weights import (  # noqa: E402
    DEFAULT_LOW_LEVEL_CKPT,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
)
_low_path = os.path.abspath(
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL
    if args_cli.low_level_convention == "official" else DEFAULT_LOW_LEVEL_CKPT)
if args_cli.low_level_ckpt is not None:
    _low_path = os.path.abspath(args_cli.low_level_ckpt)
if not os.path.isfile(_low_path):
    raise SystemExit(f"[hkick][probe] 低层 ckpt 不存在: {_low_path}")
print(f"[hkick][probe] 低层资产守卫通过 ({args_cli.low_level_convention}): "
      f"{_low_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- isaaclab / torch 可用区（Kit 110 纪律：launch 后才 import torch） ---

import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from hierarchical_kick.core.kick_metrics import KickMetricsTracker  # noqa: E402
from hierarchical_kick.robots.go2 import GO2_FOOT_BODIES  # noqa: E402
from hierarchical_kick.utils.parse_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    # 1) env cfg（与 eval_kick.py 同构造链；不设 agent cfg——无 runner） --- #
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.actions.joint_pos.low_level_ckpt_path = _low_path
    env_cfg.actions.joint_pos.low_level_convention = \
        str(args_cli.low_level_convention)
    if args_cli.ball_x is not None:
        old = tuple(env_cfg.scene.ball.init_state.pos)
        env_cfg.scene.ball.init_state.pos = (
            float(args_cli.ball_x), old[1], old[2])
        print(f"[hkick][probe] 球 spawn x 覆写: {old} -> "
              f"{tuple(env_cfg.scene.ball.init_state.pos)}")

    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    sim_env = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=10.0)
    num_envs = int(env.num_envs)
    device = sim_env.device

    robot = sim_env.scene["robot"]
    ball = sim_env.scene["ball"]
    # 足端+base body 索引（狗端点→球心距离用；鼻端无 body，足端/躯干保守面）
    body_names = list(robot.data.body_names)
    probe_bodies = ["base", *GO2_FOOT_BODIES]
    body_idx = [body_names.index(b) for b in probe_bodies]
    print(f"[hkick][probe] 探针 body: {list(zip(probe_bodies, body_idx))}")
    print(f"[hkick][probe] 球 spawn: {tuple(ball.data.default_root_state[0, :3].tolist())}"
          f"  狗 spawn: {tuple(robot.data.default_root_state[0, :3].tolist())}")

    tracker = KickMetricsTracker(num_envs=num_envs, device=device)
    const_action = torch.tensor([args_cli.cmd_vx, 0.0, 0.0],
                                device=device).repeat(num_envs, 1)

    # episode 段锚点在 reset 后初始化（done 重锚——球被 reset 拉回 spawn）
    min_body_ball_dist = None  # 探针 body→球心最小中心距（全程）
    total_kick_steps = 0
    max_ball_speed_seen = 0.0
    done_events: list[str] = []
    # 步态振荡仪表（行走门判据②：joint_vel 交替）——对角大腿对 FL/RL
    _qd_names = list(robot.data.joint_names)
    qd_probe = [_qd_names.index("FL_thigh_joint"), _qd_names.index("RL_thigh_joint")]
    qd_hist: list = []

    obs_dict, _ = env.reset()
    # episode 段锚点（done 后重锚——球被 reset 拉回 spawn；reset 后即锚定）
    ball_anchor = ball.data.root_pos_w.detach()[:, :2].clone()
    base_start = robot.data.root_pos_w.detach()[:, :2].clone()
    print("\n[hkick][probe] step | ball_xy (mean) | |v_ball| (max) | "
          "base_xy (mean) | |Δbase_xy| (mean) | contact_force_max (N)")
    for step in range(1, int(args_cli.steps) + 1):
        with torch.inference_mode():
            obs_dict, _rew, dones, _extras = env.step(const_action)
        tracker.update(sim_env)
        qd_hist.append(robot.data.joint_vel[0, qd_probe].detach())

        ball_pos = ball.data.root_pos_w.detach()
        ball_vel_n = torch.linalg.vector_norm(
            ball.data.root_lin_vel_w.detach(), dim=-1)
        base_pos = robot.data.root_pos_w.detach()
        force_max = tracker._robot_contact_force(sim_env)  # (N,) per-env max

        # body→球心最小距离（全程追踪）
        bp = robot.data.body_pos_w.detach()[:, body_idx, :]  # (N, B, 3)
        dist = torch.linalg.vector_norm(
            bp - ball_pos.unsqueeze(1), dim=-1)  # (N, B)
        d_min = dist.min()
        min_body_ball_dist = (d_min if min_body_ball_dist is None
                              else torch.minimum(min_body_ball_dist, d_min))

        done_mask = dones.detach().to(torch.bool).reshape(-1)
        if bool(done_mask.any()):
            rows = tracker.pop_done(sim_env, dones)
            kc = int(sum(r["kick_count"] for r in rows))
            total_kick_steps += kc
            terms = []
            for name in ("goal_scored", "ball_out", "robot_fallen",
                         "base_contact", "robot_out_of_field", "time_out"):
                try:
                    f = sim_env.termination_manager.get_term(name)
                    if bool(f[done_mask].any()):
                        terms.append(name)
                except Exception:
                    pass
            done_events.append(f"step{step}:{'+'.join(terms) or '?'}"
                               f"(kick{kc})")
            # done env 重锚（ManagerBasedRLEnv step 内已 auto-reset）
            ball_anchor[done_mask] = ball_pos[done_mask, :2]
            base_start[done_mask] = base_pos[done_mask, :2]

        max_ball_speed_seen = max(max_ball_speed_seen,
                                  float(ball_vel_n.max()))
        # 低层仪表：cmd 通道回读 / 低层动作统计 / 实际机体系速度
        # （cmd 切片按约定：official [9:12] / legacy [45:48]）
        aterm = sim_env.action_manager.get_term("joint_pos")
        cmd_sl = slice(9, 12) if aterm.low_level_convention == "official" \
            else slice(45, 48)
        low_obs_cmd = aterm._low_obs[:, cmd_sl].mean(dim=0)
        la = aterm._low_last_action
        root_vel_b = robot.data.root_lin_vel_b.detach().mean(dim=0)
        if step % int(args_cli.print_every) == 0 or step == 1:
            ball_xy = ball_pos[:, :2].mean(dim=0)
            base_xy = base_pos[:, :2].mean(dim=0)
            ball_disp = torch.linalg.vector_norm(
                ball_pos[:, :2] - ball_anchor, dim=-1).mean()
            base_disp = torch.linalg.vector_norm(
                base_pos[:, :2] - base_start, dim=-1).mean()
            print(f"[hkick][probe] {step:>4d} | "
                  f"ball=({ball_xy[0]:+.3f},{ball_xy[1]:+.3f}) "
                  f"Δ={float(ball_disp):.3f} | "
                  f"|v|={float(ball_vel_n.max()):.3f} | "
                  f"base=({base_xy[0]:+.3f},{base_xy[1]:+.3f}) "
                  f"Δ={float(base_disp):.3f} | "
                  f"F={float(force_max.max()):.3f}N | "
                  f"min_d={float(d_min):.3f}m")
            print(f"[hkick][probe]        low: cmd=({float(low_obs_cmd[0]):+.3f},"
                  f"{float(low_obs_cmd[1]):+.3f},{float(low_obs_cmd[2]):+.3f}) "
                  f"act|mean|={float(la.abs().mean()):.3f} "
                  f"act_std={float(la.std()):.3f} "
                  f"v_body=({float(root_vel_b[0]):+.3f},"
                  f"{float(root_vel_b[1]):+.3f},{float(root_vel_b[2]):+.3f})")

    # 2) 判定 -------------------------------------------------------------- #
    ball_pos = ball.data.root_pos_w.detach()
    final_disp = torch.linalg.vector_norm(
        ball_pos[:, :2] - ball_anchor, dim=-1)
    disp_max = float(final_disp.max())
    print("\n==================== 探针判定 ====================")
    print(f"球位移（末态 − 段锚点）: max={disp_max:.4f} m  "
          f"mean={float(final_disp.mean()):.4f} m")
    print(f"球速峰值: {max_ball_speed_seen:.4f} m/s")
    print(f"kick 步数（力>0.1N）: {total_kick_steps}")
    print(f"狗 base 位移: mean="
          f"{float(torch.linalg.vector_norm(base_pos[:, :2] - base_start, dim=-1).mean()):.4f} m")
    qd = torch.stack(qd_hist)  # (T, 2)
    qd_flips = int((torch.diff(torch.sign(qd[:, 0])) != 0).sum())
    print(f"步态振荡（FL_thigh joint_vel 时序）: |mean|={float(qd[:, 0].abs().mean()):.3f} "
          f"std={float(qd[:, 0].std()):.3f} rad/s  符号翻转={qd_flips}"
          f"（对角对 RL_thigh |mean|={float(qd[:, 1].abs().mean()):.3f}）"
          "——翻转>10 且 |mean|>1 = 有步态")
    print(f"狗端点(base+4足)→球心 最小中心距: "
          f"{float(min_body_ball_dist):.4f} m"
          "（接触线≈0.15m：球半径0.11+足端半径）")
    if done_events:
        print(f"done 事件: {done_events}")
    verdict = ("物理可行（球被动了，接触鸿沟不在物理层 → 归因行为/课程）"
               if disp_max > 0.2 else
               "接触物理有问题（球≈未动 → collider/材质/解算层排查）")
    print(f"\n>>> 判定: 球位移 {disp_max:.4f} m "
          f"{'>' if disp_max > 0.2 else '≤'} 0.2m 阈 → **{verdict}**")
    print("==================================================")

    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()

# ---------------------------------------------------------------------------
# 旧栈对照注记（env_isaaclab / isaaclab 0.54.3）：本脚本 import 面为 isaaclab3
# 专有（UniformNoiseCfg / isaaclab_tasks.core.velocity 层级 / .torch ProxyArray
# 适配），旧栈跑需回退 hierarchical_kick（0.54 fork 包）+ UT 侧等价探针；旧栈
# 接触可行性已有旁证：UT cmd 臂同 spawn（球 x=1.0）训练遥测 ball_speed 1.263、
# 12.44 kick/集（X-HK2 备案）——若本探针可行则旧栈对照不必跑。
# ---------------------------------------------------------------------------
