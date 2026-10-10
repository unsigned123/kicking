"""官方口径 velocity 行走门探针（--robot 分派：go2 / cyberdog2）。

**用途**：velocity 低层 ckpt 的行走质量判定（与 go2 model_walk.pt 同门）：
加载策略（rsl_rl OnPolicyRunner ckpt 的 actor MLP），常量机体系直行命令
[cmd_vx, 0, 0] 驱动 velocity env，逐段打印 v_body / 步态振荡 / 球位移。

**方法**：parse_env_cfg + gym.make（不挂 RslRlVecEnvWrapper——裸 obs dict
"policy" 面），确定性化（reset pose/velocity range 钳零、joint scale 钳 1、
push/外力事件关），命令 term 逐步覆写常量；--ball 时 scene 挂球
（kick env 同款：r0.11/m0.43/高阻尼），spawn 于行进方向 ball_x（缺省
0.3m）处。

**行走门**（三判据全过 = 会走）：
  ① v_body ≥ 0.25 m/s（站桩 ≈0）
  ② 步态振荡：对角大腿对（go2=thigh / cyberdog2=hip）joint_vel 符号翻转
     >10 且 |mean| > 1 rad/s
  ③ 推球位移 > 0.2 m（球 spawn 沿行进方向；段锚点 done 重锚）

用法（dog env_isaaclab3，包根目录）::

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=<空闲卡> \\
    PYTHONPATH=$PWD python hierarchical_kick/scripts/probe_velocity_walk.py \\
        --robot cyberdog2 --ckpt logs/rsl_rl/hk_velocity_flat_cyberdog2/<run>/model_999.pt
"""
from __future__ import annotations

import argparse
import os

# ① 显式注册拉起（字符串懒加载；与 train_velocity.py 同款）
import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import (  # noqa: F401
    HK_VELOCITY_CYBERDOG2_TASK_ID,
    HK_VELOCITY_TASK_ID,
)

parser = argparse.ArgumentParser(
    description="velocity 行走门探针（常量直行命令 + 策略 rollout，判 "
                "v_body/步态/推球三判据）")
parser.add_argument("--robot", type=str, default="go2",
                    choices=["go2", "cyberdog2"],
                    help="本体 profile（分派 task/步态关节/契约面）")
parser.add_argument("--task", type=str, default=None,
                    help="Gym task id（缺省按 --robot 分派）")
parser.add_argument("--ckpt", type=str, required=True,
                    help="velocity ckpt（rsl_rl OnPolicyRunner model_*.pt）")
parser.add_argument("--num_envs", type=int, default=1,
                    help="env 数（探针单 env 足够）")
parser.add_argument("--steps", type=int, default=250,
                    help="rollout 步数（50Hz；250 步 = 5 秒）")
parser.add_argument("--print_every", type=int, default=25,
                    help="打印间隔（步）")
parser.add_argument("--cmd_vx", type=float, default=0.4,
                    help="常量直行命令幅值 [cmd_vx, 0, 0]（m/s 机体系）")
parser.add_argument("--ball", action="store_true", default=True,
                    help="scene 挂球（推球判据；--no_ball 关）")
parser.add_argument("--no_ball", dest="ball", action="store_false")
parser.add_argument("--ball_x", type=float, default=0.3,
                    help="球 spawn x（行进方向前方距离，m）")
parser.add_argument("--seed", type=int, default=42)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- launch 缺省分流 + 守卫（AppLauncher 拉起前 fail-fast） ---- #
_TASK_BY_ROBOT = {
    "go2": HK_VELOCITY_TASK_ID,
    "cyberdog2": HK_VELOCITY_CYBERDOG2_TASK_ID,
}
if args_cli.task is None:
    args_cli.task = _TASK_BY_ROBOT[args_cli.robot]
elif args_cli.task != _TASK_BY_ROBOT[args_cli.robot]:
    raise SystemExit(
        f"[hkick][probe] --task 与 --robot 不一致: robot={args_cli.robot} "
        f"须 task {_TASK_BY_ROBOT[args_cli.robot]}（收到 --task "
        f"{args_cli.task}）")
_ckpt_path = os.path.abspath(args_cli.ckpt)
if not os.path.isfile(_ckpt_path):
    raise SystemExit(f"[hkick][probe] ckpt 不存在: {_ckpt_path}")
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- isaaclab / torch 可用区（pre-launch 零 torch import 纪律） ---

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from hierarchical_kick.robots import (  # noqa: E402
    CYBERDOG2_JOINT_ORDER_IL,
    GO2_JOINT_ORDER_IL,
)
from hierarchical_kick.utils.parse_cfg import parse_env_cfg  # noqa: E402

# 步态振荡探针对（对角腿大腿关节；go2=thigh / cyberdog2=hip 命名差）
_GAIT_JOINTS_BY_ROBOT = {
    "go2": ("FL_thigh_joint", "RR_thigh_joint"),
    "cyberdog2": ("FL_hip_joint", "RR_hip_joint"),
}


class _ActorMlp(nn.Module):
    """rsl_rl ckpt actor MLP 复原（actor.{2i} Linear 键 + ELU；末层无激活）。"""

    def __init__(self, state_dict: dict):
        super().__init__()
        idxs = sorted({int(k.split(".")[1]) for k in state_dict
                       if k.startswith("actor.")})
        linears = []
        for i in idxs:
            w = state_dict[f"actor.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data = w.clone()
            lin.bias.data = state_dict[f"actor.{i}.bias"].clone()
            linears.append(lin)
        self.layers = nn.ModuleList(linears)
        self.act = nn.ELU()

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


def main() -> None:
    # 1) env cfg（确定性化：探针语义——零初始采样、零扰动） -------------- #
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in
        ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None

    if args_cli.ball:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        env_cfg.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.11,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=0.95,
                    restitution=0.25),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=2.5,
                    angular_damping=2.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(args_cli.ball_x), 0.0, 0.11),
                rot=(0.0, 0.0, 0.0, 1.0)),
        )
        print(f"[hkick][probe] 球 spawn: "
              f"{tuple(env_cfg.scene.ball.init_state.pos)}（行进前方 "
              f"{args_cli.ball_x}m）")

    # 2) env + 策略（裸 obs dict 面） ------------------------------------ #
    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_dict, _ = env.reset()
    sim_env = env.unwrapped
    device = sim_env.device
    robot = sim_env.scene["robot"]

    joint_names = list(robot.joint_names)
    expected = (GO2_JOINT_ORDER_IL if args_cli.robot == "go2"
                else CYBERDOG2_JOINT_ORDER_IL)
    print(f"[hkick][probe] joint order ({len(joint_names)}): {joint_names}")
    if set(joint_names) != set(expected):
        raise SystemExit(f"[hkick][probe] 关节集合漂移: {joint_names} != "
                         f"{list(expected)}")

    ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    actor = _ActorMlp(state).to(device).eval()
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[hkick][probe] actor: {len(actor.layers)} 层 {n_params} 参数"
          f"（ckpt iter={ckpt.get('iterations', '?')}）")

    cmd_term = sim_env.command_manager.get_term("base_velocity")
    const_cmd = torch.tensor([args_cli.cmd_vx, 0.0, 0.0], device=device)

    gait_joints = _GAIT_JOINTS_BY_ROBOT[args_cli.robot]
    gait_idx = [joint_names.index(j) for j in gait_joints]
    ball_anchor = None
    qd_hist: list = []
    v_body_hist: list = []

    def _obs() -> torch.Tensor:
        obs = obs_dict["policy"]
        return obs if isinstance(obs, torch.Tensor) else obs["policy"]

    def _reanchor():
        nonlocal ball_anchor
        if args_cli.ball:
            ball_anchor = sim_env.scene["ball"].data.root_pos_w.detach()[:, :2].clone()
        else:
            ball_anchor = None

    _reanchor()
    print(f"\n[hkick][probe] cmd=({args_cli.cmd_vx},0,0) | step | v_body_x "
          f"(mean) | 步态 |qmean| 翻转 | ball Δ (mean/max)")
    with torch.inference_mode():
        for step in range(1, int(args_cli.steps) + 1):
            cmd_term.vel_command_b[:] = const_cmd  # 常量命令覆写（逐位）
            obs_dict, _rew, dones, _extras = env.step(actor(_obs()))
            v_body = float(robot.data.root_lin_vel_b[:, 0].mean())
            v_body_hist.append(v_body)
            qd_hist.append(robot.data.joint_vel[0, gait_idx].detach().cpu())
            if args_cli.ball:
                ball_pos = sim_env.scene["ball"].data.root_pos_w.detach()
                disp = torch.linalg.vector_norm(
                    ball_pos[:, :2] - ball_anchor, dim=-1)
            else:
                disp = None
            done_mask = dones.detach().to(torch.bool).reshape(-1)
            if bool(done_mask.any()):
                _reanchor()  # done env 已 auto-reset → 重锚
            if step % int(args_cli.print_every) == 0 or step == 1:
                qd_now = robot.data.joint_vel[0, gait_idx].detach()
                ball_s = (f"| ball Δ={float(disp.mean()):.3f}/"
                          f"{float(disp.max()):.3f}m" if disp is not None
                          else "| (no ball)")
                print(f"[hkick][probe] {step:>4d} | v_body={v_body:+.3f} m/s "
                      f"| 步态 qd=({float(qd_now[0]):+.2f},"
                      f"{float(qd_now[1]):+.2f}) rad/s {ball_s}")

    # 3) 行走门判定 -------------------------------------------------------- #
    qd = torch.stack(qd_hist)  # (T, 2)
    flips = int((torch.diff(torch.sign(qd[:, 0])) != 0).sum())
    qd_mean = float(qd[:, 0].abs().mean())
    v_last = float(sum(v_body_hist[-50:]) / max(1, len(v_body_hist[-50:])))
    v_peak = float(max(v_body_hist))
    print("\n==================== 行走门判定 ====================")
    print(f"① v_body: 末 50 步均值={v_last:.4f} m/s  峰值={v_peak:.4f} "
          f"（门 ≥0.25 → {'PASS' if v_last >= 0.25 else 'FAIL'}）")
    print(f"② 步态振荡（{gait_joints[0]} qd）: |mean|={qd_mean:.3f} rad/s "
          f"std={float(qd[:, 0].std()):.3f} 符号翻转={flips}"
          f"（门 翻转>10 且 |mean|>1 → "
          f"{'PASS' if flips > 10 and qd_mean > 1.0 else 'FAIL'}）")
    if args_cli.ball:
        ball_pos = sim_env.scene["ball"].data.root_pos_w.detach()
        final = torch.linalg.vector_norm(ball_pos[:, :2] - ball_anchor, dim=-1)
        d_max = float(final.max())
        print(f"③ 推球位移: max={d_max:.4f} m mean={float(final.mean()):.4f} "
              f"（门 >0.2 → {'PASS' if d_max > 0.2 else 'FAIL'}）")
        gate = (v_last >= 0.25 and flips > 10 and qd_mean > 1.0 and d_max > 0.2)
    else:
        gate = (v_last >= 0.25 and flips > 10 and qd_mean > 1.0)
    print(f"\n>>> 行走门: {'**PASS**（会走）' if gate else '**FAIL**'}")
    print("===================================================")

    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
