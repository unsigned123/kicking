"""U-Kick arm D「真分层」训练入口（cmd-only；Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0）。

结构 = 官方 runner 模板（parse_env_cfg + RslRlVecEnvWrapper + runner）：
上层 RL 策略动作空间 = 3 维机体系速度命令 [vx,vy,wz]，动作 term 写入
base_velocity 命令 buffer，由**冻结低层**（U4b B 臂 model_2999.pt，
deterministic mean）每步把命令转成 12 维关节目标（env 侧
FrozenLocoJointPositionAction 消费）。

[cmd-only 拆分] 自 UT scripts/run_upper.py 裁剪：
- 三臂 --actor 分支砍到 command（--actor 参数移除；task 双向守卫退化为
  单 task 校验）；--decoder_ckpt/--sfeasible_ckpt/--u31_config/--k_env/
  --k_sf/--c_sf/--pca-k/--cmd_support_box 随 ours/e2e 臂移除；
- UpperRunner（sfeasible 缝/命令课程/σ cap）→ stock rsl_rl OnPolicyRunner
  （cmd 臂下 head=None/课程 None/σcap=None 本就全短路；kick 指标 audit
  列为纯观测面，不进训练语义）；
- u31 yaml 面（entropy_coef 单源 0.001 等）内联为常量——值与 UT
  config/upper_ukick.yaml 逐字同源（[entropy 定值 0.001 2026-09-30 三拍]）；
- 冻结低层 ckpt 缺省按 ``--low_level_convention`` 分派（official →
  包内 ``weights/velocity_flat_official/model_walk.pt``（官方口径产物）；
  legacy → ``weights/model_2999.pt`` 旧 U4b B 臂契约；--low_level_ckpt 可
  覆写）。

用法（dog env_isaaclab，包根目录）::

    python hierarchical_kick/scripts/train.py --num_envs 64 --max_iterations 2

    # resume（绝对总预算语义）
    ... --resume --checkpoint model_1000.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

# ① 显式注册拉起（字符串懒加载，零 eager 导入；离线无 gymnasium 静默跳过）
import hierarchical_kick.tasks  # noqa: F401
from hierarchical_kick.tasks import KICK_CMD_TASK_ID

parser = argparse.ArgumentParser(
    description="U-Kick arm D「真分层」train upper RL policy"
                "（3 维速度命令 + 冻结低层）")
parser.add_argument("--task", type=str, default=KICK_CMD_TASK_ID,
                    help="Gym task id（hierarchical_kick.tasks 注册）")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--run_name", type=str, default="upper_ukick_cmd")
parser.add_argument("--experiment_name", type=str, default=None)
# [arm D] 冻结低层：约定分派（official 缺省）+ ckpt 路径（None = 按约定缺省，
# 构造期存在性 fail-fast）
parser.add_argument("--low_level_convention", type=str,
                    choices=["official", "legacy"], default="official",
                    help="[arm D] 低层约定（official=官方 velocity-flat 口径，"
                         "缺省；legacy=旧 U4b B 臂契约——旧 48 布局+逐关节 "
                         "scale_vec）")
parser.add_argument("--low_level_ckpt", type=str, default=None,
                    help="[arm D] 冻结低层 ckpt（缺省按 --low_level_convention"
                         " 分派：official → velocity_flat_official/model_walk.pt、"
                         "legacy → model_2999.pt；显式传路径可覆写）")
parser.add_argument("--distributed", action="store_true", default=False,
                    help="torchrun 多卡：设备逐 rank = cuda:LOCAL_RANK")
# [X-HK1] survival 奖励剂量对照旋钮（单变量对照实验；其余权重经
# --kick_rw_overrides 通用面覆写）
parser.add_argument("--survival_weight", type=float, default=0.25,
                    help="[X-HK1] survival 奖励权重（KickRewardWeightsCfg."
                         "survival，缺省 0.25 = 现行基线；--kick_rw_overrides"
                         " 同键时本项优先级低）")
parser.add_argument("--kick_rw_overrides", type=str, default=None,
                    help="[X-HK1] 奖励权重覆写 JSON（键=KickRewardWeightsCfg"
                         " 字段名，如 '{\"survival\": 0.0}'；与 "
                         "--survival_weight 同时提供时本项优先）")
# [X-HK2] PPO 旋钮（缺省 = 现状 24/0.99；实例级覆写 agents cfg——类定义不动）
parser.add_argument("--num_steps_per_env", type=int, default=24,
                    help="[X-HK2] PPO rollout 长度（agents cfg "
                         "num_steps_per_env，缺省 24 = 现状；D/E 臂 96）")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="[X-HK2] PPO 折扣因子（agents cfg algorithm.gamma，"
                         "缺省 0.99 = 现状；D/E 臂 0.998）")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- launch 缺省分流 + 守卫（AppLauncher 拉起前 fail-fast） ---- #
# [cmd-only] 单 task 面：非 cmd task id 一律拒绝（UT 三臂/Seg/oracle 面
# 已随拆分移除）。
if args_cli.task != KICK_CMD_TASK_ID:
    raise SystemExit(
        f"[hkick][armD] 仅支持 cmd 任务 id: {KICK_CMD_TASK_ID}"
        f"（收到 --task {args_cli.task}；velocity/三臂/Seg/oracle 面不在"
        "本包——真分层 cmd 面之外请回 UT 源树）")
# [arm D] 冻结资产路径守卫（约定分派缺省 + 存在性 fail-fast；md5 资产锁与
# provenance.json 已随 UT v1.0 剥离 owner 裁决移除——本包继承该形态，只做
# 路径存在性）。
# [isaaclab 3.0 迁移] pre-launch 零 torch 纪律：经 core/weights import 会
# 触发 core/__init__ eager 拉起 torch/rsl_rl——Kit 110 对 preloaded torch
# 不容忍（pxr 预加载冲突 + OpenBLAS×fork segfault），故消费顶层零依赖
# 模块 frozen_weights（core/weights 保持 re-export 单源）。
from hierarchical_kick.frozen_weights import (  # noqa: E402
    DEFAULT_LOW_LEVEL_CKPT,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
)
if args_cli.low_level_ckpt is None:
    args_cli.low_level_ckpt = (DEFAULT_LOW_LEVEL_CKPT_OFFICIAL
                               if args_cli.low_level_convention == "official"
                               else DEFAULT_LOW_LEVEL_CKPT)
_low_path = os.path.abspath(args_cli.low_level_ckpt)
if not os.path.isfile(_low_path):
    raise SystemExit(
        f"[hkick][armD] 低层 ckpt 不存在: {_low_path}"
        f"（--low_level_convention={args_cli.low_level_convention} 的包内"
        "缺省或 --low_level_ckpt 显式路径；official → "
        "hierarchical_kick/weights/velocity_flat_official/、legacy → "
        "hierarchical_kick/weights/model_2999.pt）")
print(f"[hkick][armD] 真分层（冻结低层）资产守卫通过: {_low_path} "
      f"(convention={args_cli.low_level_convention})")

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

# [U-Kick kick 面 PPO 配方] entropy_coef 单源（原 config/upper_ukick.yaml
# 唯一被 cmd 臂消费的标量；[entropy 定值 0.001 2026-09-30 三拍] 两极实测
# 后取中——0.003 时 std 单调爆、0.0 探索消亡。三臂同改，公平面在 PPO 超参）。
ENTROPY_COEF = 0.001


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
    # [arm D] 冻结低层 ckpt 路径 + 约定注入（env cfg 单点；动作 term
    # 构造期存在性 fail-fast——launch 前守卫已过，此处注入绝对路径）
    env_cfg.actions.joint_pos.low_level_ckpt_path = \
        os.path.abspath(args_cli.low_level_ckpt)
    env_cfg.actions.joint_pos.low_level_convention = \
        str(args_cli.low_level_convention)

    # [X-HK1] survival 剂量对照旋钮：survival_weight（缺省 0.25 = 现行基线）
    # + 通用 kick_rw_overrides（JSON，键 = KickRewardWeightsCfg 字段名；
    # 二者同给时 overrides 优先）→ 单源字段覆写后 sync_kick_reward_weights
    # 幂等同步到已挂 RewTerm（RewardManager env 构造期才读权重）。
    from dataclasses import fields as _dc_fields, is_dataclass as _is_dc
    _w = getattr(env_cfg, "kick_reward_weights", None)
    if not _is_dc(_w):
        raise SystemExit("[hkick][X-HK1] env cfg 缺 kick_reward_weights 单源"
                         f"（got {_w!r}；本旋钮仅适用 kick 面 env cfg）")
    _valid = {f.name for f in _dc_fields(_w)}
    _w.survival = float(args_cli.survival_weight)
    if args_cli.kick_rw_overrides is not None:
        try:
            _overrides = json.loads(args_cli.kick_rw_overrides)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"[hkick][X-HK1] --kick_rw_overrides 非合法 JSON: {e}") from e
        if not isinstance(_overrides, dict) or not _overrides:
            raise SystemExit(
                "[hkick][X-HK1] --kick_rw_overrides 须为非空 JSON object"
                f"（got {args_cli.kick_rw_overrides!r}）")
        for _k, _v in _overrides.items():
            if _k not in _valid:
                raise SystemExit(
                    f"[hkick][X-HK1] 未知权重字段 {_k!r}"
                    f"（合法面 = KickRewardWeightsCfg: {sorted(_valid)}）")
            setattr(_w, _k, float(_v))
    env_cfg.sync_kick_reward_weights()
    print(f"[hkick][X-HK1] 最终 KickRewardWeights（{len(_dc_fields(_w))} 项，"
          "已 sync_kick_reward_weights）:")
    for _f in _dc_fields(_w):
        print(f"    {_f.name:<26s} = {getattr(_w, _f.name)}")

    # 2) agent cfg --------------------------------------------------------- #
    from hierarchical_kick.agents import UpperKickCmdPPORunnerCfg as _RunnerCfg
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
    # [X-HK2] PPO 旋钮实例级覆写（构造期改 UpperKickCmdPPORunnerCfg 实例
    # 属性——类定义/公平性锚不动；缺省 24/0.99 = 现状逐位不变）
    agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
    agent_cfg.algorithm.gamma = float(args_cli.gamma)
    agent_cfg.algorithm.entropy_coef = ENTROPY_COEF
    from importlib.metadata import version as _pkg_version
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, _pkg_version("rsl-rl-lib"))

    # [X-HK2] 最终 PPO 面核对（num_steps/gamma/entropy/survival/权重表；
    # 权重表已由 X-HK1 sync 段打印——此处打印 PPO 侧 + survival 锚行）
    print(f"[hkick][X-HK2] 最终 PPO 面: num_steps_per_env="
          f"{agent_cfg.num_steps_per_env}  gamma="
          f"{agent_cfg.algorithm.gamma}  entropy_coef="
          f"{agent_cfg.algorithm.entropy_coef}  lam="
          f"{agent_cfg.algorithm.lam}  lr={agent_cfg.algorithm.learning_rate}"
          f"({agent_cfg.algorithm.schedule})  survival="
          f"{env_cfg.kick_reward_weights.survival}  "
          f"kick_event={env_cfg.kick_reward_weights.kick_event}  "
          f"max_iterations={agent_cfg.max_iterations}  "
          f"seed={agent_cfg.seed}")

    # [X-HK5] spawn 课程 + 球出生随机化调度核对（纯 cfg 数学，env 构造前；
    # common_step_counter 口径 = 每 env.step +1（manager_based_rl_env.py:249）
    # → 实际步/it = num_steps_per_env；steps_per_iter cfg 缺省按此校准）
    from hierarchical_kick.sim.mdp.ball_mdp import (
        CURRICULUM_D_MAX_START,
        CURRICULUM_Y_MARGIN,
        GOAL_Y,
        ball_spawn_d_max,
    )
    _kr = env_cfg.kick_randomization
    _warmup_steps = float(_kr.warmup_iters) * float(_kr.steps_per_iter)
    _spi_actual = max(float(agent_cfg.num_steps_per_env), 1.0)
    _dmax0 = ball_spawn_d_max(0.0, _warmup_steps, d_max_final=float(_kr.d_max_final))
    _dmax_end = ball_spawn_d_max(
        float(agent_cfg.max_iterations) * _spi_actual, _warmup_steps,
        d_max_final=float(_kr.d_max_final))
    _it_full_actual = _warmup_steps / _spi_actual
    print(f"[hkick][X-HK5] spawn 课程面: enabled={_kr.enabled}  "
          f"curriculum={_kr.curriculum_enabled}  d_min={_kr.d_min}  "
          f"d_max={CURRICULUM_D_MAX_START}→{_kr.d_max_final}  "
          f"warmup_iters={_kr.warmup_iters}×steps_per_iter="
          f"{_kr.steps_per_iter}→warmup_steps={_warmup_steps:.0f}  "
          f"ball_xy_range={_kr.ball_xy_range}  "
          f"y_max=GOAL_Y-{CURRICULUM_Y_MARGIN}={GOAL_Y - CURRICULUM_Y_MARGIN:.1f}")
    print(f"[hkick][X-HK5] d_max 调度: it=0 → {_dmax0:.3f}m  "
          f"it={agent_cfg.max_iterations} → {_dmax_end:.3f}m"
          f"(实际日历 {agent_cfg.num_steps_per_env} 步/it)  "
          f"满域 iter: warmup_iters={_kr.warmup_iters}"
          f"(nominal {_kr.steps_per_iter} 步/it) / "
          f"actual≈{_it_full_actual:.0f}")
    if _kr.curriculum_enabled and abs(_it_full_actual - float(_kr.warmup_iters)) > max(
            0.05 * float(_kr.warmup_iters), 1.0):
        print(f"[hkick][X-HK5][WARN] steps_per_iter 缺省口径"
              f"({_kr.steps_per_iter}) 与实际 rollout({_spi_actual:.0f} 步/it)"
              f"不一致 → 实际满域在 iter≈{_it_full_actual:.0f}"
              f"（非 {_kr.warmup_iters}）——核对 warmup 口径")

    # 3) env + 关节序断言 -------------------------------------------------- #
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()  # scene spawn 完成（joint_names / group_obs_dim 可读面）
    sim_env = env.unwrapped
    joint_names = list(sim_env.scene["robot"].joint_names)
    print(f"[hkick] kick env joint order ({len(joint_names)}): {joint_names}")
    if set(joint_names) != set(GO2_JOINT_ORDER_IL):
        raise SystemExit(
            f"[hkick] env 关节集合漂移: {joint_names} != "
            f"{list(GO2_JOINT_ORDER_IL)}（fail-fast；832854c 教训）")
    if joint_names != list(GO2_JOINT_ORDER_IL):
        print(f"[hkick][WARN] 关节序与 GO2_JOINT_ORDER_IL 有序差 "
              f"{joint_names} vs {list(GO2_JOINT_ORDER_IL)}"
              "（scale_vec/std_param 槽位序按 GO2_JOINT_ORDER_IL，请核对）")
    gdim = {str(k): int(v[-1]) for k, v in
            sim_env.observation_manager.group_obs_dim.items()}
    print("[hkick] runtime group_obs_dim:", gdim)
    # [X-HK4→X-HK5] spawn 核对（一次性；env.reset 后的 init 位姿/奖励流首值
    # ——[X-HK5] 起球出生随机化，ball_distance/位置为 env0 首采样值，dog
    # 恒 (0,0) yaw=π/2；goal_scored 必须 False（球 y ≤ GOAL_Y−0.2 未过线））
    from hierarchical_kick.sim.mdp import ball_mdp as _bmdp
    _r0, _b0 = sim_env.scene["robot"], sim_env.scene["ball"]
    _bd_T = float(env_cfg.rewards.ball_distance.params["temperature"])
    _bd = float(env_cfg.rewards.ball_distance.weight) * float(
        _bmdp.ball_distance_reward(sim_env, temperature=_bd_T)[0])
    _bh = float(env_cfg.rewards.behind_ball.weight) * float(
        _bmdp.behind_ball_reward(sim_env)[0])
    _dist0 = float(torch.linalg.vector_norm(
        _b0.data.root_pos_w.torch[0, :2] - _r0.data.root_pos_w.torch[0, :2]))
    print(f"[hkick][X-HK5] spawn 首采样核对: "
          f"dog_pos={tuple(round(v, 4) for v in _r0.data.root_pos_w.torch[0, :3].tolist())} "
          f"yaw={float(_bmdp._yaw_from_quat(_r0.data.root_quat_w.torch[0:1])[0]):.4f}(期望≈{math.pi / 2:.4f}) "
          f"ball_pos={tuple(round(v, 4) for v in _b0.data.root_pos_w.torch[0, :3].tolist())} "
          f"dist0={_dist0:.4f}(d_max={_bmdp.ball_spawn_d_max(float(sim_env.common_step_counter), float(env_cfg.kick_randomization.warmup_iters) * float(env_cfg.kick_randomization.steps_per_iter), d_max_final=float(env_cfg.kick_randomization.d_max_final)):.3f}) "
          f"ball_rel_b[0]={[round(v, 4) for v in _bmdp.ball_rel_b(sim_env)[0].tolist()]} "
          f"ball_distance[0]={_bd:.4f}(T={_bd_T}) behind[0]={_bh:.4f} "
          f"goal_scored[0]={bool(_bmdp.goal_scored(sim_env)[0])}")
    # obs 维契约：cmd 面恒 kick 55（KICK_POLICY_OBS_DIM，mirror
    # ball_mdp.KICK_OBS_DIM——kick_metrics import 期交叉断言防漂移）
    from hierarchical_kick.core.kick_metrics import \
        KICK_POLICY_OBS_DIM as _KICK_OBS_DIM
    if gdim.get("policy") != _KICK_OBS_DIM:
        raise SystemExit(
            f"[hkick] policy obs 维 {gdim.get('policy')} != "
            f"{_KICK_OBS_DIM}（kick 55 维契约装配漂移，fail-fast）")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 4) 日志目录 + 配置落盘 ------------------------------------------------ #
    log_dir = _make_log_dir(agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print_dict({"num_envs": env_cfg.scene.num_envs,
                "device": agent_cfg.device,
                "experiment_name": agent_cfg.experiment_name,
                "entropy_coef": ENTROPY_COEF,
                "survival_weight": env_cfg.kick_reward_weights.survival,
                "num_steps_per_env": agent_cfg.num_steps_per_env,
                "gamma": agent_cfg.algorithm.gamma,
                "low_level_convention": str(args_cli.low_level_convention),
                "low_level_ckpt": os.path.abspath(args_cli.low_level_ckpt)})

    # 5) runner + 训练（stock OnPolicyRunner——cmd 臂无缝/课程/σ cap 面） -- #
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
        print(f"[hkick] resume @ iter {start_it} -> remaining iterations "
              f"{num_iters} (max_iterations 总预算不变)")

    runner.learn(num_learning_iterations=num_iters,
                 init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
