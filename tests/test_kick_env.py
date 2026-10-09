"""U-Kick 球 env WP2 — ball_mdp 数学面 + 常量契约 + cfg/register 源码断言。

判据（docs/UKICK_BALL_ENV_DESIGN.md §3 规格、§7 拍板、§8-WP2 名称契约；
奖励/终止规则接口 = owner 二次拍板 2026-09-30 对齐 soccer_absolute phase0，
权重按 go2 平台适配单源 KickRewardWeightsCfg）：
- ball_mdp 纯函数数学：ball_rel_b（yaw 0/±90° 手算 body 系）、goal_rel_b
  （sin²+cos²=1 恒等式 + 方向）、push_geometry_b（规范位形 cos_behind≈1、
  零向量防除零）、goal_scored 真假例、ball_out 半径边界；
  奖励规则族 donor 数值语义单测：robot_fallen 双条件（重力/高度支 + 严格
  边界）、approach_ball（投影/封顶/正负支/距离窗/低速门）、ball_distance
  （exp 核）、behind_ball（球后位形核 + 方向参数）、ball_progress_dense
  （Δy 单边 clamp + 近球门控 + 缓存生命周期 + reset 免疫）、
  ball_out_of_bounds（mask ≡ ball_out）、robot_fallen_penalty（1.0 mask）、
  robot_out_of_field（工作区框 + 逐 env origin 帧）、wall_stuck（dist+speed
  双条件，接口在位 cfg 不挂载）；
  逐 env goal 帧回归（A100 10-env oracle 实测确诊的世界系 bug：判线/出界/
  方向 obs 以 env_origins + GOAL_CENTER 为锚，远端 env 不再误判）；
  旧 ball_progress_shaping prev-cur 缓存面移除防呆（函数面/缓存属性名/
  reset 钩子消费点零残留）。
- 常量与维度：KICK_OBS_DIM == 55 == 42+3+3+4+3；球参数/摔判定/工作区框
  常量 == donor 值或注释论证值。
- ball_env_cfg.py 源码断言：task-only 置 None 面、权重单源
  KickRewardWeightsCfg 字段默认值、8 项奖励挂载 + survival 显式化、
  终止面（root_height→robot_fallen 双条件替换 + robot_out_of_field +
  wall_stuck 不挂载）、episode 10s、随机化接口默认关、Oracle 面恢复
  commands/UpperPolicyCfg、球参数逐值。
- register 双任务 id + entry point 指向正确 cfg 类名字符串。

离线（无 isaaclab）：ball_mdp 是唯一离线可 import 的 mdp 模块（纯 torch +
仓库离线数学单源），但父包 upper/sim/mdp/__init__ 顶层 import isaaclab →
直接 import 子模块会连带拉起父包。故按 test_body_vel_fk 同款 importlib
spec 面按路径加载 ball_mdp（不 gate）；cfg 文件用源码/AST 断言。
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import types

import pytest
import torch

_PKG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_PKG_ROOT = os.path.abspath(_PKG_ROOT)
_BALL_MDP_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "sim", "mdp", "ball_mdp.py")
_BALL_CFG_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "sim", "ball_env_cfg.py")
_REGISTER_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "tasks", "__init__.py")
_MDP_INIT_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "sim", "mdp", "__init__.py")

_YAW_Q0 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # yaw=0（单位四元数 xyzw；[isaaclab 3.0 迁移] wxyz→xyzw）
_HALF = math.sqrt(0.5)
_YAW_QP90 = torch.tensor([[0.0, 0.0, _HALF, _HALF]])  # yaw=+90°（xyzw）
_YAW_QM90 = torch.tensor([[0.0, 0.0, -_HALF, _HALF]])  # yaw=-90°（xyzw）


def _load_ball_mdp():
    """按路径加载 ball_mdp（父包 __init__ 含 isaaclab 面，离线不可连带拉起）。"""
    spec = importlib.util.spec_from_file_location("hkick_ball_mdp_offline", _BALL_MDP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ball_mdp = _load_ball_mdp()


# --------------------------------------------------------------------------- #
# fake env（wrapper 级驱动；manager-based 取数面 env.scene["robot"/"ball"]）
# --------------------------------------------------------------------------- #

class _FakeScene(dict):
    """scene dict + ``env_origins`` 属性面（wrapper 逐 env goal 帧取数）。"""


def _fake_env(base_pos: torch.Tensor, quat: torch.Tensor, ball_pos: torch.Tensor,
              env_origins: torch.Tensor | None = None,
              lin_vel_w: torch.Tensor | None = None,
              proj_grav_z: torch.Tensor | None = None,
              lin_vel_b: torch.Tensor | None = None):
    """env.scene 取数面 stub（ball_mdp wrapper 消费 root_pos_w/root_quat_w/
    root_lin_vel_w/root_lin_vel_b/projected_gravity_b）。

    env_origins=None → 全零（单 env 原点场景，与修前逐位一致）；逐 env goal
    帧回归测试显式传网格 origins（如 [[0,0,0],[10,0,0]]）。lin_vel_w/lin_vel_b
    缺省零速度；proj_grav_z 缺省 −1（直立站姿）。
    """
    n = base_pos.shape[0]

    class _Proxy:
        """[isaaclab 3.0 迁移] 数据面镜像：.data 属性 = ProxyArray 形态
        （.torch 取 torch 视图；本桩零拷贝回 self）。"""

        def __init__(self, t: torch.Tensor):
            self._t = t

        @property
        def torch(self) -> torch.Tensor:
            return self._t

    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_pos_w=_Proxy(base_pos), root_quat_w=_Proxy(quat),
        root_lin_vel_w=_Proxy(torch.zeros_like(base_pos) if lin_vel_w is None else lin_vel_w),
        root_lin_vel_b=_Proxy(torch.zeros_like(base_pos) if lin_vel_b is None else lin_vel_b),
        projected_gravity_b=_Proxy(
            torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3).contiguous()
            if proj_grav_z is None else
            torch.stack([torch.zeros(n), torch.zeros(n), proj_grav_z], dim=-1)),
    ))
    ball = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=_Proxy(ball_pos)))
    scene = _FakeScene({"robot": robot, "ball": ball})
    scene.env_origins = (torch.zeros(base_pos.shape[0], 3) if env_origins is None
                         else env_origins)
    return types.SimpleNamespace(scene=scene,
                                 num_envs=base_pos.shape[0], device=base_pos.device)


# --------------------------------------------------------------------------- #
# 1. ball_mdp 纯函数数学
# --------------------------------------------------------------------------- #

def test_ball_rel_b_impl_yaw_zero_and_ninety():
    """已知 base pos/quat/ball pos 手算 body 系期望值（yaw=0/±90°）。"""
    base = torch.tensor([[0.0, 0.0, 0.2]])
    ball = torch.tensor([[1.0, 0.5, 0.1]])
    # yaw=0：body 系 = 世界系差 (1, 0.5)
    got = ball_mdp._ball_rel_b_impl(base, ball, _YAW_Q0)
    expect = torch.tensor([[1.0 / 3.0, 0.5 / 3.0, math.sqrt(1.25) / 3.0]])
    assert torch.allclose(got, expect, atol=1e-6)
    # yaw=+90°：世界 (1,0) → body (0,-1)
    got = ball_mdp._ball_rel_b_impl(base, torch.tensor([[1.0, 0.0, 0.1]]), _YAW_QP90)
    assert torch.allclose(got, torch.tensor([[0.0, -1.0 / 3.0, 1.0 / 3.0]]), atol=1e-6)
    # yaw=-90°：世界 (1,0) → body (0,+1)
    got = ball_mdp._ball_rel_b_impl(base, torch.tensor([[1.0, 0.0, 0.1]]), _YAW_QM90)
    assert torch.allclose(got, torch.tensor([[0.0, 1.0 / 3.0, 1.0 / 3.0]]), atol=1e-6)


def test_ball_rel_b_wrapper_matches_impl():
    """wrapper 数据管道（robot/ball 取数 + root_quat_w）与 impl 逐位一致。"""
    g = torch.Generator().manual_seed(3)
    n = 8
    base = torch.randn(n, 3, generator=g)
    ball = torch.randn(n, 3, generator=g)
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    env = _fake_env(base, quats, ball)
    got = ball_mdp.ball_rel_b(env)
    expect = ball_mdp._ball_rel_b_impl(base, ball, quats)
    assert got.shape == (n, 3)
    assert torch.allclose(got, expect, atol=1e-6)


def test_goal_rel_b_direction_and_sin_cos_identity():
    """方向例（yaw=0/90° 正前门）+ 随机位形 sin²+cos²=1 恒等式。"""
    base = torch.tensor([[0.0, 0.0, 0.2]])
    # yaw=0，门 (0,3) 正左方：body rel (0,3) → [0, 1, sin=1, cos=0]
    got = ball_mdp._goal_rel_b_impl(base, _YAW_Q0)
    assert torch.allclose(got, torch.tensor([[0.0, 1.0, 1.0, 0.0]]), atol=1e-6)
    # yaw=+90°，门正前：body rel (3,0) → [1, 0, 0, 1]
    got = ball_mdp._goal_rel_b_impl(base, _YAW_QP90)
    assert torch.allclose(got, torch.tensor([[1.0, 0.0, 0.0, 1.0]]), atol=1e-6)
    # 随机位形恒等式（sin²+cos²=1）+ xy/3 归一
    g = torch.Generator().manual_seed(5)
    n = 16
    bases = torch.randn(n, 3, generator=g)
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    out = ball_mdp._goal_rel_b_impl(bases, quats)
    assert out.shape == (n, 4)
    assert torch.allclose(out[:, 2] ** 2 + out[:, 3] ** 2, torch.ones(n), atol=1e-6)
    rel_xy = out[:, :2] * 3.0
    angle = torch.atan2(rel_xy[:, 1], rel_xy[:, 0])
    assert torch.allclose(out[:, 2], torch.sin(angle), atol=1e-6)
    assert torch.allclose(out[:, 3], torch.cos(angle), atol=1e-6)
    # wrapper 数据管道与 impl 一致
    env = _fake_env(bases, quats, torch.zeros(n, 3))
    assert torch.allclose(ball_mdp.goal_rel_b(env),
                          ball_mdp._goal_rel_b_impl(bases, quats), atol=1e-6)


def test_push_geometry_b_canonical_and_robustness():
    """规范位形（狗在球正后方、球指向门）cos_behind≈1 + 零向量防除零。"""
    base = torch.tensor([[0.0, 0.0, 0.2]])
    ball = torch.tensor([[0.0, 1.0, 0.11]])  # 球在狗正前（+y），门 (0,3) 同向
    got = ball_mdp._push_geometry_b_impl(base, ball, _YAW_Q0)
    # 狗→球 = 球→门 → cos_behind=1；fwd=(1,0) 垂直 btog=(0,1) → sin=1, cos=0
    assert torch.allclose(got, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-6)
    # 垂直位形：狗→球 (0,1)，球→门 (−1,0)（球 (1,3)、狗 (1,2)）→ cos_behind=0
    got = ball_mdp._push_geometry_b_impl(
        torch.tensor([[1.0, 2.0, 0.2]]), torch.tensor([[1.0, 3.0, 0.11]]), _YAW_Q0)
    assert torch.allclose(got, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)
    # 零向量（球在门心）→ 1e-8 防除零，全有限无 NaN
    got = ball_mdp._push_geometry_b_impl(base, torch.tensor([[0.0, 3.0, 0.11]]), _YAW_Q0)
    assert torch.isfinite(got).all()
    # wrapper 与 impl 一致（数据管道面）
    env = _fake_env(base, _YAW_Q0, ball)
    assert torch.allclose(ball_mdp.push_geometry_b(env),
                          ball_mdp._push_geometry_b_impl(base, ball, _YAW_Q0), atol=1e-6)


def test_goal_scored_impl_truth_table():
    """判线真假例：y 过线且 |x| < 0.75；边界严格不等号。"""
    cases = [
        (0.1, 3.5, True),   # 门内过线
        (-0.7, 4.0, True),  # 门内（负 x）
        (0.8, 3.5, False),  # |x| ≥ half_width
        (0.1, 2.99, False),  # 未过线
        (0.75, 3.5, False),  # |x| = half_width（严格 <）
        (0.0, 3.0, False),  # y = goal_y（严格 >）
    ]
    for x, y, expect in cases:
        got = ball_mdp._goal_scored_impl(torch.tensor([[x, y, 0.11]]))
        assert bool(got[0]) is expect, (x, y, expect, got)


def test_goal_scored_wrapper_and_reward():
    """wrapper 真假 + goal_scored_reward 输出 0/1 浮点（幅值由 weight 承载）。"""
    ball = torch.tensor([[0.1, 3.5, 0.11], [0.0, 0.5, 0.11]])
    env = _fake_env(torch.zeros(2, 3), _YAW_Q0.expand(2, 4), ball)
    assert ball_mdp.goal_scored(env).dtype == torch.bool
    assert ball_mdp.goal_scored(env).tolist() == [True, False]
    rew = ball_mdp.goal_scored_reward(env)
    assert torch.allclose(rew, torch.tensor([1.0, 0.0]))
    assert rew.dtype == torch.float32


def test_ball_out_impl_radius_boundary():
    """出界半径边界：dist == radius 不触发（严格 >），越过即触发；center 可换。"""
    # 门 (0,3)，球 (0,9)：dist = 6.0 → 不触发
    assert bool(ball_mdp._ball_out_impl(torch.tensor([[0.0, 9.0, 0.11]]))[0]) is False
    # (0,9.001)：dist > 6 → 触发
    assert bool(ball_mdp._ball_out_impl(torch.tensor([[0.0, 9.001, 0.11]]))[0]) is True
    # 横向越界：dist = 6.1 → 触发
    assert bool(ball_mdp._ball_out_impl(torch.tensor([[6.1, 3.0, 0.11]]))[0]) is True
    # center 参数可换（语义纯参数化）
    assert bool(ball_mdp._ball_out_impl(torch.tensor([[11.0, 0.0, 0.11]]), center=(5.0, 0.0))[0]) is False
    assert bool(ball_mdp._ball_out_impl(torch.tensor([[11.1, 0.0, 0.11]]), center=(5.0, 0.0))[0]) is True
    # [N,2] 逐 env tensor 入口（wrapper 同款；tuple 广播面 == 修前逐位一致）
    ball2 = torch.tensor([[11.0, 0.0, 0.11], [16.9, 0.0, 0.11]])
    centers = torch.tensor([[5.0, 0.0], [10.0, 0.0]])
    assert ball_mdp._ball_out_impl(ball2, center=centers).tolist() == [False, True]


# --------------------------------------------------------------------------- #
# 1b. 逐 env goal 帧（核心回归；A100 10-env oracle 实测确诊的世界系 bug）
# --------------------------------------------------------------------------- #

def test_goal_scored_per_env_frame():
    """判线以逐 env goal center 为原点：远端 env 世界系坐标不得误判。

    env_origins = env0 (0,0) / env1 (10,0)。env1 球在 origin+(0,2.9) →
    False（未过线）；origin+(0,3.1) 且 |env 系 x|<0.75 → True；球在世界系
    (0,3.5)（env 系 x=−10）→ False——旧世界系版本此处误 True（|x|<0.75、
    y>3 逐字面命中），修复后必须为 False。
    """
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    ball = torch.tensor([
        [0.1, 3.5, 0.11],    # env0：门内过线 → True
        [10.0, 2.9, 0.11],   # env1 env 系 (0, 2.9)：未过线 → False
        [10.5, 3.1, 0.11],   # env2 env 系 (0.5, 3.1)：过线门内 → True
        [0.0, 3.5, 0.11],    # env3 世界系 (0,3.5)：世界系判线误 True → env 系 False
        [11.0, 3.5, 0.11],   # env4 env 系 x=1.0 ≥ half_width → False
    ])
    env = _fake_env(torch.zeros(5, 3), _YAW_Q0.expand(5, 4), ball,
                    env_origins=origins)
    got = ball_mdp.goal_scored(env)
    assert got.dtype == torch.bool
    assert got.tolist() == [True, False, True, False, False]
    # goal_scored_reward 与 DoneTerm 同帧同判（0/1 浮点）
    rew = ball_mdp.goal_scored_reward(env)
    assert torch.allclose(rew, torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]))


def test_goal_scored_impl_tensor_goal_matches_world_when_origin_zero():
    """impl 纯函数面 [N,2] tensor goal：origin=0 广播面与旧版本逐位一致。"""
    cases = [(0.1, 3.5, True), (0.8, 3.5, False), (0.1, 2.99, False)]
    for x, y, expect in cases:
        got = ball_mdp._goal_scored_impl(
            torch.tensor([[x, y, 0.11]]), goal_center=torch.tensor([[0.0, 3.0]]))
        assert bool(got[0]) is expect


def test_ball_out_per_env_frame():
    """出界半径以逐 env goal center 为心：远端 env 不得 insta-terminate。

    env1（origin (10,0)）球在 origin+(1,0)：dist=1 → False（世界系下距
    (0,3)≈10.05>6 会误 True——oracle 1008 episodes 1000 次 BALL_OUT 死循环
    根因）；origin+(7,0)：dist=7>6 → True。
    """
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    ball = torch.tensor([
        [0.0, 9.0, 0.11],    # env0：距门 (0,3) 恰 6.0（严格 > 不触发）→ False
        [11.0, 0.0, 0.11],   # env1 env 系 (1,0)：dist=1 → False（世界系误 True）
        [17.0, 0.0, 0.11],   # env2 env 系 (7,0)：dist=7 → True
    ])
    env = _fake_env(torch.zeros(3, 3), _YAW_Q0.expand(3, 4), ball,
                    env_origins=origins)
    assert ball_mdp.ball_out(env).tolist() == [False, False, True]


def test_goal_rel_b_per_env_frame():
    """goal 方向 obs 以 env 系 goal 为锚：相同 env 系位形 → 逐 env obs 一致。

    env1（origin (10,0)）机器人在 origin+(0,1)：goal 在 env 系 (0,3) →
    body rel (0,2) → [0, 2/3, sin=1, cos=0]。世界系常量 goal (0,3) 下
    diff=(−10,2) → rel_y<0（指向 −y 的错误值）。
    """
    n = 2
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    base = torch.tensor([[0.0, 1.0, 0.2], [10.0, 1.0, 0.2]])
    env = _fake_env(base, _YAW_Q0.expand(n, 4), torch.zeros(n, 3),
                    env_origins=origins)
    got = ball_mdp.goal_rel_b(env)
    expect_row = torch.tensor([0.0, 2.0 / 3.0, 1.0, 0.0])
    assert torch.allclose(got[0], expect_row, atol=1e-6)
    assert torch.allclose(got[1], expect_row, atol=1e-6)
    # bug 面钉：同一 env1 位形喂世界系常量 goal → rel_x<0、cos(方向角)≈−0.98
    # （门被错指到 −x 方向；修复后 env 系 goal 给 rel_x=0、cos=0）
    wrong = ball_mdp._goal_rel_b_impl(base, _YAW_Q0.expand(n, 4),
                                      goal_center=(0.0, 3.0))
    assert wrong[1, 0] < 0
    assert wrong[1, 3] < -0.5


def test_push_geometry_b_per_env_frame():
    """球→门方向以 env 系 goal 为锚：相同 env 系位形 → 逐 env 特征一致。

    env1（origin (10,0)）狗 origin+(0,0)、球 origin+(0,1)：btog=(0,2) →
    cos_behind=1、sin=1、cos=0。世界系常量 goal (0,3) 下 btog=(−10,2) →
    cos_behind≈0.196 的错误值。
    """
    n = 2
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    base = torch.tensor([[0.0, 0.0, 0.2], [10.0, 0.0, 0.2]])
    ball = torch.tensor([[0.0, 1.0, 0.11], [10.0, 1.0, 0.11]])
    env = _fake_env(base, _YAW_Q0.expand(n, 4), ball, env_origins=origins)
    got = ball_mdp.push_geometry_b(env)
    expect_row = torch.tensor([1.0, 1.0, 0.0])
    assert torch.allclose(got[0], expect_row, atol=1e-6)
    assert torch.allclose(got[1], expect_row, atol=1e-6)
    # bug 面钉：世界系常量 goal 下 env1 cos_behind 明显偏离 1
    wrong = ball_mdp._push_geometry_b_impl(base, ball, _YAW_Q0.expand(n, 4),
                                           goal_center=(0.0, 3.0))
    assert abs(float(wrong[1, 0]) - 1.0) > 0.5


# --------------------------------------------------------------------------- #
# 1c. 奖励/终止规则族（owner 二次拍板 2026-09-30：phase0 规则接口对齐；
#     donor 数值语义单测，公式面 = donor rewards.py/terminations.py 逐行）
# --------------------------------------------------------------------------- #

def test_robot_fallen_impl_truth_table():
    """摔判定双条件（donor terminations.py:42-59 语义）：重力支/高度支/严格边界。

    站立 proj_grav_z≈−1（直立）不判摔；倾斜超 ~60°（gz>−0.5）判摔；
    base 高 < 0.16 判摔（go2 站高 0.227）。两支均严格不等号（donor 同款）。
    """
    cases = [
        (-1.0, 0.227, False),  # 站立
        (-0.4, 0.227, True),   # 重力支：倾斜超 60°（gz > −0.5）
        (-0.5, 0.227, False),  # gz = 阈值（严格 > 不触发）
        (-1.0, 0.16, False),   # 高度支：z = 阈值（严格 < 不触发）
        (-1.0, 0.15, True),    # 高度支：低高度（go2 站高 0.227 的 0.15 同款旧阈值必触发）
        (-0.2, 0.10, True),    # 双支同触发
        (-0.9, 0.20, False),   # 蹲伏波谷（站高 70%）不误杀
    ]
    gz = torch.tensor([c[0] for c in cases])
    bz = torch.tensor([c[1] for c in cases])
    got = ball_mdp._fallen_mask_impl(gz, bz)
    assert got.dtype == torch.bool
    assert got.tolist() == [c[2] for c in cases]
    # 阈值参数化（可调面）：默认 −0.5 下 gz=−0.4 判摔；收紧到 −0.3 后不判
    assert bool(ball_mdp._fallen_mask_impl(torch.tensor([-0.4]), torch.tensor([0.227]))[0]) is True
    assert bool(ball_mdp._fallen_mask_impl(torch.tensor([-0.4]), torch.tensor([0.227]),
                                           grav_threshold=-0.3)[0]) is False


def test_robot_fallen_penalty_wrapper_mask():
    """摔倒罚 wrapper：状态直检 1.0 mask（float32；幅值由 weight<0 承载）。

    时序面：不读 termination manager（fake env 无该属性仍可算——奖励先于
    终止计算）。阈值与 robot_fallen DoneTerm 同源常量。
    """
    n = 3
    base = torch.tensor([[0.0, 0.0, 0.227], [0.0, 0.0, 0.10], [0.0, 0.0, 0.227]])
    gz = torch.tensor([-1.0, -0.3, -0.4])  # env1 高度支摔、env2 重力支摔
    env = _fake_env(base, _YAW_Q0.expand(n, 4), torch.zeros(n, 3), proj_grav_z=gz)
    rew = ball_mdp.robot_fallen_penalty(env)
    assert rew.dtype == torch.float32
    assert torch.allclose(rew, torch.tensor([0.0, 1.0, 1.0]))
    # 阈值参数化（wrapper 签名同 donor penalty/termination 面的 threshold 形参）
    rew2 = ball_mdp.robot_fallen_penalty(env, grav_threshold=-0.35)
    assert torch.allclose(rew2, torch.tensor([0.0, 1.0, 0.0]))


def test_approach_ball_velocity_impl_donor_semantics():
    """朝球速度投影（donor rewards.py:267-311 数值语义）：投影/封顶/正负支/
    距离窗/低速门逐面手算。"""
    base = torch.tensor([[0.0, 0.0, 0.11]])
    ball = torch.tensor([[1.0, 0.0, 0.11]])  # 同高 → dist=1，视向 = +x
    # 朝球 0.5 m/s：proj=0.5 → cap(±0.5) 不截 → +1.0×0.5 = 0.5（正支 coeff 1.0）
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, ball, torch.tensor([[0.5, 0.0, 0.0]]))
    assert abs(float(got[0]) - 0.5) < 1e-6
    # 远离球：proj=−0.5 → 负支 −2.0×0.5 = −1.0（负支更重，donor 同）
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, ball, torch.tensor([[-0.5, 0.0, 0.0]]))
    assert abs(float(got[0]) - (-1.0)) < 1e-6
    # 封顶：proj=2.0 > max_speed 0.5 → capped 0.5 → +0.5（高速冲撞不额外给分）
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, ball, torch.tensor([[2.0, 0.0, 0.0]]))
    assert abs(float(got[0]) - 0.5) < 1e-6
    # 距离窗外（dist=5 > max_dist 3）：零
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, torch.tensor([[5.0, 0.0, 0.11]]), torch.tensor([[0.5, 0.0, 0.0]]))
    assert float(got[0]) == 0.0
    # 低速门（speed < speed_threshold）：零（donor :306-309 语义）
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, ball, torch.tensor([[0.5, 0.0, 0.0]]), speed_threshold=1.0)
    assert float(got[0]) == 0.0
    # 侧向速度无投影分量：proj=0，不落正支（capped=0 → where 条件 >0 为假 → −0）
    got = ball_mdp._approach_ball_velocity_reward_impl(
        base, ball, torch.tensor([[0.0, 0.5, 0.0]]))
    assert float(got[0]) == 0.0


def test_ball_distance_impl_exp_kernel():
    """球距 exp 核（donor rewards.py:723-737 数值语义）：scale·exp(−d/T)。"""
    base = torch.tensor([[0.0, 0.0, 0.2]])
    ball = torch.tensor([[2.0, 0.0, 0.11]])  # 水平距 2（impl 2D 口径，donor 同）
    got = ball_mdp._ball_distance_reward_impl(base, ball)  # scale=1, T=2（donor 默认）
    assert abs(float(got[0]) - math.exp(-1.0)) < 1e-6
    # 参数化：scale 线性、temperature 换核宽
    got = ball_mdp._ball_distance_reward_impl(base, ball, scale=0.05, temperature=1.0)
    assert abs(float(got[0]) - 0.05 * math.exp(-2.0)) < 1e-6
    # 零距 = scale（核上界）
    got = ball_mdp._ball_distance_reward_impl(base, torch.tensor([[0.0, 0.0, 0.11]]))
    assert abs(float(got[0]) - 1.0) < 1e-6


def test_behind_ball_impl_exp_kernel_and_sign():
    """球后位形核（donor rewards.py:778-795 数值语义）：behind_point = 球位
    沿己方球门方向平移 behind_offset；=1 当狗恰在球正后（进攻 +y → 后 = −y）。"""
    ball = torch.tensor([[0.0, 1.0, 0.11]])
    behind_pt = torch.tensor([[0.0, 1.0 - 0.45, 0.2]])  # 球后 0.45m（donor offset 默认）
    got = ball_mdp._behind_ball_reward_impl(behind_pt, ball)  # scale=0.35, T=1.5（donor 默认）
    assert abs(float(got[0]) - 0.35) < 1e-6  # dist=0 → exp(0)=1
    # 狗在球位：dist=0.45 → 0.35·exp(−0.3)（手算 donor 核）
    got = ball_mdp._behind_ball_reward_impl(torch.tensor([[0.0, 1.0, 0.2]]), ball)
    assert abs(float(got[0]) - 0.35 * math.exp(-0.45 / 1.5)) < 1e-6
    # attack_sign 翻转方向（进攻 −y → 后 = +y）
    got = ball_mdp._behind_ball_reward_impl(
        torch.tensor([[0.0, 1.0 + 0.45, 0.2]]), ball, attack_sign=-1.0)
    assert abs(float(got[0]) - 0.35) < 1e-6


def test_ball_progress_dense_impl_donor_semantics():
    """Δy 进攻方向单边 clamp 核（donor rewards.py:677-702 数值语义）：
    clamp(sign·Δy)·近球门控 + NaN 懒重播种。"""
    base = torch.tensor([[0.0, 0.0, 0.2]])
    ball = torch.tensor([[0.0, 1.0, 0.11]])  # 距球 1.0 < max_dist 1.0？严格 < → 需更近
    ball[0, 1] = 0.5  # 距球 0.5
    # 前进 Δy=+0.5 → +scale×0.5（scale 默认 5 → 2.5）
    rew, cache = ball_mdp._ball_progress_dense_impl(torch.tensor([1.0]), torch.tensor([1.5]),
                                                    base, ball)
    assert abs(float(rew[0]) - 2.5) < 1e-6 and float(cache[0]) == 1.5
    # 后退 Δy=−0.5 → 单边 clamp 0（donor 只奖进攻方向，不罚后退）
    rew, _ = ball_mdp._ball_progress_dense_impl(torch.tensor([1.0]), torch.tensor([0.5]),
                                                base, ball)
    assert float(rew[0]) == 0.0
    # NaN prev（懒重播种）→ 差分 0
    rew, cache = ball_mdp._ball_progress_dense_impl(torch.tensor([float("nan")]),
                                                    torch.tensor([1.5]), base, ball)
    assert float(rew[0]) == 0.0 and float(cache[0]) == 1.5
    # 近球门控：机器人远球（dist ≥ max_dist）→ 0（donor "只在贴身控球时给分"）
    far = torch.tensor([[0.0, 5.0, 0.2]])
    rew, _ = ball_mdp._ball_progress_dense_impl(torch.tensor([1.0]), torch.tensor([1.5]),
                                                far, ball)
    assert float(rew[0]) == 0.0
    # sign 翻转（进攻 −y → −Δy 才得分）
    rew, _ = ball_mdp._ball_progress_dense_impl(torch.tensor([1.5]), torch.tensor([1.0]),
                                                base, ball, sign=-1.0)
    assert abs(float(rew[0]) - 2.5) < 1e-6


def test_ball_progress_dense_wrapper_reset_reseed():
    """[X-HK4] 每集首步重播种：残留缓存（上集终点 y_prev=5.0）→ reset 后球
    y=2.8（> y_prev，旧"单边 clamp 免疫"论证失效的退化洞）→ 首步 progress
    必须=0（prev 取 cur）+ 缓存重播种；次步起恢复正常差分。"""
    # 狗 (0,2.7)、球 spawn (0,2.8)——H 臂布局（近球门控内 dist=0.1）
    base = torch.tensor([[0.0, 2.7, 0.2], [0.0, 2.7, 0.2]])
    quat = _YAW_QP90.expand(2, 4).contiguous()
    ball = torch.tensor([[0.0, 2.8, 0.11], [0.0, 2.8, 0.11]])
    env = _fake_env(base, quat, ball)

    # 模拟上一集终点残留缓存 + reset 后首步：y_prev=2.0（上集球在线前）→
    # reset 2.8 → 裸差分 Δy=+0.8×5=4.0 正向假信号（旧实现白给）→ 首步必须记 0
    setattr(env, "_kick_prev_ball_y", torch.full((2,), 2.0))
    env.episode_length_buf = torch.tensor([1, 1])
    rew0 = ball_mdp.ball_progress_dense(env)
    assert torch.allclose(rew0, torch.zeros(2))  # 首步差分记 0（退化洞封死）
    assert torch.allclose(getattr(env, "_kick_prev_ball_y"), torch.full((2,), 2.8))
    assert rew0.dtype == torch.float32

    # 次步（episode_length_buf=2）：球进 +0.2 → 正常差分 0.2×5=1.0
    ball[:, 1] = 3.0
    env.episode_length_buf = torch.tensor([2, 2])
    rew1 = ball_mdp.ball_progress_dense(env)
    assert torch.allclose(rew1, torch.full((2,), 1.0))

    # 混合批：env0 首步重播种 / env1 次步正常差分——逐 env 独立
    env.episode_length_buf = torch.tensor([1, 5])
    rew2 = ball_mdp.ball_progress_dense(env)
    assert rew2.tolist() == [0.0, 0.0]  # env0 重播种记 0；env1 Δy=0 → 0
    # 离线 fallback：无 episode_length_buf 的 fake env（旧用例形态）不炸
    env2 = _fake_env(base, quat, ball)
    assert torch.allclose(ball_mdp.ball_progress_dense(env2), torch.zeros(2))


def test_ball_progress_dense_wrapper_cache_lifecycle():
    """wrapper 缓存生命周期：懒初始化 0 → 正常差分 → reset 后跳变免疫（clamp）。

    替代旧 ball_progress_shaping 缓存面：新缓存挂 _kick_prev_ball_y，无
    reset 钩子（固定 spawn y=0 → reset 后首步 Δy ≤ 0 被单边 clamp 归零）。
    """
    base = torch.zeros(2, 3)
    quat = _YAW_Q0.expand(2, 4).contiguous()
    ball = torch.tensor([[0.5, 0.0, 0.11], [0.5, 0.0, 0.11]])  # 距狗 0.5（门控内）
    env = _fake_env(base, quat, ball)

    # 首步懒初始化：差分 0，缓存 = cur（0.0）
    rew0 = ball_mdp.ball_progress_dense(env)
    assert torch.allclose(rew0, torch.zeros(2))
    assert torch.allclose(getattr(env, "_kick_prev_ball_y"), torch.zeros(2))

    # 球沿 +y 进 0.5m：差分 = 0.5 × scale 5 = 2.5；缓存覆写
    ball[:, 1] = 0.5
    rew1 = ball_mdp.ball_progress_dense(env)
    assert torch.allclose(rew1, torch.full((2,), 2.5))
    assert torch.allclose(getattr(env, "_kick_prev_ball_y"), torch.full((2,), 0.5))

    # reset（无钩子）：球回 spawn y=0 → Δy = −0.5 → clamp 0（跳变免疫）
    ball[:, 1] = 0.0
    rew2 = ball_mdp.ball_progress_dense(env)
    assert torch.allclose(rew2, torch.zeros(2))
    assert torch.allclose(getattr(env, "_kick_prev_ball_y"), torch.zeros(2))
    # 输出面 float32（RewardManager 加权面）
    assert rew2.dtype == torch.float32


# --------------------------------------------------------------------------- #
# 1b. X-HK2 kick_event 奖励（自建过程引导项；检测面单源 kick_metrics）
# --------------------------------------------------------------------------- #

_CD = 25  # 0.5s @50Hz（ball_env_cfg D 臂配置步数）


def test_kick_event_impl_cooldown_and_reset():
    """事件核纯逻辑：首踢计费 → 冷却内不计 → 冷却边界恢复 → reset 跳变
    清缓存（无 reset 钩子——episode 步计数回退判据）→ NaN seed 懒播种。"""
    nan = torch.tensor([float("nan")])
    # 首调用（NaN seed）：step 3 踢 → 计 1.0，缓存 (prev=3, last=3)
    r, p, l = ball_mdp._kick_event_reward_impl(
        nan, nan, torch.tensor([3.0]), torch.tensor([True]), _CD)
    assert float(r[0]) == 1.0 and float(p[0]) == 3.0 and float(l[0]) == 3.0
    assert r.dtype == torch.float32
    # 冷却内（step 27，diff 24 < 25）再踢 → 0，last 保持 3
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([27.0]), torch.tensor([True]), _CD)
    assert float(r[0]) == 0.0 and float(l[0]) == 3.0
    # 冷却边界（step 28，diff 25 ≥ 25）→ 再踢计 1
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([28.0]), torch.tensor([True]), _CD)
    assert float(r[0]) == 1.0 and float(l[0]) == 28.0
    # 无踢步 → 0（缓存不动）
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([29.0]), torch.tensor([False]), _CD)
    assert float(r[0]) == 0.0 and float(l[0]) == 28.0
    # reset 跳变（episode 步计数回退 2 ≤ prev 29）→ last 清 NaN → 踢立计
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([2.0]), torch.tensor([True]), _CD)
    assert float(r[0]) == 1.0 and float(l[0]) == 2.0
    # reset 跳变且本步不踢 → last 清 NaN（上一 episode 残留冷却零泄漏）
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([2.0]), torch.tensor([False]), _CD)
    assert float(r[0]) == 0.0 and not bool(torch.isfinite(l[0]))
    # 批内独立：env0 冷却中 / env1 NaN seed 可计
    p = torch.tensor([5.0, float("nan")])
    l = torch.tensor([5.0, float("nan")])
    r, p, l = ball_mdp._kick_event_reward_impl(
        p, l, torch.tensor([6.0, 6.0]), torch.tensor([True, True]), _CD)
    assert r.tolist() == [0.0, 1.0]
    assert l.tolist() == [5.0, 6.0]


def _fake_env_kick_event(forces: torch.Tensor, episode_step: torch.Tensor,
                         step_dt: float = 0.02):
    """kick_event wrapper 取数面 stub：ball_contact_forces filter 力矩阵 +
    episode_length_buf + step_dt（其余面零消费）。"""
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(force_matrix_w=forces))
    scene = _FakeScene({"ball_contact_forces": sensor})
    scene.env_origins = torch.zeros(forces.shape[0], 3)
    return types.SimpleNamespace(scene=scene,
                                 episode_length_buf=episode_step.clone(),
                                 step_dt=step_dt)


def test_kick_event_wrapper_lifecycle_and_threshold():
    """wrapper 生命周期（force_matrix_w 取数 → 每环境最大合范数 >0.1N 判踢
    ——kick_metrics._robot_contact_force 单源）：首踢/冷却/步数换算/缓存
    挂载 _kick_event_last_step。"""
    forces = torch.zeros(2, 1, 5, 3)
    env = _fake_env_kick_event(forces, torch.tensor([0, 0]))
    # env0 足端力 0.2N（>0.1 判踢）；env1 0.05N（不判踢）
    forces[0, 0, 2, 0] = 0.2
    forces[1, 0, 0, 1] = 0.05
    r0 = ball_mdp.kick_event_reward(env)
    assert r0.tolist() == [1.0, 0.0] and r0.dtype == torch.float32
    cache = getattr(env, "_kick_event_last_step")
    assert float(cache[1][0]) == 0.0 and not bool(torch.isfinite(cache[1][1]))
    # 下一冷却步（step 1，diff 1 < 25）：力持续 → 不重复计费
    env.episode_length_buf = torch.tensor([1, 1])
    r1 = ball_mdp.kick_event_reward(env)
    assert r1.tolist() == [0.0, 0.0]
    # 冷却后（step 25）：env0 再计
    env.episode_length_buf = torch.tensor([25, 25])
    r2 = ball_mdp.kick_event_reward(env)
    assert r2.tolist() == [1.0, 0.0]
    # 严格不等号：力恰 = 阈值 0.1N 不判踢
    env.episode_length_buf = torch.tensor([50, 50])
    forces[0, :, :, :] = 0.0
    forces[0, 0, 1, 2] = 0.1
    r3 = ball_mdp.kick_event_reward(env)
    assert r3.tolist() == [0.0, 0.0]


def test_kick_event_threshold_single_source_mirror():
    """单一实现源：wrapper 检测函数/阈值 = kick_metrics 单源 + 签名默认
    镜像值 0.1（kick_metrics 模块尾交叉断言的测试面双保险）。"""
    import inspect
    from hierarchical_kick.core.kick_metrics import (
        KICK_FORCE_THRESHOLD_N, KickMetricsTracker)
    sig = inspect.signature(ball_mdp.kick_event_reward)
    assert float(sig.parameters["force_threshold"].default) == \
        float(KICK_FORCE_THRESHOLD_N) == 0.1
    assert float(sig.parameters["cooldown_s"].default) == 0.5
    # 检测函数单源：wrapper 走 KickMetricsTracker._robot_contact_force
    src = open(_BALL_MDP_PATH, encoding="utf-8").read()
    assert "KickMetricsTracker._robot_contact_force(env)" in src
    # mirror 守卫在案（kick_metrics 模块级——本测试 import 即已执行通过）
    km_src = inspect.getsource(KickMetricsTracker)
    assert "force_matrix_w" in km_src


def test_ball_out_of_bounds_penalty_mask_matches_ball_out():
    """出界步罚 mask ≡ ball_out 判据（复用同一 impl；float32 输出）。"""
    ball = torch.tensor([[0.0, 9.0, 0.11], [0.0, 9.001, 0.11], [0.0, 1.0, 0.11]])
    env = _fake_env(torch.zeros(3, 3), _YAW_Q0.expand(3, 4), ball)
    rew = ball_mdp.ball_out_of_bounds_penalty(env)
    assert rew.dtype == torch.float32
    assert torch.equal(rew, ball_mdp.ball_out(env).to(torch.float32))
    assert rew.tolist() == [0.0, 1.0, 0.0]


def test_robot_out_of_field_impl_and_wrapper_per_env_frame():
    """工作区框判据 + 逐 env origin 帧（世界系坐标必须先减 env_origins）。"""
    # impl：框 x∈[−1,5]/y∈[−4,4]
    pos = torch.tensor([[0.0, 0.0], [5.1, 0.0], [-1.1, 0.0], [0.0, 4.1], [4.9, -3.9]])
    got = ball_mdp._robot_out_of_field_impl(pos, -1.0, 5.0, -4.0, 4.0)
    assert got.tolist() == [False, True, True, True, False]
    # 边界严格不等号（框上不触发）
    got = ball_mdp._robot_out_of_field_impl(torch.tensor([[5.0, 4.0], [-1.0, -4.0]]),
                                            -1.0, 5.0, -4.0, 4.0)
    assert got.tolist() == [False, False]
    # wrapper：env1（origin (10,0)）env 系 (5.1,0) → True；世界系同 x 在 env0 (0,0) → False
    n = 3
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    base = torch.tensor([[5.1, 0.0, 0.2], [15.1, 0.0, 0.2], [4.9, 0.0, 0.2]])
    env = _fake_env(base, _YAW_Q0.expand(n, 4), torch.zeros(n, 3), env_origins=origins)
    assert ball_mdp.robot_out_of_field(env).tolist() == [True, True, False]


def test_wall_stuck_impl_interface():
    """wall_stuck 接口（donor terminations.py:78-93 dist+speed 双条件）：
    贴边 + 低速 → True；高速/离边 → False。**cfg 不挂载**（见 cfg 源码断言）。"""
    # 框 x∈[−1,5]：x=4.95 距右边 0.05 < 0.1，body 速率 0.1 < 0.3 → True
    pos = torch.tensor([[4.95, 0.0], [4.95, 0.0], [0.0, 0.0]])
    vel_b = torch.tensor([[0.1, 0.0], [0.5, 0.0], [0.1, 0.0]])
    got = ball_mdp._wall_stuck_impl(pos, vel_b, -1.0, 5.0, -4.0, 4.0)
    assert got.tolist() == [True, False, False]


def test_shaping_cache_face_removed_guards():
    """防呆：旧 ball_progress_shaping prev-cur 缓存面零残留（函数/钩子/属性名/
    cfg 消费点）。"""
    for gone in ("ball_progress_shaping", "reset_ball_progress_shaping_cache",
                 "_ball_progress_shaping_impl"):
        assert not hasattr(ball_mdp, gone), gone
    with open(_BALL_MDP_PATH, encoding="utf-8") as f:
        mdp_src = f.read()
    assert "_kick_prev_ball_goal_dist" not in mdp_src  # 旧缓存属性名
    assert "def ball_progress_shaping(" not in mdp_src  # 函数面（头注 provenance
    assert "def reset_ball_progress_shaping_cache(" not in mdp_src  # 提及不算残留）
    cfg_src = _kick_cfg_src()
    # cfg 面：无 term 接线、无 reset 事件消费点（头注叙事性提及不算残留）
    assert "func=mdp.ball_progress_shaping" not in cfg_src
    assert "reset_ball_progress_shaping_cache" not in cfg_src


# --------------------------------------------------------------------------- #
# 2. 常量与维度契约
# --------------------------------------------------------------------------- #

def test_kick_obs_dim_contract_arithmetic():
    """KICK_OBS_DIM == 55 == 42(proprio) + 3 + 3 + 4 + 3（算术断言）。"""
    proprio = 12 + 12 + 3 + 3 + 12  # joint_pos/joint_vel/base_ang_vel/proj_grav/last_action
    assert ball_mdp.KICK_OBS_DIM == 55
    assert ball_mdp.KICK_OBS_DIM == proprio + 3 + 3 + 4 + 3


def test_ball_geometry_constants_match_donor():
    """球参数/球门/摔判定/工作区框常量 == donor 逐值或注释论证值。"""
    assert ball_mdp.GOAL_Y == 3.0
    assert ball_mdp.GOAL_HALF_WIDTH == 0.75  # mouth 宽 1.5m
    assert ball_mdp.GOAL_CENTER == (0.0, 3.0)
    assert ball_mdp.BALL_OUT_RADIUS == 6.0
    assert ball_mdp.BALL_RADIUS == 0.11
    assert ball_mdp.BALL_MASS == 0.43
    # 摔判定双条件阈值（donor terminations.py:42 签名默认）
    assert ball_mdp.FALLEN_GRAV_THRESHOLD == -0.5
    assert ball_mdp.FALLEN_BODY_Z_THRESHOLD == 0.16
    # 机器人工作区框（任务域 3m + 追球缓冲；注释论证在案）
    assert (ball_mdp.ROBOT_FIELD_X_MIN, ball_mdp.ROBOT_FIELD_X_MAX) == (-1.0, 5.0)
    assert (ball_mdp.ROBOT_FIELD_Y_MIN, ball_mdp.ROBOT_FIELD_Y_MAX) == (-4.0, 4.0)


def test_ball_mdp_exports_and_offline_rotation_fallback():
    """导出面（名称契约）+ 离线旋转降级面 == math_utils 等价实现（防桩污染）。"""
    for name in ("ball_rel_b", "goal_rel_b", "push_geometry_b", "goal_scored",
                 "ball_out", "goal_scored_reward", "robot_fallen_penalty",
                 "approach_ball_velocity_reward", "ball_distance_reward",
                 "behind_ball_reward", "ball_progress_dense",
                 "kick_event_reward",
                 "ball_out_of_bounds_penalty",
                 "robot_fallen", "robot_out_of_field", "wall_stuck"):
        assert name in ball_mdp.__all__, name
    assert "ball_progress_shaping" not in ball_mdp.__all__
    assert "reset_ball_progress_shaping_cache" not in ball_mdp.__all__
    # 守卫式 import 面在案（sim 路径 = isaaclab 原函数 + 桩防呆 + 离线降级）
    src = open(_BALL_MDP_PATH, encoding="utf-8").read()
    assert "from isaaclab.utils.math import quat_apply_inverse" in src
    assert "__module__" in src  # 桩防呆（绑定面校验）
    assert "quat_rotate_inverse as _quat_apply_inverse" in src
    # 离线（含假桩注入的全套件环境）绑定面恒为仓库离线数学单源（vendored 等价）
    from hierarchical_kick.sim.math_utils import quat_rotate_inverse
    bound = getattr(ball_mdp._quat_apply_inverse, "__module__", "")
    if bound.endswith("math_utils"):
        assert ball_mdp._quat_apply_inverse is quat_rotate_inverse


# --------------------------------------------------------------------------- #
# 3. ball_env_cfg.py 源码断言（离线不可 import → 文本/AST 面）
# --------------------------------------------------------------------------- #

def _kick_cfg_src() -> str:
    with open(_BALL_CFG_PATH, encoding="utf-8") as f:
        return f.read()


def test_ball_env_cfg_syntax_and_provenance():
    """AST 可解析 + donor provenance（scene.py:215-247 / 规则接口族）在案 +
    零 donor import。"""
    src = _kick_cfg_src()
    ast.parse(src)
    assert "scene.py:215-247" in src  # 球物理 provenance
    assert "reset.py:257-280" in src  # 球 reset provenance
    assert "import soccer" not in src and "from soccer" not in src  # 红线
    # mdp 源码面同检（provenance + 红线）：obs 族 + 规则接口族逐函数标注
    with open(_BALL_MDP_PATH, encoding="utf-8") as f:
        mdp_src = f.read()
    ast.parse(mdp_src)
    for donor_ref in ("observations.py:73-79", "observations.py:81-88",
                      "observations.py:205-235",
                      "rewards.py:267-311", "rewards.py:723-737",
                      "rewards.py:778-795", "rewards.py:677-702",
                      "rewards.py:461-479", "rewards.py:37-102",
                      "terminations.py:42-59", "terminations.py:27-40",
                      "terminations.py:78-93"):
        assert donor_ref in mdp_src, donor_ref
    assert "import soccer" not in mdp_src and "from soccer" not in mdp_src


def test_ball_env_cfg_ball_scene_values():
    """球场景逐值断言（donor scene.py:215-247 逐值照抄 + A100 实跑修复面：
    contact filter 显式枚举 GO2_FOOT_BODIES、球 spawn 无 headless 死旋
    visual_material）。"""
    src = _kick_cfg_src()
    assert 'prim_path="{ENV_REGEX_NS}/Ball"' in src
    assert "radius=0.11" in src
    assert "static_friction=1.0" in src
    assert "dynamic_friction=0.95" in src
    assert "restitution=0.25" in src
    assert "linear_damping=2.5" in src
    assert "angular_damping=2.5" in src
    assert "mass=0.43" in src
    assert "collision_enabled=True" in src
    assert "activate_contact_sensors=True" in src
    # [X-HK4 H 臂] 冷启动饱和 spawn 常量面（owner 指令）：球门线前 0.2m、
    # 狗门线前 0.3m + yaw=π/2（init_state 单源；xyzw 通道序）
    assert "pos=(BALL_SPAWN_XY[0], BALL_SPAWN_XY[1], 0.11)" in src
    assert "DOG_SPAWN_XY = (0.0, mdp.GOAL_Y - 0.3)" in src
    assert "BALL_SPAWN_XY = (0.0, mdp.GOAL_Y - 0.2)" in src
    assert "math.sin(0.25 * math.pi), math.cos(0.25 * math.pi)" in src  # yaw=π/2 xyzw
    assert "_dog_init.rot = DOG_SPAWN_YAW_QUAT_XYZW" in src
    # headless 材质死旋防护（A100 实测；flat_env_cfg.py 头 U6 同型）：
    # 球 spawn 不挂 visual_material（AST 面逐 Call 查 keyword——头注提及
    # 不算挂载），防护注释在位（录像阶段带 --enable_cameras 再启用）
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            assert all(kw.arg != "visual_material" for kw in node.keywords), \
                "球 spawn 不得挂 visual_material（headless 死旋防护）"
    assert "headless 不挂" in src and "死旋" in src
    # 球侧接触传感器 filter（A100 实跑修复）：显式枚举 = base + 4 足端
    # （单源 GO2_FOOT_BODIES，逐 env 恰 5 prim）；反向断言 filter 段无
    # .* 通配 —— env_*/Robot/.*_foot 在 10 env 下跨 env 匹配 40 prim，
    # tensor API expected/found 警告 + 力矩阵形状漂移，禁止回退通配
    assert "from hierarchical_kick.robots.go2 import GO2_FOOT_BODIES" in src
    assert "filter_prim_paths_expr=[" in src
    filter_seg = src.split("filter_prim_paths_expr=[", 1)[1].split("],", 1)[0]
    assert '"{ENV_REGEX_NS}/Robot/base"' in filter_seg
    assert "for b in GO2_FOOT_BODIES" in filter_seg
    assert ".*" not in filter_seg  # 反向断言：通配死防（40 prim 形状漂移）


def test_ball_env_cfg_task_only_rewards():
    """task-only 置 None 面 + 权重单源 KickRewardWeightsCfg + 8 项规则接口挂载。"""
    src = _kick_cfg_src()
    for term in (
        "self.rewards.track_lin_vel_xy_exp = None",
        "self.rewards.track_ang_vel_z_exp = None",
        "self.rewards.feet_air_time = None",
        "self.rewards.flat_orientation_l2 = None",
        "self.rewards.lin_vel_z_l2 = None",
        "self.rewards.ang_vel_xy_l2 = None",
        "self.rewards.dof_torques_l2 = None",
        "self.rewards.dof_acc_l2 = None",
        "self.rewards.action_rate_l2 = None",
        "self.rewards.dof_pos_limits = None",
    ):
        assert term in src, f"task-only 奖励缺 None 项: {term}"
    # 权重单源数据类（phase0 规则接口对齐 + go2 适配；字段默认值 = 我方默认权重）
    assert "class KickRewardWeightsCfg" in src
    for field in ("goal_scored: float = 10.0",
                  "robot_fallen: float = -5.0",
                  "survival: float = 0.25",
                  "approach_ball_velocity: float = 0.15",
                  "ball_distance: float = 0.05",
                  "ball_distance_temperature: float = 2.0",
                  "behind_ball: float = 0.05",
                  "ball_progress: float = 0.10",
                  "ball_out_of_bounds: float = -2.0"):
        assert field in src, field
    assert "phase0 raw" in src and "量纲理由" in src  # 对照表头注释在案
    # 8 项规则接口挂载（权重一律经单源 w.*；2026-09-30 二次拍板）
    assert "self.kick_reward_weights = KickRewardWeightsCfg()" in src
    for mount in (
        "self.rewards.goal_scored = RewTerm(func=mdp.goal_scored_reward, weight=w.goal_scored)",
        "self.rewards.robot_fallen = RewTerm(",
        "func=mdp.robot_fallen_penalty, weight=w.robot_fallen,",
        "self.rewards.survival = RewTerm(func=mdp.alive_reward, weight=w.survival)",
        "self.rewards.approach_ball = RewTerm(func=mdp.approach_ball_velocity_reward, weight=w.approach_ball_velocity)",
        "self.rewards.ball_distance = RewTerm(func=mdp.ball_distance_reward, weight=w.ball_distance, params={\"temperature\": w.ball_distance_temperature})",
        "self.rewards.behind_ball = RewTerm(func=mdp.behind_ball_reward, weight=w.behind_ball)",
        "self.rewards.ball_progress = RewTerm(func=mdp.ball_progress_dense, weight=w.ball_progress)",
        "self.rewards.ball_out_of_bounds = RewTerm(func=mdp.ball_out_of_bounds_penalty, weight=w.ball_out_of_bounds)",
    ):
        assert mount in src, mount
    # 调参零摩擦：sync 方法 + 挂载后同步在位
    assert "def sync_kick_reward_weights(self)" in src
    assert "self.sync_kick_reward_weights()" in src
    # [X-HK3 F 臂] ball_distance 温度入单源：挂载 params + sync 同步面在案
    assert 'self.rewards.ball_distance.params["temperature"] = w.ball_distance_temperature' in src
    # 摔倒罚阈值与终止同源常量（单源 FALLEN_*；无裸数值第二处）
    assert src.count("mdp.FALLEN_GRAV_THRESHOLD") == 2  # 奖励 + 终止各一
    assert src.count("mdp.FALLEN_BODY_Z_THRESHOLD") == 2


def test_ball_env_cfg_kick_event_zero_default_mount():
    """[X-HK2] kick_event 第 9 项：权重缺省 0.0（零行为影响红线）+ 挂载 +
    sync 接入（D 臂经 --kick_rw_overrides 启用 0.5）。"""
    src = _kick_cfg_src()
    assert "kick_event: float = 0.0" in src
    assert "self.rewards.kick_event = RewTerm(func=mdp.kick_event_reward, weight=w.kick_event)" in src
    assert "self.rewards.kick_event.weight = w.kick_event" in src
    assert "零行为影响" in src  # 红线纪律注在案


def test_ball_env_cfg_terminations_and_episode():
    """终止面（phase0 active_terminations 对齐）：robot_fallen 替换
    root_height、robot_out_of_field 挂载、wall_stuck 接口在位不挂载 +
    episode 10s。"""
    src = _kick_cfg_src()
    # root_height 显式置 None（被 robot_fallen 高度支 0.16 严格覆盖 0.15）
    assert "self.terminations.root_height = None" in src
    # robot_fallen 双条件挂载（donor terminations.py:42 签名；阈值参数化）
    assert "self.terminations.robot_fallen = DoneTerm(" in src
    assert "func=mdp.robot_fallen," in src
    assert '"grav_threshold": mdp.FALLEN_GRAV_THRESHOLD' in src
    assert '"body_z_threshold": mdp.FALLEN_BODY_Z_THRESHOLD' in src
    # robot_out_of_field 工作区框挂载（默认启用；逐 env origin 相对）
    assert "self.terminations.robot_out_of_field = DoneTerm(" in src
    assert "func=mdp.robot_out_of_field," in src
    for b in ("ROBOT_FIELD_X_MIN", "ROBOT_FIELD_X_MAX",
              "ROBOT_FIELD_Y_MIN", "ROBOT_FIELD_Y_MAX"):
        assert f"mdp.{b}" in src, b
    # wall_stuck 接口在位**不挂载**（无墙备用；mdp 源有函数、cfg 无挂载行）
    assert "self.terminations.wall_stuck" not in src
    assert "wall_stuck" in src and "不挂载" in src  # 备用注释在案
    # 既有两新项 + 继承项不 None 化（base_contact/time_out/root_height 原挂载面）
    assert "self.terminations.goal_scored = DoneTerm(func=mdp.goal_scored)" in src
    assert "self.terminations.ball_out = DoneTerm(func=mdp.ball_out)" in src
    for term in ("base_contact", "time_out"):
        assert f"self.terminations.{term} = None" not in src
    with open(_BALL_MDP_PATH, encoding="utf-8") as f:
        mdp_src = f.read()
    assert "def wall_stuck(" in mdp_src  # 接口面在 mdp（挂载与否属 cfg 决策）
    assert "self.episode_length_s = 10.0" in src


def test_ball_env_cfg_kick_policy_and_commands():
    """55 维 obs 组替换 + 42-prefix 契约参数原样 + 无命令面。"""
    src = _kick_cfg_src()
    assert "self.observations.policy = KickPolicyCfg()" in src
    assert "self.commands.base_velocity = None" in src
    # 42-prefix 契约：五项 term + contracts 常量消费（= UpperPolicyCfg 逐参）
    for func in ("joint_pos_rel", "joint_vel_rel", "base_ang_vel", "projected_gravity",
                 "last_action", "base_lin_vel", "ball_rel_b", "goal_rel_b", "push_geometry_b"):
        assert f"func=mdp.{func}" in src, func
    for sym in ("NOISE_JOINT_POS", "NOISE_JOINT_VEL", "SCALE_JOINT_VEL",
                "NOISE_BASE_ANG_VEL", "SCALE_BASE_ANG_VEL",
                "NOISE_PROJ_GRAVITY", "NOISE_BASE_LIN_VEL"):
        assert sym in src, f"42-prefix 契约符号未消费: {sym}"
    assert "KICK_OBS_DIM" in src  # 55 维锚引用
    # 无 velocity_commands 项（命令面移除后不得悬空引用）
    assert "velocity_commands = ObsTerm" not in src
    assert "func=mdp.generated_commands" not in src


def test_ball_env_cfg_kick_randomization_disabled_by_default():
    """随机化接口存在但默认完全关闭（owner 拍板 #3）+ 固定 spawn。"""
    src = _kick_cfg_src()
    assert "self.kick_randomization = KickRandomizationCfg()" in src
    assert "enabled: bool = False" in src
    assert "ball_xy_range" in src and "dog_yaw_range" in src
    # 狗固定 spawn 原点 yaw0（pose_range 钳零；velocity_range 继承现款零速度）
    assert '"x": (0.0, 0.0)' in src and '"yaw": (0.0, 0.0)' in src


def test_ball_env_cfg_oracle_face_restores_commands_and_obs():
    """Oracle 面：恢复 UniformVelocity 命令（heading off）+ UpperPolicyCfg 48 维。"""
    src = _kick_cfg_src()
    assert "class Go2KickFlatEnvCfgOracle(Go2KickFlatEnvCfg)" in src
    assert "禁止用于三臂训练" in src  # 类注释红字
    assert "self.commands.base_velocity = mdp.UniformVelocityCommandCfg(" in src
    assert "heading_command=False" in src
    assert "self.observations.policy = UpperPolicyCfg()" in src


def test_ball_env_cfg_ball_reset_event_wires_ball_write():
    """球 reset 事件：write pose/velocity → reset，链在位（旧 shaping 缓存
    清除钩子已随 ball_progress_shaping 移除——防呆见
    test_shaping_cache_face_removed_guards）。"""
    src = _kick_cfg_src()
    assert "self.events.reset_ball = EventTerm(" in src
    assert 'mode="reset"' in src
    assert "write_root_pose_to_sim" in src
    assert "write_root_velocity_to_sim" in src
    assert "ball.reset(" in src
    assert "env_origins" in src  # env origin 偏移（donor reset.py:275 语义）
    assert "reset_ball_progress_shaping_cache" not in src  # 旧钩子零残留


# --------------------------------------------------------------------------- #
# 4. register 双任务 id + entry point
# --------------------------------------------------------------------------- #

def test_register_cmd_task_strings():
    """cmd 任务 id 字符串 + entry point 常量 == 名称契约冻结值（cmd-only
    拆分：原 KICK 原面/oracle/Seg task id 随臂移除，仅存 cmd 面）。"""
    import hierarchical_kick.tasks as reg
    assert reg.KICK_CMD_TASK_ID == "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"
    assert reg.KICK_CMD_TASK_ENTRY_POINT == "isaaclab.envs:ManagerBasedRLEnv"
    assert reg.KICK_CMD_ENV_CFG_ENTRY_POINT == (
        "hierarchical_kick.sim.ball_env_cfg:Go2KickFlatEnvCfgCommand")
    assert reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT == (
        "hierarchical_kick.agents:UpperKickCmdPPORunnerCfg")
    for name in ("KICK_CMD_TASK_ID", "KICK_CMD_TASK_ENTRY_POINT",
                 "KICK_CMD_ENV_CFG_ENTRY_POINT",
                 "KICK_CMD_RSL_RL_CFG_ENTRY_POINT"):
        assert name in reg.__all__, name


def test_register_cmd_entry_points_parseable_and_wired():
    """entry point ``module:Attr`` 形状 + gym.register kwargs 接线在位（源码面）。"""
    import hierarchical_kick.tasks as reg
    for point in (reg.KICK_CMD_ENV_CFG_ENTRY_POINT,
                  reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT):
        module_path, attr = point.split(":", 1)
        assert module_path and attr and "." in module_path
        assert not module_path.startswith("isaaclab_tasks")  # 零树改动
    src = open(_REGISTER_PATH, encoding="utf-8").read()
    ast.parse(src)
    assert "id=KICK_CMD_TASK_ID" in src
    assert '"env_cfg_entry_point": KICK_CMD_ENV_CFG_ENTRY_POINT' in src
    assert '"rsl_rl_cfg_entry_point": KICK_CMD_RSL_RL_CFG_ENTRY_POINT' in src
    # 懒加载：cfg 类名只以字符串出现，无 eager import cfg 模块
    assert "import hierarchical_kick.sim.ball_env_cfg" not in src
    assert "import hierarchical_kick.agents" not in src


# --------------------------------------------------------------------------- #
# 5. mdp/__init__ 追加 re-export（只追加，vendor 行为零改动）
# --------------------------------------------------------------------------- #

def test_mdp_init_appends_ball_mdp_reexport_only():
    """ball_mdp re-export 行在位 + 既有 vendor 结构零改动（只追加不删改）。"""
    src = open(_MDP_INIT_PATH, encoding="utf-8").read()
    assert "from .ball_mdp import *" in src
    assert "ball_mdp" in src.split('"""')[0]  # 头注 diff 说明（vendor 头惯例）在案
    # 既有 re-export 行原样保留
    for line in ("from isaaclab.envs.mdp import *",
                 "from .curriculums import *",
                 "from .rewards import *",
                 "from .terminations import *"):
        assert line in src, line


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
