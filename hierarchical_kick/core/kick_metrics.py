# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.core.kick_metrics
#
# U-Kick kick 模式指标（audit 列面）：KickMetricsTracker 每 rollout 步从 env 读
# 球 root pos/lin_vel、robot root pos、``env.scene.ball_contact_forces``
# 接触力，按 env 累积 per-episode 量，episode 结束帧（done；进球判据读
# ``env.termination_manager``，退化路径为 ball_mdp 解析判线复算）聚合：
#   - ball_disp      球位移 = 末态 − 初态（水平 xy 范数；episode 首帧锚定）
#   - max_ball_speed 球线速度批内逐步最大（3 维全量范数）
#   - kick_count     踢击事件计数 = 球侧接触力范数 > 0.1N 的步数
#                    （上游 soccer_env.py:334-341 阈值 0.1 同款思想）
#   - behind_frac    cos_behind > 0 帧占比（复用 ball_mdp 纯函数
#                    _push_geometry_b_impl 第 0 列——单一实现源）
#   - goal           episode 结束帧 goal_scored 判据真值
# audit 行窗口均值列（滚动窗口缺省 100 episode）：
#   goal_rate_win / ball_disp_ep_mean / max_ball_speed_ep_mean /
#   kick_count_ep_mean / behind_ball_frac_ep_mean
#
# 零 isaaclab：env 取数全 duck-typing（scene 字典 + data 属性 + 可选
# termination_manager），离线 fake env 直测纯逻辑。ball_mdp 复用面经守卫式
# 双路加载（sim 正常 import；离线父包 __init__ 拉 isaaclab → 按路径加载，
# test_upper_kick_env 同款降级）。
# ---------------------------------------------------------------------------

"""U-Kick kick-mode metrics tracker (offline-testable; zero isaaclab)."""

from __future__ import annotations

import importlib.util
import os
from collections import deque

import torch

KICK_FORCE_THRESHOLD_N = 0.1
"""踢击事件接触力阈值（上游 soccer_env.py:334-341 的 0.1 同款，单位 N）。"""

KICK_WINDOW_DEFAULT = 100
"""audit 行滚动窗口 episode 数（10s/500 步 episode 下平滑 ~每 21 it 一批
完成 episode 的稀疏到达；window 均值随 audit 行逐迭代输出）。"""

KICK_POLICY_OBS_DIM = 55
"""kick 面 policy obs 总维 mirror（= ball_mdp.KICK_OBS_DIM；模块尾部交叉
断言防两处漂移）。"""

KICK_METRIC_KEYS = ("goal_rate_win", "ball_disp_ep_mean",
                    "max_ball_speed_ep_mean", "kick_count_ep_mean",
                    "behind_ball_frac_ep_mean")
"""audit 行/张力表 kick 列名（冻结顺序 = format_audit_line 输出序）。"""

_EPISODE_KEYS = ("goal", "ball_disp", "max_ball_speed", "kick_count",
                 "behind_frames", "frames")


def _load_ball_mdp():
    """ball_mdp 复用面双路加载（单一实现源；离线降级按路径加载）。"""
    try:
        from hierarchical_kick.sim.mdp import ball_mdp
        return ball_mdp
    except Exception:
        pass
    # 离线（无 isaaclab）：父包 sim/mdp/__init__ 顶层 import isaaclab →
    # 按路径直接加载 ball_mdp（其内部 isaaclab 守卫式 import 自动降级到
    # 仓库离线数学单源；test_upper_kick_env 同款）
    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "sim", "mdp",
        "ball_mdp.py"))
    spec = importlib.util.spec_from_file_location("_ukick_ball_mdp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BMDP = _load_ball_mdp()
if int(getattr(_BMDP, "KICK_OBS_DIM", -1)) != KICK_POLICY_OBS_DIM:
    raise ValueError(
        f"[upper][ukick] KICK obs 维 mirror 漂移: kick_metrics."
        f"{KICK_POLICY_OBS_DIM} != ball_mdp.{getattr(_BMDP, 'KICK_OBS_DIM', None)}")

# kick_event 奖励签名默认阈值 mirror 交叉断言（单一实现源防漂移
# ——ball_mdp.kick_event_reward 的 force_threshold 缺省必须 = 本模块
# KICK_FORCE_THRESHOLD_N；KICK_OBS_DIM mirror 同款机制）。
_kick_evt_fn = getattr(_BMDP, "kick_event_reward", None)
_kick_evt_thr = ((_kick_evt_fn.__defaults__ or (None,))[0]
                 if _kick_evt_fn is not None else None)
if _kick_evt_thr is None or float(_kick_evt_thr) != float(KICK_FORCE_THRESHOLD_N):
    raise ValueError(
        f"[upper][ukick] kick_event 阈值 mirror 漂移: ball_mdp."
        f"kick_event_reward force_threshold 缺省 {_kick_evt_thr!r} != "
        f"kick_metrics.KICK_FORCE_THRESHOLD_N {KICK_FORCE_THRESHOLD_N!r}")


def _flatten_force_norm(force: torch.Tensor) -> torch.Tensor:
    """接触力张量 (…,3) → 每 env 最大合范数 (N,)（body/filter 维全取 max）。"""
    norms = torch.linalg.vector_norm(force.detach(), dim=-1)  # (…,)
    return norms.flatten(1).amax(dim=1) if norms.dim() > 1 else norms


class KickMetricsTracker:
    """kick 模式 per-episode 指标累积器（离线可测纯逻辑；零 isaaclab）。

    语义：
      - episode 初态锚定：``ball_start`` NaN 哨兵，update() 首帧（含 reset
        后首帧——pop_done 对 done env 清 NaN，下一帧即新 episode 初态）懒
        锚定球位；
      - update() 每步累积 max 球速 / kick 事件 / behind 帧数 / 总帧数；
      - pop_done(env, dones) 在 episode 结束帧聚合该批 env 的 episode 行
        （dict），推入滚动窗口并**清零对应槽位**（跨 reset 语义）；
      - window_means() 输出滚动窗口均值（空窗口 → 键面齐全值 None——
        audit 行固定列，解析面稳定）。
    device/num_envs 通用（构造期传入；buffer 随构造 device）。
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu",
                 kick_force_threshold: float = KICK_FORCE_THRESHOLD_N,
                 window: int = KICK_WINDOW_DEFAULT) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.kick_force_threshold = float(kick_force_threshold)
        if not (self.kick_force_threshold > 0.0):
            raise ValueError(
                f"[upper][ukick] kick_force_threshold 必须正: "
                f"{self.kick_force_threshold!r}")
        if int(window) <= 0:
            raise ValueError(f"[upper][ukick] window 必须正: {window!r}")
        self.window = int(window)
        self._reset_buffers()
        self._episodes: deque[dict] = deque(maxlen=self.window)

    # -- 状态面 ------------------------------------------------------------- #
    def _reset_buffers(self) -> None:
        z = torch.zeros(self.num_envs, device=self.device)
        self._ball_start = torch.full((self.num_envs, 3), float("nan"),
                                      device=self.device)
        self._max_speed = z.clone()
        self._kick_count = z.clone()
        self._behind_frames = z.clone()
        self._frames = z.clone()

    def reset(self, env_ids=None) -> None:
        """清零累积器（env_ids None → 全部；done 槽位由 pop_done 自动清）。"""
        if env_ids is None:
            self._reset_buffers()
            return
        idx = torch.as_tensor(env_ids, device=self.device).reshape(-1).long()
        self._ball_start[idx] = float("nan")
        self._max_speed[idx] = 0.0
        self._kick_count[idx] = 0.0
        self._behind_frames[idx] = 0.0
        self._frames[idx] = 0.0

    # -- 取数面（duck-typing；fake env 同面） -------------------------------- #
    @staticmethod
    def _ball_pos(env) -> torch.Tensor:
        return env.scene["ball"].data.root_pos_w.torch

    @staticmethod
    def _ball_lin_vel(env) -> torch.Tensor:
        return env.scene["ball"].data.root_lin_vel_w.torch

    @staticmethod
    def _robot_contact_force(env) -> torch.Tensor:
        """球侧传感器 robot↔ball 接触力 → 每 env 最大合范数 (N,)。

        主路径 = filter 面 ``force_matrix_w`` (N, B, F, 3)——只含 robot
        filter 体对球的力（ball_contact_forces filter=base+四足端）；
        ``net_forces_w`` 含地面反力（静止 ~4.2N ≫ 0.1N，必假触发），禁用
        ——无 filter 数据时 fail-fast 指路。
        """
        sensor = env.scene["ball_contact_forces"]
        data = sensor.data
        try:
            force = data.force_matrix_w  # (N, B, F, 3)；无 filter 面时 isaaclab raise
        except (AttributeError, RuntimeError) as e:
            raise RuntimeError(
                "[upper][ukick] 球侧接触传感器无 filter 力矩阵"
                "（force_matrix_w 缺失——kick 检测需 filter_prim_paths_expr "
                "配置；net_forces_w 含地面反力不可用）") from e
        return _flatten_force_norm(force)

    @classmethod
    def _cos_behind(cls, env) -> torch.Tensor:
        """cos_behind 列（ball_mdp._push_geometry_b_impl 单一实现源复用）。"""
        robot = env.scene["robot"]
        geom = _BMDP._push_geometry_b_impl(
            robot.data.root_pos_w.torch, cls._ball_pos(env), robot.data.root_quat_w.torch)
        return geom[:, 0]

    @staticmethod
    def _goal_flag(env) -> torch.Tensor:
        """goal_scored 判据真值 (N,) bool：终止面优先（authoritative——与
        DoneTerm 同函数/同参数），退化路径解析复算（离线 fake env 无
        termination_manager 面）。"""
        tm = getattr(env, "termination_manager", None)
        if tm is not None and hasattr(tm, "get_term"):
            try:
                return tm.get_term("goal_scored").to(torch.bool)
            except Exception:
                pass
        return _BMDP._goal_scored_impl(
            KickMetricsTracker._ball_pos(env)).to(torch.bool)

    # -- 每步累积 / episode 聚合 --------------------------------------------- #
    def update(self, env) -> None:
        """每 rollout 步累积（env.step 后调用；obs 时点的 scene buffer）。"""
        ball_pos = self._ball_pos(env).detach().to(self.device)
        speed = torch.linalg.vector_norm(
            self._ball_lin_vel(env).detach().to(self.device), dim=-1)
        force = self._robot_contact_force(env).to(self.device)
        behind = (self._cos_behind(env).detach().to(self.device) > 0.0)
        # episode 初态锚定（NaN 槽位懒锚——首帧/reset 后首帧）
        unanchored = torch.isnan(self._ball_start[:, 0])
        if bool(unanchored.any()):
            self._ball_start[unanchored] = ball_pos[unanchored]
        self._max_speed = torch.maximum(self._max_speed, speed)
        self._kick_count += (force > self.kick_force_threshold).to(
            self._kick_count.dtype)
        self._behind_frames += behind.to(self._behind_frames.dtype)
        self._frames += 1.0

    def pop_done(self, env, dones: torch.Tensor) -> list[dict]:
        """episode 结束帧聚合（done env 批）→ episode 行列表（并推窗口）。

        行键面：goal(bool) / ball_disp / max_ball_speed / kick_count /
        behind_frames / frames。ball_disp = |末态xy − 初态xy|（水平位移；
        z 向落地抖动不入量）。done 槽位随即清零（跨 reset 语义——下一帧
        update() 重锚定新 episode 初态）。
        """
        done = dones.detach().to(self.device).to(torch.bool).reshape(-1)
        if not bool(done.any()):
            return []
        ball_pos = self._ball_pos(env).detach().to(self.device)
        goal = self._goal_flag(env).detach().to(self.device) & done
        disp = torch.linalg.vector_norm(
            ball_pos[done, :2] - self._ball_start[done, :2], dim=-1)
        rows: list[dict] = []
        done_idx = done.nonzero(as_tuple=False).flatten()
        idxs = done_idx.tolist()
        disp_l = disp.tolist()
        goal_l = goal[done].tolist()
        ms_l = self._max_speed[done].tolist()
        kc_l = self._kick_count[done].tolist()
        bf_l = self._behind_frames[done].tolist()
        fr_l = self._frames[done].tolist()
        for env_id, g, d, ms, kc, bf, fr in zip(
                idxs, goal_l, disp_l, ms_l, kc_l, bf_l, fr_l):
            row = {"env": int(env_id), "goal": bool(g), "ball_disp": float(d),
                   "max_ball_speed": float(ms), "kick_count": float(kc),
                   "behind_frames": float(bf), "frames": float(fr)}
            rows.append(row)
        self._episodes.extend(rows)
        self.reset(done_idx)
        return rows

    def window_means(self) -> dict[str, float | None]:
        """滚动窗口均值列（键面冻结 = KICK_METRIC_KEYS；空窗口全 None）。"""
        if not self._episodes:
            return {k: None for k in KICK_METRIC_KEYS}
        rows = list(self._episodes)
        n = len(rows)
        return {
            "goal_rate_win": sum(r["goal"] for r in rows) / n,
            "ball_disp_ep_mean": sum(r["ball_disp"] for r in rows) / n,
            "max_ball_speed_ep_mean":
                sum(r["max_ball_speed"] for r in rows) / n,
            "kick_count_ep_mean": sum(r["kick_count"] for r in rows) / n,
            "behind_ball_frac_ep_mean":
                sum(r["behind_frames"] / r["frames"] for r in rows
                    if r["frames"] > 0) / max(
                    sum(1 for r in rows if r["frames"] > 0), 1),
        }

    @property
    def episode_count(self) -> int:
        """窗口内 episode 行数（诊断面）。"""
        return len(self._episodes)


def format_kick_metrics_segment(kick: dict | None) -> str:
    """audit 行 kick 尾段（纯函数；单测对象）。

    ``kick goal_rate=… ball_disp=… max_ball_speed=… kick_count=…
    behind_frac=…``（固定列序；None/非有限 → na——解析面稳定）。
    kick=None → 空串（velocity 面 audit 行逐位不变）。
    """
    if kick is None:
        return ""
    def _f(v, nd):
        if v is None:
            return "na"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "na"
        if v != v or v in (float("inf"), float("-inf")):
            return "na"
        return f"{v:.{nd}f}"
    return (" kick goal_rate=%s ball_disp=%s max_ball_speed=%s "
            "kick_count=%s behind_frac=%s" % (
                _f(kick.get("goal_rate_win"), 4),
                _f(kick.get("ball_disp_ep_mean"), 3),
                _f(kick.get("max_ball_speed_ep_mean"), 3),
                _f(kick.get("kick_count_ep_mean"), 2),
                _f(kick.get("behind_ball_frac_ep_mean"), 4)))


__all__ = ["KickMetricsTracker", "KICK_FORCE_THRESHOLD_N",
           "KICK_WINDOW_DEFAULT", "KICK_POLICY_OBS_DIM", "KICK_METRIC_KEYS",
           "format_kick_metrics_segment"]
