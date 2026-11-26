"""55 维上层策略 obs 的部署侧构造（纯函数，与训练侧逐位同义）。

训练侧契约（hierarchical_kick KickPolicyCfg，单帧无 history）::

    [ joint_pos_rel(12)  # q − default_q
      joint_vel(12)      # ×0.05
      base_ang_vel(3)    # ×0.2
      projected_gravity(3)
      last_action(12)
      base_lin_vel(3)
      ball_rel_b(3)      # body 系 xy/3 + 距离/3
      goal_rel_b(4)      # body 系 xy/3 + sin + cos
      push_geometry_b(3) # cos_behind + sin + cos ]  = 55

四元数一律 **xyzw**（isaaclab 3.0 通道序）。所有输入为训练序关节量
（见 joint_spec 重排）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    pass

DIST_NORM = 3.0
"""距离归一尺度（3m 任务域，与训练侧 _DIST_NORM 同值）。"""

GOAL_CENTER_XY = (0.0, 3.0)
"""世界（里程计/动捕）系球门中心 xy（训练侧 GOAL_CENTER = (0, GOAL_Y=3.0)）。"""

SCALE_JOINT_VEL = 0.05
SCALE_BASE_ANG_VEL = 0.2

OBS_DIM = 55

_EPS = 1e-8


def quat_apply_inverse(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """q^-1 ⊗ v ⊗ q（world → body），quat 为 xyzw。纯 numpy。"""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    q = q / (np.linalg.norm(q) + _EPS)
    x, y, z, w = q
    # q^-1 = (-x,-y,-z,w)
    ix, iy, iz, iw = -x, -y, -z, w
    # t = 2 * cross(i.xyz, v)
    tx = 2.0 * (iy * v[..., 2] - iz * v[..., 1])
    ty = 2.0 * (iz * v[..., 0] - ix * v[..., 2])
    tz = 2.0 * (ix * v[..., 1] - iy * v[..., 0])
    # v' = v + iw*t + cross(i.xyz, t)
    rx = v[..., 0] + iw * tx + (iy * tz - iz * ty)
    ry = v[..., 1] + iw * ty + (iz * tx - ix * tz)
    rz = v[..., 2] + iw * tz + (ix * ty - iy * tx)
    return np.stack([rx, ry, rz], axis=-1)


def yaw_from_quat(quat_xyzw: np.ndarray) -> float:
    """xyzw 四元数 → 偏航角（rad）。"""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    x, y, z, w = q / (np.linalg.norm(q) + _EPS)
    return float(math.atan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def projected_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """body 系重力方向（单位向量，站立时 ≈ (0,0,-1)）。"""
    return quat_apply_inverse(quat_xyzw, np.array([0.0, 0.0, -1.0]))


def ball_rel_b(robot_xy: np.ndarray, ball_xy: np.ndarray,
               quat_xyzw: np.ndarray) -> np.ndarray:
    """[3] = [body 系球 xy/3, ‖xy/3‖]（与训练 _ball_rel_b_impl 同义）。"""
    diff_xy = np.asarray(ball_xy, dtype=np.float64)[:2] \
        - np.asarray(robot_xy, dtype=np.float64)[:2]
    diff = np.array([diff_xy[0], diff_xy[1], 0.0])
    rel = quat_apply_inverse(quat_xyzw, diff)
    rel_xy = rel[:2] / DIST_NORM
    return np.array([rel_xy[0], rel_xy[1], float(np.linalg.norm(rel_xy))])


def goal_rel_b(robot_xy: np.ndarray, quat_xyzw: np.ndarray,
               goal_xy: Sequence[float] = GOAL_CENTER_XY) -> np.ndarray:
    """[4] = [body 系门 xy/3, sin(角), cos(角)]（与 _goal_rel_b_impl 同义）。"""
    diff_xy = np.asarray(goal_xy, dtype=np.float64)[:2] \
        - np.asarray(robot_xy, dtype=np.float64)[:2]
    diff = np.array([diff_xy[0], diff_xy[1], 0.0])
    rel = quat_apply_inverse(quat_xyzw, diff)
    rel_xy = rel[:2] / DIST_NORM
    angle = math.atan2(float(rel[1]), float(rel[0]))
    return np.array([rel_xy[0], rel_xy[1], math.sin(angle), math.cos(angle)])


def push_geometry_b(robot_xy: np.ndarray, ball_xy: np.ndarray,
                    quat_xyzw: np.ndarray,
                    goal_xy: Sequence[float] = GOAL_CENTER_XY) -> np.ndarray:
    """[3] = [cos_behind, sin(fwd×btog), cos(fwd·btog)]（同训练侧实现）。"""
    robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]
    ball_xy = np.asarray(ball_xy, dtype=np.float64)[:2]
    goal = np.asarray(goal_xy, dtype=np.float64)[:2]
    yaw = yaw_from_quat(quat_xyzw)

    btog = goal - ball_xy
    btog_dir = btog / (float(np.linalg.norm(btog)) + _EPS)
    to_ball = ball_xy - robot_xy
    to_ball_dir = to_ball / (float(np.linalg.norm(to_ball)) + _EPS)
    cos_behind = float(np.dot(to_ball_dir, btog_dir))

    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    # sin(fwd×btog)（二维叉积 z 分量）与 cos(fwd·btog)
    sin_ang = float(fwd[0] * btog_dir[1] - fwd[1] * btog_dir[0])
    cos_ang = float(np.dot(fwd, btog_dir))
    return np.array([cos_behind, sin_ang, cos_ang])


def build_obs(joint_pos: Sequence[float], joint_vel: Sequence[float],
              base_ang_vel: Sequence[float], quat_xyzw: Sequence[float],
              last_action: Sequence[float], base_lin_vel: Sequence[float],
              robot_xy: Sequence[float], ball_xy: Sequence[float],
              default_q: Sequence[float],
              goal_xy: Sequence[float] = GOAL_CENTER_XY) -> np.ndarray:
    """构造 55 维 obs（输入关节量一律训练序，12 维）。"""
    q = np.asarray(joint_pos, dtype=np.float64)
    qd = np.asarray(joint_vel, dtype=np.float64)
    av = np.asarray(base_ang_vel, dtype=np.float64)
    lv = np.asarray(base_lin_vel, dtype=np.float64)
    lq = np.asarray(last_action, dtype=np.float64)
    dq = np.asarray(default_q, dtype=np.float64)
    for name, arr, dim in (("joint_pos", q, 12), ("joint_vel", qd, 12),
                           ("last_action", lq, 12), ("base_ang_vel", av, 3),
                           ("base_lin_vel", lv, 3)):
        if arr.shape != (dim,):
            raise ValueError(f"{name} 须 {dim} 维，实际 {arr.shape}")
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quat 须 4 维（xyzw），实际 {quat.shape}")
    obs = np.concatenate([
        q - dq,
        qd * SCALE_JOINT_VEL,
        av * SCALE_BASE_ANG_VEL,
        projected_gravity(quat),
        lq,
        lv,
        ball_rel_b(robot_xy, ball_xy, quat),
        goal_rel_b(robot_xy, quat, goal_xy),
        push_geometry_b(robot_xy, ball_xy, quat, goal_xy),
    ]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"obs 维错误: {obs.shape}"
    return obs


# --------------------------------------------------------------------------- #
# 坐标系变换（odom 系 → 动捕/世界系）：位置平移 + 偏航旋转（xyzw 四元数复合）
# --------------------------------------------------------------------------- #
def yaw_quat(yaw: float) -> np.ndarray:
    """偏航角 → 四元数 xyzw。"""
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """四元数乘法 a ⊗ b（xyzw）。"""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def rotate_xy(yaw: float, v: Sequence[float]) -> np.ndarray:
    """绕 z 轴旋转二维向量。"""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = float(v[0]), float(v[1])
    return np.array([c * x - s * y, s * x + c * y])


def transform_pose_2d(xy: Sequence[float], quat_xyzw: Sequence[float],
                      t_xy: Sequence[float], yaw_offset: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """把 odom 系的(位置, 姿态)刚体变换到世界系：p' = R·p + t, q' = q_yaw ⊗ q。

    机体系速度（vBody/omegaBody）**不需要**变换（旋转不变性：它们本来就表达
    在机体系）。仅当狗与球/门来自不同世界系时才需要本函数。
    """
    p = rotate_xy(yaw_offset, np.asarray(xy, dtype=np.float64)[:2]) \
        + np.asarray(t_xy, dtype=np.float64)[:2]
    q = quat_mul(yaw_quat(yaw_offset), np.asarray(quat_xyzw, dtype=np.float64))
    q = q / (np.linalg.norm(q) + _EPS)
    return p, q
