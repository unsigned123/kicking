"""obs_builder / joint_spec / policy 离线单测（无 ROS2 / 无 isaaclab 依赖）。"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kick_sim2real import joint_spec, obs_builder  # noqa: E402


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def test_permutation_and_defaults():
    perm = joint_spec.sdk_to_train_permutation()
    assert len(perm) == 12
    # 训练序第 0/4/8 位 = FR_abad / FR_hip / FR_knee → SDK 序 0/1/2
    assert perm[0] == 0 and perm[4] == 1 and perm[8] == 2
    sdk = list(range(12))
    train = joint_spec.reorder_sdk_to_train(sdk)
    # SDK 0/1/2 → 训练 0/4/8；SDK 3/4/5 → 训练 1/5/9
    assert train[0] == 0 and train[4] == 1 and train[8] == 2
    assert train[1] == 3 and train[5] == 4 and train[9] == 5
    assert len(joint_spec.TRAIN_DEFAULT_Q) == 12


def test_projected_gravity_upright():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    g = obs_builder.projected_gravity(q)
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-9)
    # yaw 旋转不改变重力投影
    g2 = obs_builder.projected_gravity(_quat_yaw(1.2))
    np.testing.assert_allclose(g2, [0.0, 0.0, -1.0], atol=1e-9)


def test_ball_rel_b_golden():
    """狗在原点朝 +x，球在正前方 1.5m → [0, 0.5, 0.5]（/3 归一）。"""
    q = np.array([0.0, 0.0, 0.0, 1.0])
    out = obs_builder.ball_rel_b(np.zeros(2), np.array([1.5, 0.0]), q)
    np.testing.assert_allclose(out, [0.5, 0.0, 0.5], atol=1e-6)
    # 狗朝 +y（yaw=π/2），球在 (0,1.5) world → body 系 (1.5→ rel x? )
    qy = _quat_yaw(math.pi / 2)
    out2 = obs_builder.ball_rel_b(np.zeros(2), np.array([0.0, 1.5]), qy)
    np.testing.assert_allclose(out2, [0.5, 0.0, 0.5], atol=1e-6)


def test_goal_rel_b_golden():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    out = obs_builder.goal_rel_b(np.zeros(2), q, goal_xy=(0.0, 3.0))
    # 门在 +y，body 系 rel=(0,1) → xy=(0,1), angle=π/2 → sin=1 cos=0
    np.testing.assert_allclose(out, [0.0, 1.0, 1.0, 0.0], atol=1e-6)


def test_push_geometry_behind():
    """狗在球正后方（球在狗与门连线上）→ cos_behind=1。"""
    q = np.array([0.0, 0.0, 0.0, 1.0])   # 朝 +x
    robot = np.array([0.0, 0.0])
    ball = np.array([1.0, 0.0])
    goal = (2.0, 0.0)                     # 门在球正前方
    out = obs_builder.push_geometry_b(robot, ball, q, goal_xy=goal)
    assert out[0] == pytest.approx(1.0, abs=1e-6)   # 正后方
    assert out[2] == pytest.approx(1.0, abs=1e-6)   # 朝向=球→门 方向


def test_build_obs_shape_and_prefix():
    q = np.array(joint_spec.TRAIN_DEFAULT_Q)
    obs = obs_builder.build_obs(
        joint_pos=q, joint_vel=np.zeros(12), base_ang_vel=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), last_action=np.zeros(12),
        base_lin_vel=np.zeros(3), robot_xy=np.zeros(2),
        ball_xy=np.array([1.0, 0.0]), default_q=q,
        goal_xy=(0.0, 3.0))
    assert obs.shape == (55,)
    assert obs.dtype == np.float32
    # joint_pos_rel 前缀全 0（q == default）
    np.testing.assert_allclose(obs[:12], 0.0, atol=1e-9)
    # 通道布局切片自检
    assert obs[12:24].shape == (12,)   # joint_vel
    assert obs[30:42].shape == (12,)   # last_action
    assert obs[45:48].shape == (3,)    # ball_rel_b


def test_policy_loader_roundtrip(tmp_path):
    """合成 rsl_rl 风格 ckpt → 导出 → 加载 → 输出维 3。"""
    import torch
    from torch import nn

    from kick_sim2real.policy import PolicyRuntime, export_ckpt

    layers = nn.Sequential(nn.Linear(55, 16), nn.ELU(),
                           nn.Linear(16, 16), nn.ELU(),
                           nn.Linear(16, 3))
    # rsl_rl 键面
    state = {
        "mlp.0.weight": layers[0].weight, "mlp.0.bias": layers[0].bias,
        "mlp.2.weight": layers[2].weight, "mlp.2.bias": layers[2].bias,
        "mlp.4.weight": layers[4].weight, "mlp.4.bias": layers[4].bias,
        "distribution.std_param": torch.zeros(3),
    }
    src = tmp_path / "raw.pt"
    torch.save({"model_state_dict": state, "iter": 7}, src)
    dst = tmp_path / "policy.pt"
    meta = export_ckpt(str(src), str(dst))
    assert meta["obs_dim"] == 55 and meta["hidden_dims"] == [16, 16]
    rt = PolicyRuntime(str(dst))
    cmd = rt.act(np.zeros(55, dtype=np.float32))
    assert cmd.shape == (3,)


def test_select_gait_and_limit():
    from kick_sim2real.transport import GAITS, select_gait

    # 慢速 → slow(303)；中速 → mid(308)；快速 → fast(305)
    assert select_gait(0.3, 0.1, 0.5)[0] == "slow"
    assert select_gait(0.9, 0.1, 0.9)[0] == "mid"
    assert select_gait(1.5, 0.4, 2.0)[0] == "fast"
    # 超限幅
    name, cmd = select_gait(3.0, 9.0, 9.0)
    assert name == "fast"
    np.testing.assert_allclose(cmd, [1.6, 0.55, 2.5])
    assert GAITS["slow"]["id"] == 303 and GAITS["mid"]["id"] == 308 \
        and GAITS["fast"]["id"] == 305


def test_spi_data_decode_signs_and_order():
    """合成 spi_data 包：验证符号翻转 + SDK→训练序重排。"""
    import struct

    from kick_sim2real import joint_spec
    from kick_sim2real.lcm_sources import (decode_spi_data,
                                           SPI_PAYLOAD_OFFSET)

    # 每腿 (abad, hip, knee) 原始读数 = 1.0/2.0/3.0（SDK 腿序 FR,FL,RR,RL）
    buf = bytearray(SPI_PAYLOAD_OFFSET + 260)
    for field_off, val in ((0, 1.0), (16, 2.0), (32, 3.0),
                           (48, 0.1), (64, 0.2), (80, 0.3)):
        struct.pack_into("<4f", buf, SPI_PAYLOAD_OFFSET + field_off,
                         val, val, val, val)
    dec = decode_spi_data(bytes(buf))
    # FR: [+1, −2, −3]；FL 同；RR/RL 同
    assert dec["q"][:3] == (1.0, -2.0, -3.0)
    assert dec["q"][3:6] == (1.0, -2.0, -3.0)
    assert abs(dec["qd"][1] + 0.2) < 1e-6
    # SDK 序 → 训练序：SDK[0..2]=FR 三连 → 训练 idx 0/4/8
    train = joint_spec.reorder_sdk_to_train(dec["q"])
    assert train[0] == 1.0 and train[4] == -2.0 and train[8] == -3.0


def test_global_transform_invariance():
    """全局平移+偏航变换（狗/球/门同步变换）→ 55 维 obs 逐位不变。"""
    from kick_sim2real.obs_builder import (build_obs, rotate_xy,
                                           transform_pose_2d)
    q = np.array([0.0, 0.0, math.sin(0.3), math.cos(0.3)])
    xy, ball, goal = np.array([1.0, 1.0]), np.array([2.0, 0.5]), (0.0, 3.0)
    kw = dict(joint_pos=np.zeros(12), joint_vel=np.zeros(12),
              base_ang_vel=np.zeros(3), last_action=np.zeros(12),
              base_lin_vel=np.zeros(3), default_q=np.zeros(12))
    o1 = build_obs(quat_xyzw=q, robot_xy=xy, ball_xy=ball, goal_xy=goal, **kw)
    T, yaw = np.array([2.0, -1.0]), 0.7
    xy2, q2 = transform_pose_2d(xy, q, T, yaw)
    ball2 = rotate_xy(yaw, ball) + T
    goal2 = tuple(rotate_xy(yaw, goal) + T)
    o2 = build_obs(quat_xyzw=q2, robot_xy=xy2, ball_xy=ball2,
                   goal_xy=goal2, **kw)
    np.testing.assert_allclose(o1, o2, atol=1e-6)


def test_transform_pose_2d_quaternion():
    """姿态变换：偏航复合 = 偏航角相加；位置按 R·p+t。"""
    from kick_sim2real.obs_builder import transform_pose_2d, yaw_from_quat
    p, q = transform_pose_2d([0.0, 0.0], [0, 0, 0, 1], [1.0, 2.0], math.pi / 2)
    np.testing.assert_allclose(p, [1.0, 2.0], atol=1e-9)
    assert abs(yaw_from_quat(q) - math.pi / 2) < 1e-6
