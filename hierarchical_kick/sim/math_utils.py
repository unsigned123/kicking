"""Pure-torch quaternion helpers（离线数学单源；ball_mdp 降级路径唯一消费方）。

vendored from UT mdp/math_utils（standalone 纯 torch 实现，
与 isaaclab.utils.math 的 ``quat_rotate_inverse`` 等价）；cmd-only 拆分时
按依赖闭包收入本包——ball_mdp 的 isaaclab 绑定面不可用（离线/测试桩）时
降级到本模块。

约定：quaternion 一律 xyzw（w 在末）。

[isaaclab 3.0 迁移] 约定从 2.x wxyz 翻转为 3.x xyzw：数据层
``.torch`` 视图与 isaaclab.utils.math 均已 xyzw，本离线单源同步翻转，
保证 isaaclab 绑定面与离线降级面在全链同序。
"""
from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse of unit quaternion == conjugate (input normalized internally)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return quat_conjugate(q)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by unit quaternion q (xyzw), broadcast over leading dims."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = q[..., 3:]
    xyz = q[..., :3]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of unit quaternion q (xyzw)."""
    return quat_rotate(quat_inv(q), v)


__all__ = ["quat_conjugate", "quat_inv", "quat_rotate", "quat_rotate_inverse"]
