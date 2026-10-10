"""U1 ⑥ — 注册字符串面 + 通道契约适配表 + sim cfg 落地标记（cmd-only 拆分版：register 面 = hierarchical_kick.tasks）。

判据（docs/U0_UPPER_DESIGN.md §5 U1-6 与 §1.1 表）：
- register 模块离线 import（无 gymnasium 静默跳过）+ entry point 字符串面
  精确匹配设计 §4；字符串可解析（module:Attr 形状）。
- §1.1 适配表常量（contracts = 单一真相源）逐通道 == 设计值；⚠️ 官方项
  参数（joint_vel U±1.5×0.01 / ang_vel scale1）不匹配设计值——防照抄锚。
- flat_env_cfg 落地标记（isaaclab 面离线不可 import，源码标记 +
  contracts 符号消费为落地证据）。
"""
from __future__ import annotations

import ast
import os

import pytest

import hierarchical_kick.tasks as reg
from hierarchical_kick.core import contracts as C
from hierarchical_kick.core.scale_vec import (
    go2_action_scale_table,
    go2_joint_names_slot_order,
)
from hierarchical_kick.robots.go2 import GO2_JOINT_ORDER_IL

_PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(reg.__file__))))


# --------------------------------------------------------------------------- #
# 注册字符串面（U1-⑥）
# --------------------------------------------------------------------------- #
def test_register_strings_exact():
    """任务 id / entry point 字符串 == 名称契约冻结值（仅存 cmd 面）。"""
    assert reg.KICK_CMD_TASK_ID == "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"
    assert reg.KICK_CMD_TASK_ENTRY_POINT == "isaaclab.envs:ManagerBasedRLEnv"
    assert reg.KICK_CMD_ENV_CFG_ENTRY_POINT == (
        "hierarchical_kick.sim.ball_env_cfg:Go2KickFlatEnvCfgCommand")
    assert reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT == (
        "hierarchical_kick.agents:UpperKickCmdPPORunnerCfg")


def test_register_entry_points_parseable():
    """entry point 全部 ``module.path:Attr`` 形状（懒加载可解析）。"""
    for point in (reg.KICK_CMD_TASK_ENTRY_POINT, reg.KICK_CMD_ENV_CFG_ENTRY_POINT,
                  reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT):
        module_path, attr = point.split(":", 1)
        assert module_path and attr and "." in module_path
        assert not module_path.startswith("isaaclab_tasks"), \
            "零 isaaclab_tasks 树改动（U0 §4）"


def test_register_module_imports_offline_silently():
    """离线（无 gymnasium/isaaclab）import 静默跳过（try/except 面在案）。"""
    src = open(os.path.join(_PKG, "hierarchical_kick", "tasks",
                            "__init__.py"), encoding="utf-8").read()
    assert "except (ImportError, ModuleNotFoundError)" in src
    assert "import gymnasium as gym" in src
    # gym.register kwargs 全部消费模块常量（无内联字符串漂移面）
    assert 'id=KICK_CMD_TASK_ID' in src
    assert 'entry_point=KICK_CMD_TASK_ENTRY_POINT' in src
    assert '"env_cfg_entry_point": KICK_CMD_ENV_CFG_ENTRY_POINT' in src
    assert '"rsl_rl_cfg_entry_point": KICK_CMD_RSL_RL_CFG_ENTRY_POINT' in src


# --------------------------------------------------------------------------- #
# §1.1 通道适配表（contracts = 单一真相源；U1-④ cfg 参数面）
# --------------------------------------------------------------------------- #
def test_adaptation_table_channels():
    """42 通道偏移 + 48/12/32 维度契约（§1.1 表列号；段空间 Z_DIM=32，
    旧 64 面由 LEGACY_Z_DIM 复现位承载）。"""
    assert (C.OBS_DIM, C.PROPRIO_DIM, C.Z_DIM, C.ACTION_DIM) == \
        (48, 42, 32, 12)
    assert C.LEGACY_Z_DIM == 64
    assert (C.SEG_RATE_HZ, C.FRAMES_PER_SEG, C.SEG_LEN_S) == (12.5, 4, 0.08)
    assert C.FRAMES_PER_SEG * C.SEG_RATE_HZ == 50.0   # 段对齐时钟律
    assert (C.OFF_JOINT_POS, C.OFF_JOINT_VEL, C.OFF_BASE_ANG_VEL,
            C.OFF_PROJ_GRAVITY, C.OFF_LAST_ACTION) == (0, 12, 24, 27, 30)


def test_adaptation_table_noise_scale_design_values():
    """逐通道噪声/缩放 == 设计 §1.1 值；官方项参数显式不匹配（防照抄）。"""
    assert C.NOISE_JOINT_POS == 0.01
    assert C.NOISE_JOINT_VEL == 0.5 and C.SCALE_JOINT_VEL == 0.05
    assert C.NOISE_BASE_ANG_VEL == 0.2 and C.SCALE_BASE_ANG_VEL == 0.2
    assert C.NOISE_PROJ_GRAVITY == 0.05
    assert C.NOISE_BASE_LIN_VEL == 0.1
    # ⚠️ 官方 velocity 项参数（不得照抄的负锚）
    assert (C.NOISE_JOINT_VEL, C.SCALE_JOINT_VEL) != (1.5, 0.01)
    assert C.SCALE_BASE_ANG_VEL != 1.0
    # 探索/scale 公式常量（§1.4）
    assert C.SIGMA_SEM == 0.2 and C.SCALE_COEF == 0.25 and C.KP == 40.0


def _approx(x):
    return pytest.approx(x, abs=1e-12)


def test_scale_vec_slot_order_and_values():
    """scale 表槽位序 = PhysX（GO2_JOINT_ORDER_IL）；值 = 0.25×effort/KP。"""
    names = go2_joint_names_slot_order()
    assert names == list(GO2_JOINT_ORDER_IL)
    table = go2_action_scale_table()
    effort = {"abad": 23.7, "hip": 23.7, "knee": 45.43}
    suffix2part = {"hip": "abad", "thigh": "hip", "calf": "knee"}
    for n in names:
        part = suffix2part[n.split("_", 1)[1].removesuffix("_joint")]
        assert table[n] == _approx(0.25 * effort[part] / 40.0)


# --------------------------------------------------------------------------- #
# flat_env_cfg 落地标记（离线源码面）
# --------------------------------------------------------------------------- #
def test_flat_env_cfg_upper_markers():
    """适配 U1–U5 的落地证据（源码标记 + contracts 符号消费）。"""
    src = open(os.path.join(_PKG, "hierarchical_kick", "sim", "flat_env_cfg.py"),
               encoding="utf-8").read()
    # [U1] obs 整组替换 + 42 前缀切片契约符号
    assert "UpperPolicyCfg" in src and "UpperObservationsCfg" in src
    for sym in ("NOISE_JOINT_POS", "NOISE_JOINT_VEL", "SCALE_JOINT_VEL",
                "NOISE_BASE_ANG_VEL", "SCALE_BASE_ANG_VEL",
                "NOISE_PROJ_GRAVITY", "NOISE_BASE_LIN_VEL"):
        assert sym in src, f"适配表符号未消费: {sym}"
    # [U2] 逐关节 scale dict；[U3] heading off；[U4] height 0.15；[U5] β1
    assert "go2_action_scale_table()" in src
    assert "heading_command = False" in src
    assert '"threshold": 0.15' in src and "mdp.root_height" in src
    assert "mdp.alive_reward" in src and "weight=0.25" in src
    # tracking 装配单源（D3；官方 UNITREE_GO2_CFG 直用不出现）
    assert "make_go2_track_profile" in src
    assert "soft_joint_pos_limit_factor = 0.9" in src
    assert "from isaaclab_assets" not in src   # 官方资产面零直用（§3 禁用）


def test_flat_env_cfg_syntax_and_vendor_headers():
    """sim 包可解析（AST）；vendor 文件头来源块在案。"""
    for rel in ("hierarchical_kick/sim/flat_env_cfg.py",
                "hierarchical_kick/sim/velocity_env_cfg.py",
                "hierarchical_kick/sim/mdp/rewards.py",
                "hierarchical_kick/sim/mdp/curriculums.py",
                "hierarchical_kick/sim/mdp/terminations.py"):
        path = os.path.join(_PKG, rel)
        ast.parse(open(path, encoding="utf-8").read())   # 语法可解析
    # velocity_env_cfg 与官方原文唯一 diff = mdp import（D1）
    src = open(os.path.join(_PKG, "hierarchical_kick", "sim",
                            "velocity_env_cfg.py"),
               encoding="utf-8").read()
    assert "from hierarchical_kick.sim import mdp" in src
    assert "isaaclab_tasks.manager_based.locomotion" not in src
