"""cyberdog2 kick 面接入单测（阶段二；离线面：常量/表/源码标记）。

覆盖面（go2 kick 链零回归由既有套件保障，本文件只测 cyberdog2 追加面）：
- core/scale_vec：cyberdog2 表 = xiaomi deploy 快照 uniform 0.25 ×12
  （槽位序 = CYBERDOG2_JOINT_ORDER_IL）、vec、分派面 Unknown fail-fast、
  go2 表逐位不变；
- config/robots/cyberdog2.yaml：registry 字段一致性（KP/scale/base_body/
  spawn_height/fall_height/joint_map 恒等名）；
- ball_mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD = 0.20（常量契约 + __all__）；
- ball_env_cfg 源码标记：本体置换 C1-C7 逐项（profile/filter/body names/
  阈值覆写两处同源/scale 表分派）+ 类定义 + go2 面 go2 表分派不变；
- tasks 注册：cyberdog2 kick cmd 任务 id/entry point 冻结字符串 + gym 接线；
- agents：experiment 隔离子类（唯一 diff = experiment_name）；
- frozen_weights：cyberdog2 official 缺省路径 + 分派函数（Unknown fail-fast）；
- train/eval --robot 分派源码标记。
"""
from __future__ import annotations

import ast
import importlib.util
import os

import pytest
import torch
import yaml

from hierarchical_kick.core import scale_vec
from hierarchical_kick.frozen_weights import (
    DEFAULT_LOW_LEVEL_CKPT,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2,
    default_low_level_ckpt_official,
)
from hierarchical_kick.robots import (
    CYBERDOG2_FOOT_BODIES,
    CYBERDOG2_JOINT_ORDER_IL,
)

_PKG_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
_BALL_MDP_PATH = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "sim", "mdp", "ball_mdp.py")
_BALL_CFG_PATH = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "sim", "ball_env_cfg.py")
_REGISTER_PATH = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "tasks", "__init__.py")
_CYBERDOG2_YAML = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "config", "robots", "cyberdog2.yaml")


def _read(rel: str) -> str:
    with open(os.path.join(_PKG_ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _parse(rel: str) -> None:
    ast.parse(_read(rel))


def _load_ball_mdp():
    """按路径加载 ball_mdp（父包 __init__ 含 isaaclab 面，离线不可连带）。"""
    spec = importlib.util.spec_from_file_location(
        "hkick_ball_mdp_cyberdog2", _BALL_MDP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. scale_vec：cyberdog2 表（xiaomi deploy 快照 uniform 0.25）
# --------------------------------------------------------------------------- #

def test_cyberdog2_action_scale_table_snapshot_uniform():
    """表 = 快照 uniform 0.25 ×12；键序 = CYBERDOG2_JOINT_ORDER_IL 槽位序。"""
    table = scale_vec.cyberdog2_action_scale_table()
    assert list(table) == list(CYBERDOG2_JOINT_ORDER_IL)
    assert len(table) == 12
    assert set(table.values()) == {0.25}
    vec = scale_vec.cyberdog2_action_scale_vec()
    assert vec.shape == (12,) and vec.dtype == torch.float32
    assert torch.equal(
        vec, torch.full((12,), 0.25, dtype=torch.float32))


def test_cyberdog2_yaml_registry_consistency():
    """cyberdog2.yaml 字段一致性（scale_vec 校验面 + 文档面）。"""
    assert os.path.isfile(_CYBERDOG2_YAML)
    with open(_CYBERDOG2_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert raw["name"] == "cyberdog2"
    assert raw["actuator"]["stiffness"] == scale_vec.CYBERDOG2_KP == 25.0
    assert raw["actuator"]["damping"] == 0.5
    assert raw["actuator"]["effort"] == 12.0
    assert raw["action_scale_value"] == \
        scale_vec.CYBERDOG2_ACTION_SCALE_SNAPSHOT == 0.25
    assert raw["base_body"] == "body"
    assert raw["spawn_height"] == 0.30      # FK 校正值（站高 0.236）
    assert raw["fall_height"] == 0.17       # kick 面 robot_fallen 高度支（探针标定）
    assert raw["soft_joint_pos_limit_factor"] == 0.9
    # joint_map 恒等名（abad/hip/knee 直名，无 go2 的 hip→thigh 改名）
    for leg in ("FL", "FR", "RL", "RR"):
        assert raw["joint_map"][leg] == {
            "abad": f"{leg}_abad_joint",
            "hip": f"{leg}_hip_joint",
            "knee": f"{leg}_knee_joint"}


def test_action_scale_table_for_dispatch():
    """分派面：go2/cyberdog2 同名注册表分派 + Unknown fail-fast。"""
    assert scale_vec.action_scale_table_for("go2") == \
        scale_vec.go2_action_scale_table()
    assert scale_vec.action_scale_table_for("cyberdog2") == \
        scale_vec.cyberdog2_action_scale_table()
    with pytest.raises(ValueError):
        scale_vec.action_scale_table_for("go1")


def test_go2_table_regression_unchanged():
    """go2 表逐位回归（0.148125×8 + 0.2839375×4；cyberdog2 追加不扰动）。"""
    table = scale_vec.go2_action_scale_table()
    vals = sorted(set(table.values()))
    assert vals == [pytest.approx(0.148125), pytest.approx(0.2839375)]
    assert sum(1 for v in table.values() if abs(v - 0.148125) < 1e-9) == 8
    assert sum(1 for v in table.values() if abs(v - 0.2839375) < 1e-9) == 4


# --------------------------------------------------------------------------- #
# 2. ball_mdp：cyberdog2 摔判定高度阈值常量
# --------------------------------------------------------------------------- #

def test_ball_mdp_cyberdog2_threshold_constant():
    """0.17 常量契约 + __all__ 导出（go2 0.16 不动；owner 建议值 0.20 被探针
    步态波谷实测 0.1839 否决——依据见常量 docstring）。"""
    mdp = _load_ball_mdp()
    assert mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD == 0.17
    assert mdp.FALLEN_BODY_Z_THRESHOLD == 0.16
    assert "CYBERDOG2_FALLEN_BODY_Z_THRESHOLD" in mdp.__all__


# --------------------------------------------------------------------------- #
# 3. ball_env_cfg：cyberdog2 本体置换（C1-C7）源码标记
# --------------------------------------------------------------------------- #

def test_ball_env_cfg_cyberdog2_source_markers():
    """本体置换单点 helper + 双 cfg 类 + C1-C7 逐项标记（AST 可解析）。"""
    src = _read("hierarchical_kick/sim/ball_env_cfg.py")
    ast.parse(src)
    # 类定义（训练面 + cmd 面；单亲继承保持 MRO 线性）
    assert "class CyberDog2KickFlatEnvCfg(Go2KickFlatEnvCfg):" in src
    assert ("class CyberDog2KickFlatEnvCfgCommand("
            "Go2KickFlatEnvCfgCommand):" in src)
    # C1 本体置换（profile 复用 velocity 面装配；soft limit 0.9 显式）
    assert "make_cyberdog2_track_profile(prim_path=\"{ENV_REGEX_NS}/Robot\")" in src
    # C2 狗 spawn 常量重放（置换后重放；z 沿 profile 现款）
    assert src.count("_dog_init.rot = DOG_SPAWN_YAW_QUAT_XYZW") == 2
    # C3 球接触 filter（body + CYBERDOG2_FOOT_BODIES；显式枚举无通配）
    assert "from hierarchical_kick.robots import CYBERDOG2_FOOT_BODIES" in src
    swap_seg = src.split("def _apply_cyberdog2_body_swap", 1)[1]
    assert '"{ENV_REGEX_NS}/Robot/body"' in swap_seg
    assert "for b in CYBERDOG2_FOOT_BODIES" in swap_seg
    assert ".*" not in swap_seg.split("filter_prim_paths_expr=[", 1)[1]\
        .split("]", 1)[0]
    # C4/C5 events + base_contact body names（body）
    assert swap_seg.count('.body_names = "body"') == 3
    # C6 阈值覆写：终止 + 奖励两处同源（mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD；
    # swap 段第 3 处 = 注释提及，实际赋值 = 终止/奖励 params 两处）
    assert swap_seg.count(
        'params["body_z_threshold"] = \\\n        mdp.CYBERDOG2_FALLEN_BODY_Z_THRESHOLD') == 2
    # C7 cmd 面 scale 表分派（cyberdog2 表；go2 面保持 go2 分派）
    assert 'action_scale_table_for("go2")' in src
    assert 'action_scale_table_for("cyberdog2")' in src
    # C8 低层计算频率分派（cyberdog2 cmd 面 50Hz；frozen_loco_action 旗标在位）
    assert "low_level_compute_per_substep = False" in src
    fl_src = _read("hierarchical_kick/sim/mdp/frozen_loco_action.py")
    ast.parse(fl_src)
    assert "low_level_compute_per_substep: bool = True" in fl_src
    assert "if not self._per_substep:" in fl_src
    # __all__ 导出
    assert '"CyberDog2KickFlatEnvCfg"' in src
    assert '"CyberDog2KickFlatEnvCfgCommand"' in src


# --------------------------------------------------------------------------- #
# 4. tasks / agents / frozen_weights / scripts 分派面
# --------------------------------------------------------------------------- #

def test_tasks_cyberdog2_kick_registration_strings():
    """cyberdog2 kick cmd 任务 id / entry point 冻结字符串 + __all__。"""
    import hierarchical_kick.tasks as reg
    assert reg.KICK_CMD_CYBERDOG2_TASK_ID == \
        "Isaac-Kick-Flat-CyberDog2-Upper-Cmd-v0"
    assert reg.KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT == (
        "hierarchical_kick.sim.ball_env_cfg:CyberDog2KickFlatEnvCfgCommand")
    assert reg.KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT == (
        "hierarchical_kick.agents:UpperCyberDog2KickCmdPPORunnerCfg")
    for name in ("KICK_CMD_CYBERDOG2_TASK_ID",
                 "KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT",
                 "KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT"):
        assert name in reg.__all__, name
    # gym 接线（源码面：register kwargs 在位；懒加载零 eager import）
    src = open(_REGISTER_PATH, encoding="utf-8").read()
    ast.parse(src)
    assert "id=KICK_CMD_CYBERDOG2_TASK_ID" in src
    assert '"env_cfg_entry_point": KICK_CMD_CYBERDOG2_ENV_CFG_ENTRY_POINT' in src
    assert '"rsl_rl_cfg_entry_point": KICK_CMD_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT' in src


def test_agents_cyberdog2_experiment_isolation():
    """cyberdog2 runner cfg：唯一 diff = experiment_name（公平性纪律）。"""
    src = _read("hierarchical_kick/agents.py")
    ast.parse(src)
    assert ("class UpperCyberDog2KickCmdPPORunnerCfg("
            "UpperKickCmdPPORunnerCfg):" in src)
    assert 'experiment_name = "upper_ukick_cmd_cyberdog2"' in src
    # go2 版 experiment 不动
    assert 'experiment_name = "upper_ukick_cmd"' in src


def test_frozen_weights_cyberdog2_dispatch():
    """official 缺省路径分派（go2 平面不变 / cyberdog2 子目录）+ fail-fast。"""
    assert DEFAULT_LOW_LEVEL_CKPT_OFFICIAL.endswith(
        os.path.join("velocity_flat_official", "model_walk.pt"))
    assert DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2.endswith(os.path.join(
        "velocity_flat_official", "cyberdog2", "model_walk.pt"))
    assert default_low_level_ckpt_official("go2") == \
        DEFAULT_LOW_LEVEL_CKPT_OFFICIAL
    assert default_low_level_ckpt_official("cyberdog2") == \
        DEFAULT_LOW_LEVEL_CKPT_OFFICIAL_CYBERDOG2
    with pytest.raises(ValueError):
        default_low_level_ckpt_official("go1")
    # legacy 面不变
    assert DEFAULT_LOW_LEVEL_CKPT.endswith("model_2999.pt")


def test_train_eval_robot_dispatch_markers():
    """train/eval --robot 分派（task map / 低层 ckpt 分派 / 关节序契约）。"""
    for rel in ("hierarchical_kick/scripts/train.py",
                "hierarchical_kick/scripts/eval_kick.py"):
        src = _read(rel)
        ast.parse(src)
        assert 'choices=["go2", "cyberdog2"]' in src
        assert "default_low_level_ckpt_official(args_cli.robot)" in src
        assert "CYBERDOG2_JOINT_ORDER_IL" in src
        assert "KICK_CMD_CYBERDOG2_TASK_ID" in src
        assert "_JOINT_ORDER_BY_ROBOT" in src
