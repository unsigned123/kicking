"""robots 注册表面 + cyberdog2 velocity 链单测（离线面：常量/字符串/源码标记）。

判据：
- 注册表分派：ROBOT_PROFILE_NAMES 面、Unknown fail-fast、分派单源
  （velocity cfg / train / probe 同源消费）；
- cyberdog2 本体常量（xiaomi 成功快照逐项）：12 关节（IL/SDK 集合相等）、
  足端 4、默认站姿全覆盖、base body = body；
- 包内资产面：URDF 在包 + 相对引用 meshes 全在场；
- velocity cfg cyberdog2 链源码标记（isaaclab 面离线不可 import——AST/
  标记面，学 test_sim_register 范式）：通用基座继承、rc1 三修复、官方
  四覆写、cyberdog2 body names、快照命令 ranges、undesired_contacts=None；
- go2 零变化守卫：go2 类链路源码仍在（注册表路由 + 官方覆写序列）。
"""
from __future__ import annotations

import ast
import os
import re

import pytest

import hierarchical_kick.tasks as reg
from hierarchical_kick.robots import (
    CYBERDOG2_DEFAULT_Q,
    CYBERDOG2_FOOT_BODIES,
    CYBERDOG2_JOINT_ORDER_IL,
    CYBERDOG2_JOINT_ORDER_SDK,
    ROBOT_PROFILE_NAMES,
    make_robot_profile,
)
from hierarchical_kick.robots.cyberdog2 import CYBERDOG2_URDF_PATH

_PKG = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(reg.__file__))))


def _read(rel: str) -> str:
    return open(os.path.join(_PKG, rel), encoding="utf-8").read()


def _parse(rel: str):
    return ast.parse(_read(rel))


# --------------------------------------------------------------------------- #
# 注册表分派面
# --------------------------------------------------------------------------- #
def test_registry_names_surface():
    """注册名面 = {go2, cyberdog2}（train --robot choices 同源）。"""
    assert ROBOT_PROFILE_NAMES == ("cyberdog2", "go2")


def test_registry_unknown_robot_fail_fast():
    """Unknown name fail-fast（ValueError，报注册名集合）。"""
    with pytest.raises(ValueError, match="go2"):
        make_robot_profile("unitree_go2")


def test_cyberdog2_joint_constants():
    """12 关节 × {abad,hip,knee}×{FL,FR,RL,RR}；IL/SDK 集合相等（仅序差）。"""
    legs = ("FL", "FR", "RL", "RR")
    parts = ("abad", "hip", "knee")
    expect = {f"{l}_{p}_joint" for l in legs for p in parts}
    assert len(CYBERDOG2_JOINT_ORDER_IL) == 12
    assert set(CYBERDOG2_JOINT_ORDER_IL) == expect
    assert len(CYBERDOG2_JOINT_ORDER_SDK) == 12
    assert set(CYBERDOG2_JOINT_ORDER_SDK) == expect
    # SDK 序 = FR-first（xiaomi CYBERDOG2_JOINT_SDK_NAMES 契约）
    assert CYBERDOG2_JOINT_ORDER_SDK[:3] == [
        "FR_abad_joint", "FR_hip_joint", "FR_knee_joint"]
    # IL 实测序 = probed：BFS 深度分层（abad→hip→knee）× 层内 URDF 声明序
    # （FR,FL,RR,RL——首训 env build 实测回填，非字母序）
    assert CYBERDOG2_JOINT_ORDER_IL[:4] == [
        "FR_abad_joint", "FL_abad_joint", "RR_abad_joint", "RL_abad_joint"]


def test_cyberdog2_body_constants():
    """足端 4（FL,FR,RL,RR × foot）；base body = body（xiaomi 快照语义）。"""
    assert CYBERDOG2_FOOT_BODIES == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert set(CYBERDOG2_DEFAULT_Q) == set(CYBERDOG2_JOINT_ORDER_IL)
    # 默认站姿关键值（快照 init_state：abad ±0.10 / hip F0.80 R1.00 / knee −1.5）
    assert CYBERDOG2_DEFAULT_Q["FL_abad_joint"] == 0.10
    assert CYBERDOG2_DEFAULT_Q["FR_abad_joint"] == -0.10
    assert CYBERDOG2_DEFAULT_Q["FL_hip_joint"] == 0.80
    assert CYBERDOG2_DEFAULT_Q["RR_hip_joint"] == 1.00
    assert CYBERDOG2_DEFAULT_Q["FL_knee_joint"] == -1.50


def test_cyberdog2_urdf_asset_in_package():
    """URDF 在包内 + 相对引用 ../meshes/*.dae 全在场（零外部依赖）。"""
    assert os.path.isfile(CYBERDOG2_URDF_PATH)
    urdf = open(CYBERDOG2_URDF_PATH, encoding="utf-8").read()
    mesh_refs = set(re.findall(r'filename="(\.\./meshes/[^"]+)"', urdf))
    assert mesh_refs, "URDF 应相对引用 ../meshes"
    for ref in mesh_refs:
        assert os.path.isfile(os.path.join(
            os.path.dirname(CYBERDOG2_URDF_PATH), ref)), f"缺 mesh: {ref}"


def test_velocity_cfg_source_markers():
    """velocity_official_cfg cyberdog2 链落地标记（AST 可解析 + 标记面）。"""
    src = _read("hierarchical_kick/sim/velocity_official_cfg.py")
    _parse("hierarchical_kick/sim/velocity_official_cfg.py")
    # 通用基座继承（cyberdog2 链不继承 go2-flat）
    assert "LocomotionVelocityRoughEnvCfg" in src
    assert "class VelocityOfficialCyberDog2FlatEnvCfg(" \
           "LocomotionVelocityRoughEnvCfg)" in src
    # rc1 三修复（两链同款）
    assert src.count("self.sim.physics = PhysxAutoCfg(") == 2
    assert src.count("self.events.base_com = self.events.base_com.default") == 2
    assert src.count("make_robot_profile(") == 2
    # 官方口径四覆写（两链同款）
    assert src.count(
        "self.observations.policy.joint_vel.scale = 1.0") == 2
    assert src.count("self.actions.joint_pos.scale = 0.5") == 2
    assert src.count(
        "self.commands.base_velocity.heading_command = False") == 2
    # cyberdog2 body names（快照语义）+ flat 化 + contact 全 body 匹配
    assert 'body_names = "body"' in src
    assert 'body_names = ".*_foot"' in src
    assert "self.rewards.undesired_contacts = None" in src
    assert '"{ENV_REGEX_NS}/Robot/.*"' in src
    # 快照命令 ranges（xiaomi vf_500_benchmark 本体参数）
    assert "(-1.6, 1.6)" in src and "(-0.55, 0.55)" in src \
        and "(-2.5, 2.5)" in src
    # 注册表工厂（Unknown fail-fast 面）
    assert 'def VelocityOfficialFlatEnvCfg(robot: str = "go2")' in src
    # 抗塌陷塑形（round-2 非契约面：survival 正信号/抖动加罚/温和 reset）
    assert "velocity_mdp.is_alive" in src
    assert "self.rewards.action_rate_l2.weight = -0.10" in src
    assert "(0.75, 1.25)" in src


def test_tasks_cyberdog2_registration_strings():
    """cyberdog2 任务 id / entry point 字符串契约冻结值。"""
    assert reg.HK_VELOCITY_CYBERDOG2_TASK_ID == "HK-Velocity-Flat-CyberDog2-v0"
    assert reg.HK_VELOCITY_CYBERDOG2_ENV_CFG_ENTRY_POINT == (
        "hierarchical_kick.sim.velocity_official_cfg:"
        "VelocityOfficialCyberDog2FlatEnvCfg")
    assert reg.HK_VELOCITY_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT == (
        "hierarchical_kick.agents_velocity:"
        "HKVelocityOfficialCyberDog2FlatPPORunnerCfg")
    for point in (reg.HK_VELOCITY_CYBERDOG2_ENV_CFG_ENTRY_POINT,
                  reg.HK_VELOCITY_CYBERDOG2_RSL_RL_CFG_ENTRY_POINT):
        module_path, attr = point.split(":", 1)
        assert module_path and attr and "." in module_path


def test_train_velocity_robot_dispatch_markers():
    """train_velocity --robot 分派面（choices 与注册表同源、task 守卫）。"""
    src = _read("hierarchical_kick/scripts/train_velocity.py")
    _parse("hierarchical_kick/scripts/train_velocity.py")
    assert 'default="go2"' in src
    assert 'choices=["go2", "cyberdog2"]' in src
    assert "HK_VELOCITY_CYBERDOG2_TASK_ID" in src
    assert "HKVelocityOfficialCyberDog2FlatPPORunnerCfg" in src
    assert "CYBERDOG2_JOINT_ORDER_IL" in src
    # go2 缺省路径行为守卫：分派表 go2 → 历史任务 id
    assert '"go2": HK_VELOCITY_TASK_ID' in src


def test_probe_walk_robot_dispatch_markers():
    """probe_velocity_walk --robot 分派面 + 行走门三判据阈值在案。"""
    src = _read("hierarchical_kick/scripts/probe_velocity_walk.py")
    _parse("hierarchical_kick/scripts/probe_velocity_walk.py")
    assert 'choices=["go2", "cyberdog2"]' in src
    assert '"go2": ("FL_thigh_joint", "RR_thigh_joint")' in src
    assert '"cyberdog2": ("FL_hip_joint", "RR_hip_joint")' in src
    assert "0.25" in src and "0.2" in src  # v_body 门 / 球位移门


def test_agents_velocity_cyberdog2_runner():
    """cyberdog2 runner = go2 flat 同参换名（超参/policy 结构零漂移）。"""
    src = _read("hierarchical_kick/agents_velocity.py")
    _parse("hierarchical_kick/agents_velocity.py")
    assert ("class HKVelocityOfficialCyberDog2FlatPPORunnerCfg(\n"
            "        HKVelocityOfficialGo2FlatPPORunnerCfg)" in src)
    assert 'self.experiment_name = "hk_velocity_flat_cyberdog2"' in src


# --------------------------------------------------------------------------- #
# profile 构造（isaaclab 面在岗时才跑；离线 skip——本地面无 isaaclab）
# --------------------------------------------------------------------------- #
def test_profile_construction_dispatch():
    """注册表分派构造两本体 profile（name/joints/default_q/资产字段）。"""
    pytest.importorskip("isaaclab")
    go2 = make_robot_profile("go2")
    cyber = make_robot_profile("cyberdog2")
    assert go2.name == "go2" and cyber.name == "cyberdog2"
    assert cyber.num_joints == 12 and cyber.num_feet == 4
    assert set(cyber.default_joint_pos) == set(CYBERDOG2_JOINT_ORDER_IL)
    spawn = cyber.articulation_cfg.spawn
    assert getattr(spawn, "fix_base") is False
    assert getattr(spawn, "merge_fixed_joints") is False
    assert getattr(spawn, "activate_contact_sensors") is True
    assert str(CYBERDOG2_URDF_PATH) == spawn.asset_path
    legs = cyber.articulation_cfg.actuators["legs"]
    assert (legs.stiffness, legs.damping) == (25.0, 0.5)
    assert legs.saturation_effort == 12.0
    assert cyber.articulation_cfg.init_state.pos[2] == 0.40
