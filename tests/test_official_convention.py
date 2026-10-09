"""官方 velocity-flat 口径对齐（2026-10-09）——离线判据。

- contracts 官方常量（OFFICIAL_* 偏移 / 均匀 0.5 动作 scale）。
- core/weights 缺省路径分派（official → velocity_flat_official、legacy →
  model_2999；零 isaaclab 可 import 面）。
- core/low_level_actor 官方 ckpt 结构校验（sd 反推 hidden dims 对账官方
  agents cfg [128,128,128]；fake rsl_rl 桩 + 合成 sd 往返）。
- 注册/源码面：HK-Velocity-Flat-Unitree-Go2-v0 注册常量、
  velocity_official_cfg 上游口径覆写锚（joint_vel scale 1.0 / 单帧 /
  动作 0.5）、agents_velocity 官方 agents cfg 逐字锚、train_velocity 守卫
  与 runner 面、train.py 约定开关注入。

离线（无 isaaclab/gymnasium）：sim 面（velocity_official_cfg/
frozen_loco_action/ball_env_cfg）→ 源码/AST 面；core 面 → 直测。
"""
from __future__ import annotations

import ast
import importlib.util
import os

import pytest
import torch

from hierarchical_kick.core import contracts as C

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
_PKG_PKG = os.path.join(_PKG_ROOT, "hierarchical_kick")
_REGISTER_PATH = os.path.join(_PKG_PKG, "tasks", "__init__.py")
_VELOCITY_CFG_PATH = os.path.join(_PKG_PKG, "sim", "velocity_official_cfg.py")
_AGENTS_V_PATH = os.path.join(_PKG_PKG, "agents_velocity.py")
_TRAIN_V_PATH = os.path.join(_PKG_PKG, "scripts", "train_velocity.py")
_TRAIN_PATH = os.path.join(_PKG_PKG, "scripts", "train.py")
_FROZEN_MATH_PATH = os.path.join(
    _PKG_PKG, "sim", "mdp", "frozen_loco_math.py")


def _source(path: str) -> str:
    return open(path, encoding="utf-8").read()


def _load_frozen_math():
    """按路径加载 frozen_loco_math（父包 mdp/__init__ 拉 isaaclab）。"""
    spec = importlib.util.spec_from_file_location(
        "_hkick_official_math", _FROZEN_MATH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _low_level_inputs(n: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        joint_pos=torch.randn(n, 12, generator=g),
        default_joint_pos=torch.randn(n, 12, generator=g),
        joint_vel=torch.randn(n, 12, generator=g),
        base_ang_vel=torch.randn(n, 3, generator=g),
        projected_gravity=torch.randn(n, 3, generator=g),
        last_action=torch.randn(n, 12, generator=g),
        base_lin_vel=torch.randn(n, 3, generator=g),
        command=torch.randn(n, 3, generator=g),
    )


# --------------------------------------------------------------------------- #
# 1. contracts 官方常量
# --------------------------------------------------------------------------- #
def test_official_contracts_constants():
    """官方口径常量锚：均匀 0.5 动作 scale + 48 维偏移表（上游顺序）。"""
    assert C.OFFICIAL_ACTION_SCALE == 0.5
    assert (C.OFFICIAL_OBS_LIN_VEL, C.OFFICIAL_OBS_ANG_VEL,
            C.OFFICIAL_OBS_GRAVITY, C.OFFICIAL_OBS_CMD,
            C.OFFICIAL_OBS_JOINT_POS, C.OFFICIAL_OBS_JOINT_VEL,
            C.OFFICIAL_OBS_LAST_ACTION) == (0, 3, 6, 9, 12, 24, 36)
    assert C.OFFICIAL_OBS_LAST_ACTION + C.ACTION_DIM == C.OBS_DIM == 48


# --------------------------------------------------------------------------- #
# 2. core/weights 缺省路径分派
# --------------------------------------------------------------------------- #
def test_weights_default_paths_by_convention():
    """official/legacy 缺省 ckpt 路径单源（core/weights；零 isaaclab）。"""
    from hierarchical_kick.core.weights import (
        DEFAULT_LOW_LEVEL_CKPT, DEFAULT_LOW_LEVEL_CKPT_OFFICIAL)
    assert DEFAULT_LOW_LEVEL_CKPT.endswith(
        os.path.join("weights", "model_2999.pt"))
    assert DEFAULT_LOW_LEVEL_CKPT_OFFICIAL.endswith(os.path.join(
        "weights", "velocity_flat_official", "model_walk.pt"))
    # 路径锚定包内 weights/（相对包根；包内资产移动即 fail-fast 面）
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(
        __import__("hierarchical_kick.core.weights",
                   fromlist=["x"]).__file__)))
    assert DEFAULT_LOW_LEVEL_CKPT.startswith(pkg_root)
    assert DEFAULT_LOW_LEVEL_CKPT_OFFICIAL.startswith(pkg_root)


# --------------------------------------------------------------------------- #
# 3. 官方 assemble：与 velocity 训练面 policy obs 逐位同构（数值面）
# --------------------------------------------------------------------------- #
def test_official_assemble_matches_training_obs_semantics():
    """官方 48 = [linvel, ang_vel, gravity, cmd, q−q0, qd(raw), last_act]。

    手工逐段拼装期望向量（不读模块实现偏移）——对训练面 policy obs 语义
    （上游 velocity policy 组顺序 + joint_vel 无 scale）逐位对账。
    """
    m = _load_frozen_math()
    inp = _low_level_inputs(n=4, seed=7)
    obs = m.assemble_low_level_obs(**inp)
    expect = torch.cat([
        inp["base_lin_vel"],
        inp["base_ang_vel"],
        inp["projected_gravity"],
        inp["command"],
        inp["joint_pos"] - inp["default_joint_pos"],
        inp["joint_vel"],                       # raw rad/s（无 scale）
        inp["last_action"],
    ], dim=-1)
    assert obs.shape == (4, 48)
    assert torch.equal(obs, expect)


def test_official_assemble_out_buffer_and_shape_failfast():
    """out= 复用 buffer + 形状漂移 fail-fast（列名指路）。"""
    m = _load_frozen_math()
    inp = _low_level_inputs(n=2, seed=3)
    out = torch.full((2, 48), float("nan"))
    ret = m.assemble_low_level_obs(**inp, out=out)
    assert ret is out and torch.isfinite(out).all()
    bad = dict(inp)
    bad["joint_vel"] = torch.randn(2, 10)
    with pytest.raises(ValueError, match="joint_vel"):
        m.assemble_low_level_obs(**bad)
    bad = dict(inp)
    bad["base_lin_vel"] = torch.randn(2, 5)
    with pytest.raises(ValueError, match="base_lin_vel"):
        m.assemble_low_level_obs(**bad)


def test_official_action_semantics_uniform_half():
    """官方动作语义 = default + 0.5⊙action（均匀标量；非逐关节表）。"""
    m = _load_frozen_math()
    default = torch.randn(3, 12, generator=torch.Generator().manual_seed(1))
    action = torch.randn(3, 12, generator=torch.Generator().manual_seed(2))
    scale_vec = torch.full((3, 12), C.OFFICIAL_ACTION_SCALE)
    tgt = m.joint_target_from_low_action(default, scale_vec, action)
    assert torch.allclose(tgt, default + 0.5 * action)


# --------------------------------------------------------------------------- #
# 4. core/low_level_actor：官方 ckpt 结构校验（fake 桩 + 合成 sd）
# --------------------------------------------------------------------------- #
_DIST_CFG = {"class_name": "rsl_rl.modules:GaussianDistribution",
             "init_std": C.SIGMA_SEM}


def _build_actor(obs_dim: int = 48, hidden=(128, 128, 128)):
    from hierarchical_kick.core.low_level_actor import UpperDirectActor
    g = torch.Generator().manual_seed(0)
    obs = {"policy": torch.randn(2, obs_dim, generator=g)}
    return UpperDirectActor(
        obs, {"actor": ["policy"]}, "actor", C.ACTION_DIM,
        hidden_dims=list(hidden), activation="elu", obs_normalization=False,
        distribution_cfg=dict(_DIST_CFG))


def _official_style_sd(hidden=(128, 128, 128), obs_dim: int = 48):
    """合成官方存面 actor sd（rsl_rl 5.1.0 分离面键形 mlp.<2i>.weight）。"""
    actor = _build_actor(obs_dim, hidden)
    return {k: v.clone() for k, v in actor.state_dict().items()}


def test_infer_and_validate_official_structure():
    from hierarchical_kick.core.low_level_actor import (
        OFFICIAL_ACTOR_HIDDEN_DIMS, infer_actor_hidden_dims,
        validate_official_actor_structure)
    sd = _official_style_sd()
    hidden = infer_actor_hidden_dims(sd)
    assert hidden == [128, 128, 128] == OFFICIAL_ACTOR_HIDDEN_DIMS
    assert validate_official_actor_structure(sd) == [128, 128, 128]
    # 结构漂移 fail-fast（非官方 hidden）
    sd_bad = _official_style_sd(hidden=(64, 64))
    with pytest.raises(ValueError, match="结构漂移"):
        validate_official_actor_structure(sd_bad)
    # 末层输出漂移 fail-fast（非 12 维动作面）
    sd_out = _official_style_sd()
    sd_out["mlp.6.weight"] = torch.zeros(3, 128)
    sd_out["mlp.6.bias"] = torch.zeros(3)
    with pytest.raises(ValueError, match="动作维"):
        infer_actor_hidden_dims(sd_out)


def test_resolve_actor_state_dict_key_face():
    """ckpt 键面兼容：分离存面 actor_state_dict / 合一面 model_state_dict /
    未知键面 fail-fast。"""
    from hierarchical_kick.core.low_level_actor import resolve_actor_state_dict
    sd = _official_style_sd()
    full, got = resolve_actor_state_dict({"actor_state_dict": sd, "iter": 300})
    assert full["iter"] == 300 and got is sd
    _, got2 = resolve_actor_state_dict({"model_state_dict": sd})
    assert got2 is sd
    with pytest.raises(KeyError, match="键面不识别"):
        resolve_actor_state_dict({"unexpected": sd})


def test_official_ckpt_roundtrip_via_upper_direct_actor():
    """合成官方 sd → UpperDirectActor[128,128,128] strict 装载往返（键/形
    全对齐——官方训练产物同面）。"""
    from hierarchical_kick.core.low_level_actor import (
        build_upper_direct_actor, resolve_actor_state_dict,
        validate_official_actor_structure)
    sd = _official_style_sd()
    validate_official_actor_structure(sd)
    dummy = {"policy": torch.zeros(2, 48)}
    actor = build_upper_direct_actor(
        dummy, actor_cfg=None, hidden_dims=[128, 128, 128],
        activation="elu", obs_normalization=False,
        distribution_cfg=dict(_DIST_CFG), sigma_sem=C.SIGMA_SEM,
        obs_dim_contract=C.OBS_DIM)
    _, got = resolve_actor_state_dict({"actor_state_dict": sd})
    actor.load_state_dict(got, strict=True)
    with torch.no_grad():
        out = actor({"policy": torch.randn(2, 48)}, stochastic_output=False)
    assert out.shape == (2, C.ACTION_DIM)


# --------------------------------------------------------------------------- #
# 5. 注册字符串面 + 源码锚（sim 面 isaaclab 离线不可 import → 文本面）
# --------------------------------------------------------------------------- #
def test_register_velocity_task_constants():
    import hierarchical_kick.tasks as reg
    assert reg.HK_VELOCITY_TASK_ID == "HK-Velocity-Flat-Unitree-Go2-v0"
    assert reg.HK_VELOCITY_TASK_ENTRY_POINT == "isaaclab.envs:ManagerBasedRLEnv"
    assert reg.HK_VELOCITY_ENV_CFG_ENTRY_POINT.endswith(
        "velocity_official_cfg:VelocityOfficialGo2FlatEnvCfg")
    assert reg.HK_VELOCITY_RSL_RL_CFG_ENTRY_POINT.endswith(
        "agents_velocity:HKVelocityOfficialGo2FlatPPORunnerCfg")
    src = _source(_REGISTER_PATH)
    ast.parse(src)
    assert 'id=HK_VELOCITY_TASK_ID' in src
    assert '"HK_VELOCITY_TASK_ID"' in src
    # kick cmd 面常量保持冻结（共存不漂移）
    assert reg.KICK_CMD_TASK_ID == "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"


def test_velocity_official_cfg_source_anchors():
    """官方口径覆写锚：joint_vel scale 1.0 / history_length None（单帧）/
    动作 0.5；基类 = fork UnitreeGo2FlatEnvCfg（非本包 vendor 链）。"""
    src = _source(_VELOCITY_CFG_PATH)
    ast.parse(src)
    assert "self.observations.policy.joint_vel.scale = 1.0" in src
    assert "self.observations.policy.history_length = None" in src
    assert "self.actions.joint_pos.scale = 0.5" in src
    assert ("from isaaclab_tasks.core.velocity.config.go2."
            "flat_env_cfg import") in src
    assert "class VelocityOfficialGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):" in src
    # 单帧语义注释在案（ObsGroup None vs 1 的语义辨析防回退）
    assert "单帧" in src and "CircularBuffer" in src


def test_agents_velocity_source_anchors():
    """官方 go2 flat agents cfg 逐字锚：300 it / [128,128,128] / lr 1e-3 /
    entropy 0.01 / init_noise_std 1.0 / save_interval 50。"""
    src = _source(_AGENTS_V_PATH)
    ast.parse(src)
    assert "class HKVelocityOfficialGo2FlatPPORunnerCfg(" \
        "HKVelocityOfficialGo2RoughPPORunnerCfg):" in src
    assert "self.max_iterations = 300" in src
    assert "self.policy.actor_hidden_dims = [128, 128, 128]" in src
    assert "self.policy.critic_hidden_dims = [128, 128, 128]" in src
    assert "actor_hidden_dims=[512, 256, 128]" in src      # rough 基类官方原文
    assert "init_noise_std=1.0" in src
    assert "learning_rate=1.0e-3" in src
    assert "entropy_coef=0.01" in src
    assert "num_steps_per_env = 24" in src
    assert "save_interval = 50" in src
    assert 'experiment_name = "hk_velocity_flat_official"' in src


def test_train_velocity_source_anchors():
    """train_velocity 守卫/runner/关节序打印/obs 48 契约面。"""
    src = _source(_TRAIN_V_PATH)
    ast.parse(src)
    assert "HK_VELOCITY_TASK_ID" in src
    assert "仅支持任务 id" in src
    guard_pos = src.index("仅支持任务 id")
    launcher_pos = src.index("app_launcher = AppLauncher")
    assert guard_pos < launcher_pos
    # 官方口径装配自检（覆写漂移 fail-fast）
    assert "joint_vel.scale" in src and "history_length" in src
    assert "官方口径装配漂移" in src
    # 关节序打印（与 kick env 比对锚）+ obs 48 契约
    assert 'f"[hkick][velocity] velocity env joint order' in src
    assert "GO2_JOINT_ORDER_IL" in src
    assert "!= 48" in src
    # 官方 runner 面
    assert "from rsl_rl.runners import OnPolicyRunner" in src
    assert "OnPolicyRunner(" in src
    assert "handle_deprecated_rsl_rl_cfg" in src
    assert "HKVelocityOfficialGo2FlatPPORunnerCfg" in src


def test_train_convention_flag_anchors():
    """kick train.py 约定开关：--low_level_convention {official,legacy} 缺省
    official + 按约定分派缺省 ckpt + env cfg 注入。"""
    src = _source(_TRAIN_PATH)
    ast.parse(src)
    assert '"--low_level_convention"' in src
    assert 'choices=["official", "legacy"]' in src
    assert 'default="official"' in src
    assert "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL" in src
    assert "env_cfg.actions.joint_pos.low_level_convention" in src
    # legacy 缺省锚保持（旧 ckpt 兼容面）
    assert "model_2999.pt" in src


def test_frozen_action_official_dispatch_anchors():
    """frozen_loco_action 官方口径锚：均匀 0.5（contracts 单源）/ legacy
    逐关节 dict 分支 / 官方结构校验 / legacy 布局分派。"""
    path = os.path.join(_PKG_PKG, "sim", "mdp", "frozen_loco_action.py")
    src = _source(path)
    ast.parse(src)
    assert "OFFICIAL_ACTION_SCALE" in src
    assert "validate_official_actor_structure" in src
    assert "assemble_low_level_obs_legacy" in src
    assert "official" in src and "legacy" in src
    # 官方 ckpt 缺省路径常量（core/weights 单源 re-export）
    assert "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL" in src


def test_ball_env_cfg_cmd_face_convention_note():
    """cmd 面 cfg：ckpt 路径留空 = 约定缺省（不再钉死 model_2999）。"""
    path = os.path.join(_PKG_PKG, "sim", "ball_env_cfg.py")
    src = _source(path)
    ast.parse(src)
    # Go2KickFlatEnvCfgCommand 类体内不再显式传 DEFAULT_LOW_LEVEL_CKPT
    tree = ast.parse(src)
    body_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and \
                node.name == "Go2KickFlatEnvCfgCommand":
            body_src = ast.get_source_segment(src, node)
    assert body_src is not None
    assert "DEFAULT_LOW_LEVEL_CKPT" not in body_src
    assert "go2_action_scale_table()" in body_src  # legacy 面占位保持


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
