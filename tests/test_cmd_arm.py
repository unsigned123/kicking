"""U-Kick arm D「真分层」对照 — 离线判据（docs/UKICK_BALL_ENV_DESIGN.md §0
谱系扩展 + 任务书 B 部分）。

- frozen_loco_math 纯函数（按路径加载，ball_mdp 同款降级）：低层 48 维 obs
  组装逐通道 == contracts.py §1.1 适配表（offset/scale 单源）；关节目标公式
  = default + scale_vec ⊙ 低层输出；形状 fail-fast。
- UpperCmdActor 纯逻辑（fake rsl_rl 桩）：3 维 Gaussian、σ init = σ_sem
  标量填充（命令空间无 scale_vec）、obs 契约 55 守卫、state_dict 键面
  （零 decoder/z/scale_vec 面）。
- ActionTerm 源码断言（isaaclab 面离线不可 import）：路径 fail-fast 在位、
  命令写入/obs 组装/冻结 forward/关节目标四步序、last_action reset 清零、
  注解引号纪律、无时间抽象说明。
- cfg/注册/train.py/agents 源码 + AST 面：Go2KickFlatEnvCfgCommand 恢复
  命令面 + 动作项替换 + obs 55 不动（奖励/终止零改动）；register 常量；
  UpperKickCmdPPORunnerCfg 最小子类公平性；run_upper --actor command 双向
  守卫 + --low_level_ckpt 路径注入（md5/provenance 面已随 v1.0 剥离移除）。
- 公平性：arm D runner cfg 与三 kick runner cfg 除 actor/experiment_name
  外逐字等；UpperCmdActorCfg 与 UpperDirectActorCfg 共享字段逐字同 AST。

离线（无 isaaclab）：ball_env_cfg/frozen_loco_action 不可 import → 源码/AST
面；core 模块（cmd_actor/frozen_loco_math）经 fakes/rsl_rl 桩或按路径加载
直测。
"""
from __future__ import annotations

import ast
import importlib.util
import os

import pytest
import torch

from hierarchical_kick.core.cmd_actor import (
    CMD_ACTION_DIM, UPPER_CMD_ACTOR_CLASS_PATH, UpperCmdActor)
from hierarchical_kick.core.contracts import (
    OFF_BASE_ANG_VEL, OFF_JOINT_POS, OFF_JOINT_VEL, OFF_LAST_ACTION,
    OFF_PROJ_GRAVITY, PROPRIO_DIM, SCALE_BASE_ANG_VEL, SCALE_JOINT_VEL,
    SIGMA_SEM)
from hierarchical_kick.core.scale_vec import go2_action_scale_vec

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
_AGENTS_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "agents.py")
_RUN_UPPER_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "scripts",
                               "train.py")
_BALL_ENV_CFG_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "sim",
                                  "ball_env_cfg.py")
_REGISTER_PATH = os.path.join(_PKG_ROOT, "hierarchical_kick", "tasks",
                              "__init__.py")
_FROZEN_ACTION_PATH = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "sim", "mdp", "frozen_loco_action.py")
_FROZEN_MATH_PATH = os.path.join(
    _PKG_ROOT, "hierarchical_kick", "sim", "mdp", "frozen_loco_math.py")

_DIST_CFG = {"class_name": "rsl_rl.modules:GaussianDistribution",
             "init_std": SIGMA_SEM}


def _load_frozen_math():
    """按路径加载 frozen_loco_math（父包 mdp/__init__ 拉 isaaclab；
    test_upper_kick_env 同款降级）。"""
    spec = importlib.util.spec_from_file_location(
        "_hkick_armd_math", _FROZEN_MATH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source(path: str) -> str:
    return open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# 1. frozen_loco_math：低层 48 维 obs 组装（对 contracts 适配表逐位） + 公式
# --------------------------------------------------------------------------- #

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


def test_assemble_low_level_obs_channel_order_matches_contracts():
    """48 维组装逐通道 == 官方 velocity-flat 布局（contracts OFFICIAL_* 偏移
    单源；joint_vel/base_ang_vel raw 无缩放；2026-10-09 官方口径缺省）。"""
    from hierarchical_kick.core.contracts import (
        OFFICIAL_OBS_ANG_VEL, OFFICIAL_OBS_CMD, OFFICIAL_OBS_GRAVITY,
        OFFICIAL_OBS_JOINT_POS, OFFICIAL_OBS_JOINT_VEL,
        OFFICIAL_OBS_LAST_ACTION, OFFICIAL_OBS_LIN_VEL)
    m = _load_frozen_math()
    n = 3
    inp = _low_level_inputs(n, seed=1)
    obs = m.assemble_low_level_obs(**inp)
    assert obs.shape == (n, 48)
    assert torch.equal(obs[:, OFFICIAL_OBS_LIN_VEL:OFFICIAL_OBS_ANG_VEL],
                       inp["base_lin_vel"])
    assert torch.equal(obs[:, OFFICIAL_OBS_ANG_VEL:OFFICIAL_OBS_GRAVITY],
                       inp["base_ang_vel"])
    assert torch.equal(obs[:, OFFICIAL_OBS_GRAVITY:OFFICIAL_OBS_CMD],
                       inp["projected_gravity"])
    assert torch.equal(obs[:, OFFICIAL_OBS_CMD:OFFICIAL_OBS_JOINT_POS],
                       inp["command"])
    assert torch.equal(obs[:, OFFICIAL_OBS_JOINT_POS:OFFICIAL_OBS_JOINT_VEL],
                       inp["joint_pos"] - inp["default_joint_pos"])
    assert torch.equal(obs[:, OFFICIAL_OBS_JOINT_VEL:OFFICIAL_OBS_LAST_ACTION],
                       inp["joint_vel"])
    assert torch.equal(obs[:, OFFICIAL_OBS_LAST_ACTION:
                           OFFICIAL_OBS_LAST_ACTION + 12],
                       inp["last_action"])
    # 偏移常量锚（官方顺序：linvel→ang_vel→gravity→cmd→q→qd→last_action）
    assert (OFFICIAL_OBS_LIN_VEL, OFFICIAL_OBS_ANG_VEL, OFFICIAL_OBS_GRAVITY,
            OFFICIAL_OBS_CMD, OFFICIAL_OBS_JOINT_POS, OFFICIAL_OBS_JOINT_VEL,
            OFFICIAL_OBS_LAST_ACTION) == (0, 3, 6, 9, 12, 24, 36)
    # 推理面：无噪声注入（纯函数无随机源——同输入两次组装逐位相等）
    obs2 = m.assemble_low_level_obs(**inp)
    assert torch.equal(obs, obs2)


def test_assemble_low_level_obs_legacy_branch_layout():
    """legacy 分支（旧 U4b B 臂 48 布局）保持逐位不变（joint_vel ×0.05 /
    ang_vel ×0.2 适配表；proprio42 前缀 + linvel + cmd）。"""
    from hierarchical_kick.core.contracts import (
        OFF_BASE_ANG_VEL, OFF_JOINT_POS, OFF_JOINT_VEL, OFF_LAST_ACTION,
        OFF_PROJ_GRAVITY, SCALE_BASE_ANG_VEL, SCALE_JOINT_VEL)
    m = _load_frozen_math()
    n = 3
    inp = _low_level_inputs(n, seed=1)
    obs = m.assemble_low_level_obs_legacy(**inp)
    assert obs.shape == (n, 48)
    assert torch.equal(obs[:, OFF_JOINT_POS:OFF_JOINT_VEL],
                       inp["joint_pos"] - inp["default_joint_pos"])
    assert torch.allclose(obs[:, OFF_JOINT_VEL:OFF_BASE_ANG_VEL],
                          inp["joint_vel"] * SCALE_JOINT_VEL)
    assert torch.allclose(obs[:, OFF_BASE_ANG_VEL:OFF_PROJ_GRAVITY],
                          inp["base_ang_vel"] * SCALE_BASE_ANG_VEL)
    assert torch.equal(obs[:, OFF_PROJ_GRAVITY:OFF_LAST_ACTION],
                       inp["projected_gravity"])
    assert torch.equal(obs[:, OFF_LAST_ACTION:OFF_LAST_ACTION + 12],
                       inp["last_action"])
    assert torch.equal(obs[:, 42:45], inp["base_lin_vel"])
    assert torch.equal(obs[:, 45:48], inp["command"])
    # 官方缺省布局 ≠ legacy 布局（分派真在案：同输入双分支必不同）
    off = m.assemble_low_level_obs(**inp)
    assert not torch.equal(off, obs)


def test_assemble_low_level_obs_out_buffer_and_shapes():
    """out= 复用 buffer 零分配语义 + 形状漂移 fail-fast（列号指路）。"""
    m = _load_frozen_math()
    n = 2
    inp = _low_level_inputs(n, seed=2)
    out = torch.full((n, 48), float("nan"))
    ret = m.assemble_low_level_obs(**inp, out=out)
    assert ret is out and torch.isfinite(out).all()
    bad = dict(inp)
    bad["command"] = torch.randn(n, 4)   # cmd 维漂移
    with pytest.raises(ValueError, match="command"):
        m.assemble_low_level_obs(**bad)
    bad = dict(inp)
    bad["joint_pos"] = torch.randn(n, 10)  # 关节维漂移
    with pytest.raises(ValueError, match="joint_pos"):
        m.assemble_low_level_obs(**bad)


def test_joint_target_formula_is_joint_position_action_semantics():
    """关节目标 = default + scale_vec ⊙ 低层输出（JointPositionAction
    use_default_offset 语义；scale_vec = core/scale_vec 单源槽位序）。"""
    m = _load_frozen_math()
    sv = go2_action_scale_vec()
    assert sv.shape == (12,)
    tgt = m.joint_target_from_low_action(sv * 0.0 + 0.3, sv, sv * 2.0)
    assert torch.allclose(tgt, 0.3 + sv * sv * 2.0)
    # 形状漂移 fail-fast
    with pytest.raises(ValueError, match="形状不符"):
        m.joint_target_from_low_action(torch.zeros(12), torch.zeros(12),
                                       torch.zeros(11))
    # LOW_LEVEL_OBS_DIM 契约 = contracts.OBS_DIM 同源 48
    from hierarchical_kick.core.contracts import OBS_DIM
    assert m.LOW_LEVEL_OBS_DIM == OBS_DIM == 48


# --------------------------------------------------------------------------- #
# 2. UpperCmdActor 纯逻辑（fake rsl_rl 桩）
# --------------------------------------------------------------------------- #

def _make_cmd_actor(num_envs: int = 4, seed: int = 3, **kw):
    g = torch.Generator().manual_seed(seed)
    obs = {"policy": torch.randn(num_envs, 55, generator=g)}
    defaults = dict(distribution_cfg=dict(_DIST_CFG))
    defaults.update(kw)
    output_dim = defaults.pop("output_dim", CMD_ACTION_DIM)
    actor = UpperCmdActor(obs, {"actor": ["policy"]}, "actor",
                          output_dim, **defaults)
    return actor, obs


def test_cmd_actor_output_and_sigma_sem_fill():
    """3 维 Gaussian 输出；σ init = σ_sem 标量逐维填充（命令空间无 scale_vec）。"""
    actor, obs = _make_cmd_actor()
    with torch.no_grad():
        det = actor(obs, stochastic_output=False)
    assert det.shape == (4, 3)
    assert actor(obs).shape == (4, 3)   # 探索采样
    assert torch.allclose(actor.distribution.std_param,
                          torch.full((3,), SIGMA_SEM))
    # 契约参数化缺省面 = kick 55
    assert actor.obs_dim_contract == 55
    assert actor.mlp[0].weight.shape == (128, 55)
    assert actor.mlp[-1].weight.shape == (3, 128)


def test_cmd_actor_guards():
    """守卫面（direct_actor 同款换 3 维命令面；fail-fast）。"""
    g = torch.Generator().manual_seed(0)
    obs55 = {"policy": torch.randn(4, 55, generator=g)}
    with pytest.raises(ValueError, match="obs_normalization"):
        _make_cmd_actor(obs_normalization=True)
    with pytest.raises(ValueError, match="仅限 actor"):
        UpperCmdActor(
            {"policy": obs55["policy"]},
            {"actor": ["policy"], "critic": ["policy"]}, "critic",
            CMD_ACTION_DIM, distribution_cfg=dict(_DIST_CFG))
    with pytest.raises(ValueError, match="obs_groups"):
        UpperCmdActor({"policy": obs55["policy"],
                       "cmd": torch.randn(4, 3)},
                      {"actor": ["cmd"]}, "actor", CMD_ACTION_DIM,
                      distribution_cfg=dict(_DIST_CFG))
    with pytest.raises(ValueError, match="output_dim"):
        _make_cmd_actor(output_dim=12)
    # obs 48（velocity 面）≠ kick 契约 55 → 拒绝
    obs48 = {"policy": torch.randn(4, 48, generator=g)}
    with pytest.raises(ValueError, match="obs 维契约"):
        UpperCmdActor(obs48, {"actor": ["policy"]}, "actor", CMD_ACTION_DIM,
                      distribution_cfg=dict(_DIST_CFG))


def test_cmd_actor_state_dict_key_face():
    """ckpt 键面承诺：mlp.* + distribution.std_param；零 decoder/z/scale_vec
    面（上层无冻结资产——冻结低层在 env 侧动作 term）。"""
    actor, _ = _make_cmd_actor()
    keys = set(actor.state_dict().keys())
    for k in ("mlp.0.weight", "mlp.6.bias", "distribution.std_param"):
        assert k in keys, k
    assert not any(k.startswith(("decoder", "z_anchor", "z_scale", "z_stats"))
                   for k in keys)
    assert not any("scale_vec" in k for k in keys)


# --------------------------------------------------------------------------- #
# 3. frozen_loco_action 源码断言（isaaclab 面，离线不可 import → 文本面）
# --------------------------------------------------------------------------- #

def test_frozen_action_term_source_contract():
    """ActionTerm 四步序 + 路径 fail-fast + 命令写入 + reset 清零 + 注解纪律。"""
    src = _source(_FROZEN_ACTION_PATH)
    ast.parse(src)
    # 继承 ActionTerm（非 JointAction——action_dim 必须是 3 命令维）
    assert "class FrozenLocoJointPositionAction(ActionTerm):" in src
    # 构造期 fail-fast：路径存在性（md5 锁已随 v1.0 剥离移除）
    assert "low_level_ckpt_path 缺失" in src
    assert "FileNotFoundError" in src
    # apply_actions 四步序（命令写入 → obs 组装 → 冻结 forward → 关节目标）
    assert "command[:, :3] = self._raw_actions" in src
    # 约定分派（official 缺省 / legacy 分支；ctor 期单点 self._assemble）
    assert "self._assemble(" in src
    assert "assemble_low_level_obs if self._convention" in src or \
        "assemble_low_level_obs if" in src
    assert "assemble_low_level_obs_legacy" in src
    assert 'LOW_LEVEL_CONVENTIONS = ("official", "legacy")' in src
    assert "stochastic_output=False" in src
    assert "joint_target_from_low_action(" in src
    assert "set_joint_position_target(" in src
    # 冻结三件套（no_grad/eval/requires_grad False）
    assert "torch.no_grad()" in src and ".eval()" in src
    assert "requires_grad_(False)" in src
    # reset 语义：raw 命令 + 低层 last_action 清零
    assert "self._raw_actions[env_ids] = 0.0" in src
    assert "self._low_last_action[env_ids] = 0.0" in src
    # 类型注解纪律（ball_env_cfg NameError 教训）：env 注解加引号 +
    # TYPE_CHECKING import ManagerBasedRLEnv
    assert "'ManagerBasedRLEnv'" in src
    assert "if TYPE_CHECKING:" in src
    # 高层 obs 换源函数（last_action 通道 = 低层输出，55 维契约保持）
    assert "def low_level_last_action_obs" in src
    # cfg 契约：命令 term 名可配
    assert "command_term_name" in src


def test_frozen_action_no_time_abstraction_note():
    """docstring 钉死 50Hz 逐步命令（无时间抽象/action repeat）+ U33 审计注。"""
    src = _source(_FROZEN_ACTION_PATH)
    assert "50Hz" in src
    assert "无时间抽象" in src
    assert "U33" in src
    assert "action repeat" in src or "5 步 hold" in src


def test_frozen_action_math_single_source_no_param_copy():
    """组装公式复用 frozen_loco_math 单源 + contracts 适配表常量（禁止复制参数）。"""
    src = _source(_FROZEN_ACTION_PATH)
    assert "from .frozen_loco_math import" in src
    assert "assemble_low_level_obs" in src
    # 本文件不得出现字面缩放参数（单源纪律——参数只在 contracts/frozen_loco_math）
    assert "0.05" not in src and "0.2" not in src.split("50Hz")[0].split("def apply")[0]


# --------------------------------------------------------------------------- #
# 4. cfg 面：Go2KickFlatEnvCfgCommand 源码断言
# --------------------------------------------------------------------------- #

def test_command_env_cfg_source_contract():
    """cmd 面：继承训练面 + 恢复命令面（照 Oracle 写法）+ 动作项替换 +
    obs 组替换；奖励/终止/球物理零改动（无 rewards/terminations/scene 赋值）。"""
    src = _source(_BALL_ENV_CFG_PATH)
    ast.parse(src)
    assert "class Go2KickFlatEnvCfgCommand(Go2KickFlatEnvCfg):" in src
    assert "class KickPolicyCfgCmd(KickPolicyCfg):" in src
    # 提取 Go2KickFlatEnvCfgCommand 类体
    tree = ast.parse(src)
    body_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and \
                node.name == "Go2KickFlatEnvCfgCommand":
            body_src = ast.get_source_segment(src, node)
    assert body_src is not None
    # 恢复命令面（oracle 同款 UniformVelocityCommandCfg 挂载）
    assert "self.commands.base_velocity = mdp.UniformVelocityCommandCfg(" in body_src
    # 动作项替换（冻结低层 ActionTerm + scale_vec 单源）
    assert "self.actions.joint_pos = FrozenLocoJointPositionActionCfg(" in body_src
    assert "go2_action_scale_table()" in body_src
    # obs 组替换（actions 项换源，55 维契约保持）
    assert "self.observations.policy = KickPolicyCfgCmd()" in body_src
    # 公平性：类体内**零**奖励/终止/球物理/episode 赋值（继承训练面不动）
    assert "self.rewards." not in body_src
    assert "self.terminations." not in body_src
    assert "self.scene." not in body_src
    assert "episode_length_s" not in body_src
    assert "kick_reward_weights" not in body_src
    # KickPolicyCfgCmd：唯一 diff = actions 项换源；__post_init__ 仅 super 调用
    kick_cmd_src = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "KickPolicyCfgCmd":
            kick_cmd_src = ast.get_source_segment(src, node)
    assert "actions = ObsTerm(func=low_level_last_action_obs)" in kick_cmd_src
    assert "self.rewards" not in kick_cmd_src \
        and "self.terminations" not in kick_cmd_src


def test_register_cmd_task_constants():
    """register 常量面（模式照旧：常量 + try/except gym.register + __all__）。"""
    import hierarchical_kick.tasks as reg
    assert reg.KICK_CMD_TASK_ID == "Isaac-Kick-Flat-Unitree-Go2-Upper-Cmd-v0"
    assert reg.KICK_CMD_ENV_CFG_ENTRY_POINT.endswith(
        "ball_env_cfg:Go2KickFlatEnvCfgCommand")
    assert reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT.endswith(
        "agents:UpperKickCmdPPORunnerCfg")
    src = _source(_REGISTER_PATH)
    ast.parse(src)
    assert 'id=KICK_CMD_TASK_ID' in src
    assert '"KICK_CMD_TASK_ID"' in src   # __all__ 在案
    for point in (reg.KICK_CMD_ENV_CFG_ENTRY_POINT,
                  reg.KICK_CMD_RSL_RL_CFG_ENTRY_POINT):
        module_path, attr = point.split(":", 1)
        assert module_path and attr and "." in module_path


# --------------------------------------------------------------------------- #
# 5. agents 公平性（AST 结构面——isaaclab 离线不可构造）
# --------------------------------------------------------------------------- #

def _class_bodies(source: str) -> dict:
    """agents.py → {class_name: (bases, {assigned_attr: ast.dump(expr)})}。"""
    out: dict[str, tuple[list[str], dict]] = {}
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [ast.unparse(b) for b in node.bases]
        assigns: dict[str, str] = {}
        for stmt in node.body:
            targets, value = [], None
            if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            elif isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            for t in targets:
                if isinstance(t, ast.Name):
                    assigns[t.id] = ast.dump(value)
        out[node.name] = (bases, assigns)
    return out


def _raw_assigns(source: str, cls: str) -> dict:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out = {}
            for stmt in node.body:
                value, targets = None, []
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    targets, value = [stmt.target], stmt.value
                elif isinstance(stmt, ast.Assign):
                    targets, value = stmt.targets, stmt.value
                for t in targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = ast.unparse(value)
            return out
    raise AssertionError(cls)


def test_cmd_runner_cfg_is_min_subclass_fair_with_three_arms():
    """arm D runner cfg 与三 kick runner cfg 公平性：同一基类最小子类，仅
    experiment_name/actor 两处差异（PPO 超参/obs_groups/critic/clip_actions
    逐字继承）+ 臂名 upper_ukick_cmd + actor = UpperCmdActorCfg()。"""
    src = _source(_AGENTS_PATH)
    bodies = _class_bodies(src)
    assert "UpperKickCmdPPORunnerCfg" in bodies
    bases, assigns = bodies["UpperKickCmdPPORunnerCfg"]
    assert bases == ["UpperGo2FlatPPORunnerCfg"], (bases,)
    assert set(assigns) == {"experiment_name", "actor"}, (assigns,)
    raw = _raw_assigns(src, "UpperKickCmdPPORunnerCfg")
    assert raw["experiment_name"] == repr("upper_ukick_cmd")
    assert raw["actor"] == "UpperCmdActorCfg()"
    # [cmd-only 拆分] e2e/hier/ours runner cfg 已随臂移除——公平性锚改为
    # 基类逐字同源断言（PPO 超参/obs_groups/clip_actions/critic 全在基类，
    # UT 三臂本就逐字继承同一基类）。
    for cls in ("UpperKickE2EPPORunnerCfg", "UpperKickHierPPORunnerCfg",
                "UpperKickOursPPORunnerCfg"):
        assert cls not in bodies, cls


def test_cmd_actor_cfg_field_face():
    """UpperCmdActorCfg 字段面（同 trunk/同探索/同 obs 面；UT 三臂同源值）。

    [cmd-only 拆分] 原 UpperDirectActorCfg 对比锚随 e2e 臂移除——共享字段
    值改为字面锚（值与 UT 三臂逐字同源：hidden_dims [128,128,128] ELU /
    σ_sem 分布 / obs 契约缺省 55）。
    """
    src = _source(_AGENTS_PATH)
    bodies = _class_bodies(src)
    c_bases, c_assigns = bodies["UpperCmdActorCfg"]
    assert c_bases == ["RslRlMLPModelCfg"]
    raw = _raw_assigns(src, "UpperCmdActorCfg")
    assert raw["hidden_dims"] == "[128, 128, 128]"
    assert raw["activation"] == repr("elu")
    assert raw["obs_normalization"] == "False"
    assert raw["sigma_sem"] == "SIGMA_SEM"
    # obs 契约缺省面：arm D 恒 kick 55
    assert raw["obs_dim_contract"] == "KICK_POLICY_OBS_DIM"
    # 无 decoder/z/k_env 面（上层无冻结资产消费）
    assert set(c_assigns) == {"class_name", "hidden_dims", "activation",
                              "obs_normalization", "distribution_cfg",
                              "sigma_sem", "obs_dim_contract"}
    assert "z_stats_path" not in c_assigns and "k_env" not in c_assigns
    assert "decoder_path" not in c_assigns
    # class_name 路径消费在位（agents.py 勿内联点分路径）
    assert "UPPER_CMD_ACTOR_CLASS_PATH" in src


# --------------------------------------------------------------------------- #
# 6. run_upper 源码断言（--actor command 分支 + 双向守卫 + 低层参数）
# --------------------------------------------------------------------------- #

def test_train_command_actor_wiring():
    """train.py cmd-only 面：task 守卫（先于 AppLauncher）+ stock runner +
    低层 ckpt 路径注入 + 存在性守卫（--actor 分支随三臂拆分移除）。"""
    src = _source(_RUN_UPPER_PATH)
    ast.parse(src)
    # [cmd-only] 无 --actor argparse 分支（docstring 裁剪注提及不算）；
    # task 守卫指向 cmd 面
    assert 'add_argument("--actor"' not in src
    assert "KICK_CMD_TASK_ID" in src
    assert "仅支持 cmd 任务 id" in src
    # 守卫先于 AppLauncher（launch 最早面 fail-fast）
    guard_pos = src.index("仅支持 cmd 任务 id")
    launcher_pos = src.index("app_launcher = AppLauncher")
    assert guard_pos < launcher_pos
    # 低层 ckpt 参数化 + env cfg 注入 + 存在性守卫
    assert "--low_level_ckpt" in src
    assert "env_cfg.actions.joint_pos.low_level_ckpt_path" in src
    assert "低层 ckpt 不存在" in src
    # ours/e2e 专属 CLI 面零残留（argparse 面；docstring 裁剪注提及不算）
    for gone in ("--decoder_ckpt", "--sfeasible_ckpt", "--u31_config",
                 "--pca-k", "--cmd_support_box"):
        assert f'add_argument("{gone}"' not in src, gone
    # stock rsl_rl runner（cmd 臂下 UT UpperRunner 附加面全短路）
    assert "from rsl_rl.runners import OnPolicyRunner" in src
    assert "OnPolicyRunner(" in src
    # kick 面 PPO 配方单源（entropy 0.001，原 upper_ukick.yaml 值内联）
    assert "ENTROPY_COEF = 0.001" in src
    assert "agent_cfg.algorithm.entropy_coef = ENTROPY_COEF" in src


def test_xhk2_ppo_knobs_and_eval_stochastic_source():
    """[X-HK2] PPO 旋钮 + eval stochastic 旗标源码面：train.py
    --num_steps_per_env/--gamma 实例级覆写（类定义不动）+ PPO 面核对打印；
    eval_kick.py --stochastic 随机采样口径 + 固定 seed 臂间可比。"""
    tsrc = _source(_RUN_UPPER_PATH)
    ast.parse(tsrc)
    # CLI（缺省 = 现状 24/0.99）
    assert 'add_argument("--num_steps_per_env", type=int, default=24' in tsrc
    assert 'add_argument("--gamma", type=float, default=0.99' in tsrc
    # 实例级覆写（agents cfg 实例属性——UpperGo2FlatPPORunnerCfg 类定义不动）
    assert "agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)" in tsrc
    assert "agent_cfg.algorithm.gamma = float(args_cli.gamma)" in tsrc
    # 启动日志 PPO 面核对行（num_steps/gamma/entropy/survival/权重表锚）
    assert "[hkick][X-HK2] 最终 PPO 面" in tsrc
    assert "num_steps_per_env=" in tsrc and "gamma=" in tsrc
    # eval 仪器：--stochastic 随机采样 + 固定 seed
    esrc = _source(os.path.join(_PKG_ROOT, "hierarchical_kick", "scripts",
                                "eval_kick.py"))
    ast.parse(esrc)
    assert 'add_argument("--stochastic", action="store_true"' in esrc
    assert "stochastic_output=stochastic" in esrc
    assert "torch.manual_seed" in esrc
    assert '"stochastic": stochastic' in esrc  # json 口径落盘


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
