# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ---------------------------------------------------------------------------
# UPPER 自建（非 vendor）— hierarchical_kick.sim.mdp.frozen_loco_action
#
# U-Kick 第四臂「真分层」对照（arm D；docs/UKICK_BALL_ENV_DESIGN.md §0 谱系
# 扩展）：上层 RL 策略动作空间 = 3 维机体系速度命令 [vx,vy,wz]，动作不进
# 关节，而是写入 base_velocity 命令 buffer，由**冻结低层** policy
# （deterministic mean）每步把命令转成 12 维关节目标。经典「冻结低层
# locomotion + 高层命令 RL」分层（低层冻结、高层可训）。
#
# 低层约定（2026-10-09 官方口径对齐；cfg.low_level_convention 分派）：
#   【official（缺省）】上游官方 velocity-flat 契约——低层 ckpt =
#     HK-Velocity-Flat-Unitree-Go2-v0 训练产物（sim/velocity_official_cfg.py
#     覆写回上游口径后训练；缺省 weights/velocity_flat_official/model_walk.pt）：
#     - obs 48 = 官方单帧布局（linvel→ang_vel→gravity→cmd→q→qd→last_action，
#       joint_vel/base_ang_vel raw 无缩放）——组装公式
#       frozen_loco_math.assemble_low_level_obs（纯函数，离线单测对象）；
#     - 动作语义 = default_joint_pos + 0.5⊙action（**均匀 0.5**，
#       contracts.OFFICIAL_ACTION_SCALE 单源；逐关节 scale_vec 表不走缺省路径）；
#     - 网络结构 = 官方 go2 agents cfg（MLP[128,128,128] ELU）——ckpt
#       state_dict 反推对账（core.low_level_actor.
#       validate_official_actor_structure，漂移 fail-fast）。
#   【legacy】旧 U4b B 臂契约（weights/model_2999.pt）：
#     - obs 48 = proprio42 前缀 + linvel + cmd（joint_vel/ang_vel 按旧适配
#       表缩放，参数单源在 frozen_loco_math legacy 分支/contracts）——
#       frozen_loco_math.assemble_low_level_obs_legacy；
#     - 动作语义 = default + scale_vec ⊙ action（core/scale_vec 单源逐关节
#       表，经 cfg.scale dict 传入）。
#   共同面：推理无噪声（Unoise 训练面才施加）；低层 forward 走
#   ``stochastic_output=False``（deterministic mean）+ torch.no_grad +
#   eval + requires_grad_(False)（冻结三件套）。
#
# ⚠️ 类型注解纪律：env 面注解**一律加引号或 TYPE_CHECKING**（ball_env_cfg
# 曾因 runtime 注解炸过 ManagerBasedRLEnv NameError——configclass 字段注解
# 求值早于 isaaclab.envs 完整初始化；本文件 ``from __future__ import
# annotations`` 全程生效）。
#
# 【无时间抽象 / action repeat 说明】本 term 为 50Hz 逐步命令：低层在每个
# sim 子步（decimation 内 4 次/环境步）重观测并重算关节目标，命令 buffer
# 逐子步覆写——与三臂 JointPositionAction 的子步写入语义对齐，公平性
# 「差异只在动作接口」。无 ASE 式 6Hz 低频命令 + 5 步 hold 的时间抽象
# （U33 审计注：那是后续变体，不进本对照）。
#
# 【命令 resample 竞争分析】ManagerBasedRLEnv.step 时序：apply_actions（写
# 命令 buffer，decimation 循环内）→ 终止/奖励 → reset（command_manager.
# reset 对 reset envs 重采样）→ command_manager.compute（仅 time_left≤0 的
# env 重采样）→ obs 计算。resampling_time_range=(10,10) == episode_length_s
# =10 → compute 期重采样只可能命中恰在本步 time_out 的 envs（其 reset 已
# 重采样并复位 time_left=10s）→ **永不与存活 env 的写入竞争**；且 kick 面
# 高层 obs（55 维）/奖励/终止均不消费命令 buffer——残留的随机命令对训练
# 零影响（仅 cmd term 自身 metrics 读数）。
# ---------------------------------------------------------------------------

"""Frozen-locomotion action term: high-level [vx,vy,wz] commands → frozen
low-level policy → 12 joint position targets (U-Kick arm D)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

import isaaclab.utils.string as string_utils

from hierarchical_kick.core.contracts import (
    ACTION_DIM,
    OBS_DIM,
    OFFICIAL_ACTION_SCALE,
    SIGMA_SEM,
)
from hierarchical_kick.core.low_level_actor import (
    build_upper_direct_actor,
    resolve_actor_state_dict,
    validate_official_actor_structure,
)
from hierarchical_kick.core.weights import (
    DEFAULT_LOW_LEVEL_CKPT,
    DEFAULT_LOW_LEVEL_CKPT_OFFICIAL,
)

from .frozen_loco_math import (
    LOW_LEVEL_OBS_DIM,
    assemble_low_level_obs,
    assemble_low_level_obs_legacy,
    joint_target_from_low_action,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

# arm D 上层动作维（机体系速度命令 [vx,vy,wz]；= core.cmd_actor.CMD_ACTION_DIM
# 同值——core 面不依赖 sim 面，此处独立常量并经单测锚定同值）
CMD_ACTION_DIM = 3

# 低层约定分派（cfg.low_level_convention；official 缺省）
LOW_LEVEL_CONVENTIONS = ("official", "legacy")
# 缺省 ckpt 路径单源 = core/weights（此处 import 即 re-export，__all__ 兼容旧消费面）


class FrozenLocoJointPositionAction(ActionTerm):
    """[vx,vy,wz] 命令 → 写命令 buffer → 冻结低层 → 关节位置目标。

    语义（apply_actions 四步，见模块头）：① 命令写入（命令 buffer 前 3 维）
    ② 冻结低层 48 维 obs 组装（frozen_loco_math 纯函数，official/legacy
    分派；低层 last_action = 自己上一步 pre-scale 输出，env reset 清零）
    ③ 低层 deterministic forward（no_grad/eval/冻结）④ 关节目标 =
    default + scale ⊙ 输出 → ``set_joint_position_target``（official 均匀
    0.5 / legacy 逐关节 scale_vec）。action_dim = 3（上层动作空间；非
    12——命令接口本体）。
    """

    cfg: FrozenLocoJointPositionActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: FrozenLocoJointPositionActionCfg,
                 env: 'ManagerBasedRLEnv') -> None:
        super().__init__(cfg, env)
        # -- 约定分派（official 缺省；非法值 fail-fast） --
        self._convention = str(getattr(cfg, "low_level_convention", "")
                               or "official")
        if self._convention not in LOW_LEVEL_CONVENTIONS:
            raise ValueError(
                f"[upper][armD] low_level_convention 非法: "
                f"{self._convention!r}（∈ {LOW_LEVEL_CONVENTIONS}）")
        # -- 冻结资产 fail-fast（路径存在性；空串 = 约定缺省路径） --
        path = str(cfg.low_level_ckpt_path) or (
            DEFAULT_LOW_LEVEL_CKPT_OFFICIAL
            if self._convention == "official" else DEFAULT_LOW_LEVEL_CKPT)
        if not path:
            raise ValueError(
                "[upper][armD] low_level_ckpt_path 缺失（冻结低层是显式资产；"
                "train.py --low_level_ckpt 或 cfg 显式传路径）")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[upper][armD] 低层 ckpt 不存在: {path}")
        # -- 关节解析（PhysX 槽位序；".*" → 全关节 slice） --
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names)
        self._num_joints = len(self._joint_names)
        if self._num_joints != ACTION_DIM:
            raise ValueError(
                f"[upper][armD] 低层动作契约 12 关节 != 解析 {self._num_joints}"
                f"（{self._joint_names}；go2 装配漂移，fail-fast）")
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)
        # -- 动作 scale（按约定分派） --
        # official：均匀 0.5（contracts.OFFICIAL_ACTION_SCALE 单源；cfg.scale
        #   逐关节表**不消费**——打印提示防误配）。
        # legacy：cfg.scale 逐关节 dict → 槽位序 tensor（JointAction 同款
        #   解析；core/scale_vec 单源表经 cfg 传入）。
        if self._convention == "official":
            if self.cfg.scale:
                print(f"[upper][armD] official 约定忽略 cfg.scale 逐关节表"
                      f"（均匀 {OFFICIAL_ACTION_SCALE}；官方口径不走逐关节 "
                      f"scale_vec）")
            self._scale_vec = torch.full(
                (self.num_envs, self._num_joints), float(OFFICIAL_ACTION_SCALE),
                device=self.device)
        else:
            if not isinstance(self.cfg.scale, dict):
                raise ValueError(
                    "[upper][armD] legacy 约定 scale 须为逐关节 dict"
                    "（core/scale_vec.go2_action_scale_table() 单源）")
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names)
            self._scale_vec = torch.ones(self.num_envs, self._num_joints,
                                         device=self.device)
            self._scale_vec[:, index_list] = torch.tensor(
                value_list, device=self.device)
        # -- offset = default_joint_pos（use_default_offset 语义） --
        self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        # -- buffers --
        self._raw_actions = torch.zeros(self.num_envs, CMD_ACTION_DIM,
                                        device=self.device)
        self._joint_targets = torch.zeros(
            self.num_envs, self._num_joints, device=self.device)
        self._low_last_action = torch.zeros(
            self.num_envs, ACTION_DIM, device=self.device)
        self._low_obs = torch.zeros(
            self.num_envs, LOW_LEVEL_OBS_DIM, device=self.device)
        # -- 冻结低层 policy（构造期装载 + freeze 三件套；无梯度/无优化器） --
        # 结构按约定：official = 官方 agents cfg（ckpt sd 反推对账）；legacy
        # = U4b B 臂 [128,128,128]。
        ckpt_full, sd = resolve_actor_state_dict(path, device=self.device)
        if self._convention == "official":
            hidden_dims = validate_official_actor_structure(sd)
        else:
            hidden_dims = [128, 128, 128]
        dummy_obs = {"policy": torch.zeros(
            self.num_envs, OBS_DIM, device=self.device)}
        self._low_policy = build_upper_direct_actor(
            dummy_obs, actor_cfg=None, device=self.device,
            hidden_dims=hidden_dims, activation="elu",
            obs_normalization=False,
            distribution_cfg={"class_name": "rsl_rl.modules:GaussianDistribution",
                              "init_std": SIGMA_SEM},
            sigma_sem=SIGMA_SEM, obs_dim_contract=OBS_DIM)
        self._low_policy.load_state_dict(sd, strict=True)
        self._low_policy.eval()
        for p in self._low_policy.parameters():
            p.requires_grad_(False)
        # -- 组装函数分派（official 缺省布局 / legacy 旧布局） --
        self._assemble = (assemble_low_level_obs if self._convention == "official"
                          else assemble_low_level_obs_legacy)
        # -- 命令 term 引用（懒解析：action manager 构造先于 command manager，
        #    ctor 期 env.command_manager 尚不存在——apply_actions 首步解析） --
        self._cmd_term = None
        _it = ckpt_full.get("iter", "?")
        print(f"[upper][armD] 冻结低层装载完成（{self._convention}）: {path} "
              f"(iter={_it}, hidden={hidden_dims}, obs={LOW_LEVEL_OBS_DIM} 维"
              f"契约 deterministic；50Hz 逐步命令，无时间抽象)")

    # ------------------------------------------------------------------ #
    # Properties（ActionTerm 抽象面；action_dim = 3 命令维）
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return CMD_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_targets

    @property
    def low_level_convention(self) -> str:
        """低层约定（official/legacy；cfg 分派只读面）。"""
        return self._convention

    @property
    def low_level_last_action(self) -> torch.Tensor:
        """低层上一步 pre-scale 输出 (N,12)（高层 obs ``last_action`` 通道
        取值源——arm D 高层动作是 3 维命令，kick 契约 42-prefix 的
        last_action 12 维通道改由低层输出填充，保持 55 维 obs 契约与
        「关节级 last action」通道语义）。"""
        return self._low_last_action

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def process_actions(self, actions: torch.Tensor):
        """存 raw [N,3] 命令（每环境步一次）。"""
        self._raw_actions[:] = actions.to(self.device)

    def apply_actions(self):
        """命令写入 → 低层 obs 组装 → 冻结 forward → 关节目标入 sim。

        每 sim 子步调用（decimation 内 4 次/环境步）：低层每子步重观测
        重算（50Hz 逐步命令，无时间抽象）——与三臂关节目标子步覆写语义
        对齐；deterministic 下 4 次输出随最新观测微更新，命令 buffer 同值
        覆写幂等。
        """
        # 命令 term 懒解析（构造序：action manager 先于 command manager）
        if self._cmd_term is None:
            self._cmd_term = self._env.command_manager.get_term(
                self.cfg.command_term_name)
        # ① 命令写入（buffer 前 3 维 = 机体系 [vx,vy,wz]）
        self._cmd_term.command[:, :3] = self._raw_actions
        # ② 低层 48 维 obs 组装（official/legacy 分派；推理无噪声；两分支
        #    前序 8 位置参同签名——约定分派在 ctor 期 self._assemble 单点）
        self._assemble(
            self._asset.data.joint_pos.torch[:, self._joint_ids],
            self._offset,
            self._asset.data.joint_vel.torch[:, self._joint_ids],
            self._asset.data.root_ang_vel_b.torch,
            self._asset.data.projected_gravity_b.torch,
            self._low_last_action,
            self._asset.data.root_lin_vel_b.torch,
            self._raw_actions,
            out=self._low_obs,
        )
        # ③ 冻结低层 forward（deterministic mean；无梯度）
        with torch.no_grad():
            low_out = self._low_policy(
                {"policy": self._low_obs}, stochastic_output=False)
        self._low_last_action.copy_(low_out)
        # ④ 关节目标 = default + scale ⊙ 输出 → sim
        #    （JointPositionAction 语义；frozen_loco_math 纯公式）
        self._joint_targets = joint_target_from_low_action(
            self._offset, self._scale_vec, low_out)
        self._asset.set_joint_position_target(
            self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        """env reset 语义：raw 命令 + 低层 last_action 清零（低层 obs 的
        last_action 通道回零起点——U4b 训练 env 侧 action term reset 同款）。"""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._low_last_action[env_ids] = 0.0


@configclass
class FrozenLocoJointPositionActionCfg(ActionTermCfg):
    """arm D 动作面 cfg：3 维命令输入 + 冻结低层资产（路径 fail-fast）。

    ``joint_names`` 语义同 JointPositionActionCfg（arm D 恒 ``[".*"]``）；
    ``scale`` 仅 legacy 约定消费（core/scale_vec 单源 dict；official 约定
    忽略——均匀 0.5）；``low_level_convention`` 分派（缺省 official）；
    低层资产字段 = 路径（空串 = 约定缺省，构造期存在性 fail-fast）。
    """

    class_type: type[ActionTerm] = FrozenLocoJointPositionAction
    """关联动作 term（类对象单点回填——本 cfg 类定义于 term 类之后，零
    前向引用）。"""

    joint_names: list[str] = None  # type: ignore[assignment]
    """动作映射的关节名/正则（arm D 恒 [".*"] = 12 关节 PhysX 槽位序）。"""

    scale: dict[str, float] = None  # type: ignore[assignment]
    """逐关节 action scale（**仅 legacy 约定消费**：core/scale_vec.
    go2_action_scale_table() 单源；official 约定忽略并打印提示）。"""

    low_level_ckpt_path: str = ""
    """冻结低层 ckpt 路径（空串 = 约定缺省：official → weights/
    velocity_flat_official/model_walk.pt、legacy → weights/model_2999.pt；
    launch 期由 train.py --low_level_ckpt 填充）。"""

    low_level_convention: str = "official"
    """低层约定分派（official = 官方 velocity-flat 口径，缺省；legacy =
    旧 U4b B 臂契约——旧 48 布局 + 逐关节 scale_vec）。"""

    command_term_name: str = "base_velocity"
    """命令写入目标 term 名（CommandManager.get_term 键 = commands cfg
    字段名；Go2KickFlatEnvCfgCommand 恢复的 base_velocity 面）。"""

    def __post_init__(self):
        # 缺省面（None 占位 → 字段缺省；configclass 注解经
        # ``from __future__ import annotations`` 全字符串化，零 runtime 求值）
        if self.joint_names is None:
            self.joint_names = [".*"]


def low_level_last_action_obs(env: 'ManagerBasedRLEnv') -> torch.Tensor:
    """高层 obs 项：低层上一步输出 (N,12)（KickPolicyCfgCmd 的 actions 项）。

    arm D 的 ``mdp.last_action``（= action_manager.action）是 3 维命令，
    会破坏 55 维 kick obs 契约的 42-prefix 通道序——本函数改从冻结动作
    term 的 ``low_level_last_action`` buffer 取值，通道语义（pre-scale 关节
    偏移）与三臂逐位同构，obs 维契约 55 保持。
    """
    term = env.action_manager.get_term("joint_pos")
    if not hasattr(term, "low_level_last_action"):
        raise RuntimeError(
            "[upper][armD] actions.joint_pos 非冻结低层动作 term"
            "（KickPolicyCfgCmd 仅配 Go2KickFlatEnvCfgCommand 使用）")
    return term.low_level_last_action


__all__ = [
    "CMD_ACTION_DIM",
    "DEFAULT_LOW_LEVEL_CKPT",
    "DEFAULT_LOW_LEVEL_CKPT_OFFICIAL",
    "LOW_LEVEL_CONVENTIONS",
    "FrozenLocoJointPositionActionCfg",
    "FrozenLocoJointPositionAction",
    "low_level_last_action_obs",
]
