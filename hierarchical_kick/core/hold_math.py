"""段 hold 状态机数学面（12.5Hz 上层动作 → 50Hz 解码帧流；纯 torch 零
 isaaclab；[PA] N1 换代——f4 段时钟，旧 5Hz 形态走旧分支复现位）。

 ours 段空间臂的 env 侧动作面拆分（frozen_loco 同款纪律：isaaclab-facing
 ActionTerm 薄壳在 :mod:`upper.sim.mdp.latent_hold_action`，本模块为离线
 单测对象）。

时钟契约（STVAE_DESIGN §3/§7-2b + MIGRATION_PLAN §2.5-(b) 裁定；
[PA] N1 换代 = f4 段时钟 12.5Hz）：

- 上层策略每 env 步（50Hz）输出 32 维 z；动作 term **每 FRAMES_PER_SEG=4
  步采样新段**（12.5Hz 发射；N1/f4 时钟），其余步 hold+插值——观测/动作
  解频（obs 由
  ObservationManager 每步 50Hz 组装，零改动）。
- 相位 = **逐 env 计数器**（reset 对齐）：reset 清零相位并令 hold/target
  回 μ_corpus（语料均值邻域 = 包络中心，解码安全起点）；相位 0 步 latch
  新段（reset 后首个动作立即生效，首段 = μ→z_0 的 4 帧线性过渡）。
- 段内帧流 = **线性插值**（stvae.interp.linear_seg_interp 的流式形态）：
  帧 j ∈ [0,10) 取 ``z_t = (1−α)·z_hold + α·z_target``，α = j/10——与
  闭环 decoder 训练线的 intent 观测插值同构（同一 interp 语义面）；段首
  帧 α=0 逐位等于 z_hold（端点精确，无数值侵蚀）。

包络纪律：latch 面 ``clamp_envelope`` 把策略采样动作钳回
``[μ−k_env·σ, μ+k_env·σ]``（actor 面同参冗余安全轨——稀有越界尾的结构性
兜底；包络内样本逐位恒等）。
"""
from __future__ import annotations

import torch

from .contracts import FRAMES_PER_SEG

SEG_PHASE_LIMIT = FRAMES_PER_SEG
"""相位计数周期（= 段帧数 4；phase ∈ [0, 4)）。"""


def clamp_envelope(z: torch.Tensor, mu: torch.Tensor,
                   amp: torch.Tensor) -> torch.Tensor:
    """动作 → 包络盒内（逐元素 clamp；纯函数）。

    ``amp = k_env·σ_vec``（per-dim 半宽）；包络内样本逐位恒等。
    """
    lo, hi = mu - amp, mu + amp
    return torch.maximum(torch.minimum(z, hi), lo)


def reset_hold(mu: torch.Tensor, n: int, device: torch.device,
               dtype: torch.dtype = torch.float32) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """reset 面：hold/target 回 μ（包络中心）、相位清零（纯函数）。

    返回 (z_hold, z_target, phase)，均 (N, ·)；μ 广播到 (N, D)。
    """
    if mu.ndim != 1:
        raise ValueError(f"[upper][hold] mu 须一维 (D,): {tuple(mu.shape)}")
    z = mu.to(device=device, dtype=dtype).expand(n, mu.shape[0]).clone()
    phase = torch.zeros(n, dtype=torch.long, device=device)
    return z.clone(), z, phase


def latch_step(z_hold: torch.Tensor, z_target: torch.Tensor,
               phase: torch.Tensor, actions: torch.Tensor,
               mu: torch.Tensor, amp: torch.Tensor) -> tuple[
                   torch.Tensor, torch.Tensor, torch.Tensor]:
    """单 env 步推进（process_actions 面；纯函数——hold 时序单测对象）。

    - 相位 0 的 env：``z_hold ← z_target``（上一段目标成为插值起点），
      ``z_target ← clamp_envelope(actions)``（新段 latch；包络钳）；
    - 其余 env：hold/target 原样（策略中间发射被忽略——5Hz 语义本体）；
    - 相位推进 ``phase ← (phase+1) mod FRAMES_PER_SEG``。

    形状：z_hold/z_target/actions (N, D)，mu/amp (D,) 或 (N, D)，phase (N,)。
    返回新三元组（不就地改输入——调用方 buffer 自行写回）。
    """
    n, d = z_hold.shape
    if z_target.shape != (n, d) or actions.shape != (n, d):
        raise ValueError(
            f"[upper][hold] 形状不符: hold{tuple(z_hold.shape)} "
            f"target{tuple(z_target.shape)} actions{tuple(actions.shape)}")
    if phase.shape != (n,):
        raise ValueError(f"[upper][hold] phase 形状须 ({n},): "
                         f"{tuple(phase.shape)}")
    latch = phase == 0
    new_hold = torch.where(latch.unsqueeze(-1), z_target, z_hold)
    new_target = torch.where(
        latch.unsqueeze(-1),
        clamp_envelope(actions, mu.expand(n, d), amp.expand(n, d)),
        z_target)
    new_phase = (phase + 1) % SEG_PHASE_LIMIT
    return new_hold, new_target, new_phase


def interp_frame(z_hold: torch.Tensor, z_target: torch.Tensor,
                 phase: torch.Tensor) -> torch.Tensor:
    """当前 env 步的解码输入 z（apply_actions 面；纯函数）。

    相位语义钉（test_upper_hold_action 逐位锁定）：latch 发生在本步
    process_actions（相位 0 → 推进到 1），apply_actions 在同一步读**推进
    后**相位，``α = (phase − 1) mod 4 / 4``——每段 4 个 env 步覆盖
    α ∈ {0, 1/10, …, 9/10}，与 stvae.interp.linear_seg_interp 同一插值
    原语（10 帧 α 网格 + 端点精确：z_hold 精确呈现在段首帧）。

    因果相位注（test_upper_hold_action 锚定）：策略在段 s 首步发射的
    z_s 最早在**同段**内渐近逼近、在段 s+1 首帧精确呈现（hold ← z_s）。
    相对 store 流形（linear_seg_interp 的段 s 首帧 = z_s）本流式面按
    因果必然**后移一段**——策略不可能在段 s 开始时知道段 s+1 的值，
    "动作 = 下一段 latent" 的可行语义即本面（发射值 = 本段渐近目标，
    下一精现点在 0.2s 后）。同一步内 decimation 的 4 次 apply_actions
    相位不变 → 同一解码 z（50Hz 子步覆写幂等）。

        reset      → phase=0, hold=target=μ
        step 1     → process: latch（phase==0）→ hold=μ_old_target, target=z_0,
                     phase→1；apply: z_t = lerp(hold, target, (phase−1)/10)
                     = lerp(μ, z_0, 0/10) = μ（段首帧 = 插值起点，端点精确）
        step 2     → z_t = lerp(μ, z_0, 1/10)
        ...
        step 10    → z_t = lerp(μ, z_0, 9/10)
        step 11    → process: latch → target=z_1；apply: lerp(z_0, z_1, 0)
        ...

    即每段 10 帧覆盖 α ∈ {0, 1/10, …, 9/10}——与 stvae.interp.
    linear_seg_interp 的帧级取值逐位一致（帧 f 在段 s：α=(f mod 4)/4）。
    """
    n, d = z_hold.shape
    if z_target.shape != (n, d) or phase.shape != (n,):
        raise ValueError(
            f"[upper][hold] 形状不符: hold{tuple(z_hold.shape)} "
            f"target{tuple(z_target.shape)} phase{tuple(phase.shape)}")
    alpha = ((phase - 1) % SEG_PHASE_LIMIT).to(z_hold.dtype) \
        / float(SEG_PHASE_LIMIT)
    return z_hold + alpha.unsqueeze(-1) * (z_target - z_hold)


__all__ = ["SEG_PHASE_LIMIT", "clamp_envelope", "reset_hold", "latch_step",
           "interp_frame"]
