# cyberdog2 velocity-flat 低层 ckpt（官方口径 · 行走门已过）

- **model_walk.pt**（= run 2026-10-10_21-07-38_vf3000_r12c model_2999.pt）：
  r12（dof_acc −3e-9）@999 续训至 2999（--resume 总预算 3000，再训 2000it@4096env）
  - 训练曲线：Mean reward 307.9@999 → 327.5@2999；err_vel_xy 0.218 → 0.186；
    ep_len 1000 满存活
  - 行走门（probe cmd 0.4 直行 250 步）：
    * 有球：v_body 0.290 PASS（峰 0.538）/ 步态 PASS（14.85 rad/s、35 翻转）/
      推球 0.132 FAIL（球扰动下踉跄后恢复）
    * **无球（纯步态）：v_body 0.391 PASS（峰 0.554）/ 步态 PASS（53 翻转）→ 门过**
    * body_z min 0.1839（kick 面 fallen 阈值标定证据，0.17 < min）
  - md5 **7db8e801c5e4a7a80c07eb528b00bc34**
  - 消费：kick 面 cmd 任务（Isaac-Kick-Flat-CyberDog2-Upper-Cmd-v0）official
    约定缺省低层（frozen_weights 分派）
- model_1999_gatefail.pt（原 model_1999.pt，md5 e424cc7a...）：r8c 2000it 产物，
  行走门 FAIL（v_body 0.003 碎步滑移）——留档锚（r9-r12 塑形链起点）
