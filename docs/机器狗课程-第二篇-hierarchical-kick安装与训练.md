# 机器狗课程 · 第二篇：四足踢球分层强化学习（hierarchical_kick）安装与训练

> **目标**：在讲义一装好的机器上，跑通我们的四足踢球项目——一只机器狗用"分层强化学习"（上层大脑发速度命令 + 冻结的底层跑步模型执行腿脚）把球撞进球门。完成：克隆代码包 → 1 分钟冒烟测试 → 正式训练 2000 轮（约 30 分钟）→ 评估"球放多远都能进球"的能力。
>
> **面向对象**：已经完成**讲义一**（Isaac Lab 3.0 环境安装 + Cartpole 验收）的同学。本篇所有命令同样**逐字符照抄**。
>
> 预计总耗时：**约 1 小时**（训练的 30 分钟可以走开等）。
>
> **本体覆盖**：第 1~7 节以 **Go2** 为例（`main` 分支同款）；**CyberDog2（铁蛋2）的两段式训练见第 8 节**（`cyberdog2` 分支新增内容）。

---

## 0. 你将运行什么？和讲义一是什么关系？

### 0.1 讲义一装好的东西，直接复用

| 讲义一装好的 | 本篇怎么用 |
| --- | --- |
| NVIDIA 驱动 + RTX 显卡 | 跑仿真 |
| conda 环境 `env_isaaclab`（python 3.12） | 本篇所有命令都在它里面执行 |
| Isaac Sim 6.1 + Isaac Lab 3.0 | 本篇的仿真引擎与 RL 框架 |

**本篇不重装任何底层软件**，只新增一个我们写的代码包。

### 0.2 你将运行的任务：单狗单球射门

一只 Go2 机器狗出生在场地里，前方有一颗球，远处（+y 方向 3 米）有一个球门：

- **上层策略**（我们要训练的"大脑"）：每 0.02 秒观察一次（自己的关节/姿态 + 球的位置/方位），输出 3 个数字 `[vx, vy, wz]`（前后/左右/转向速度命令）
- **底层模型**（冻结、不训练的"小脑"）：官方口径的速度跟踪模型，把速度命令翻译成 12 个关节的目标角度
- **奖励**：把球推进球门 **+500**（一次性大额）；球向门移动有小额过程奖励；摔倒 −5
- **出生课程**：球的位置每局随机——从贴脚 0.1 米到远端 2.7 米，训练中逐渐拉远

训练完成后，狗应该学会：**走向球 → 把球朝门方向推 → 球过线得分**。我们实测：球放在 0.5~2.7 米时，进球率 **74%~93%**。

### 0.3 整体流程总览

```
第 1 步  克隆代码包（1 分钟）
第 2 步  安装代码包到 conda 环境（1 分钟）
第 3 步  冒烟测试：2 轮迭代（1 分钟）
第 4 步  正式训练 2000 轮（约 30 分钟，可走开等）
第 5 步  评估：球距响应扫描（5 分钟）
```

### 0.4 ⚠️ 开跑前必读：三个大坑

1. **每开一个新终端，先做两件事**：`conda activate env_isaaclab`（看到提示符带 `(env_isaaclab)`），再 `cd ~/kicking && export PYTHONPATH=$HOME/kicking`。少任何一件，后面的命令都会报 `ModuleNotFoundError` 或找不到文件。
2. **仿真首次启动会要你同意 NVIDIA 许可证（EULA）**。纯终端环境会卡在提问处不动。解决：命令前加 `OMNI_KIT_ACCEPT_EULA=YES `（见第 3 步的完整命令，已帮你加好）。
3. **显存不足会崩**。4096 个并行环境在 32GB 显卡上约占 5GB；如果你的显卡显存 ≤8GB，把命令里的 `--num_envs 4096` 改成 `--num_envs 1024`（训练会慢一些，其余不变）。

---

## 1. 第一步：克隆代码包

打开终端，激活环境，然后克隆：

```bash
conda activate env_isaaclab
cd ~
git clone https://github.com/unsigned123/kicking.git kicking
cd kicking
export PYTHONPATH=$HOME/kicking
```

逐行解释：第一行激活讲义一的 conda 环境；第三行把代码下载到 `~/kicking` 文件夹；最后一行告诉 Python 去哪里找我们的代码（**这个 export 只对当前终端有效**，每开新终端要重新执行，见 0.4 第一坑）。

✅ **检查点 1**：确认代码和预训练模型都在：

```bash
ls hierarchical_kick/weights/velocity_flat_official/
```

你应该看到至少两个文件：`model_299.pt` 和 `model_walk.pt`（`model_walk.pt` 是预训练好的"会走路"的底层模型，训练时会自动加载）。再确认代码能被 Python 找到：

```bash
python -c "import hierarchical_kick; print('包 OK')"
```

预期输出：`包 OK`。如果报 `ModuleNotFoundError`，说明 `export PYTHONPATH` 没执行，回本节开头。

## 2. 第二步：安装代码包（可编辑模式）

```bash
pip install -e . --no-deps
```

`-e` 表示"可编辑安装"——代码还在原文件夹，但 Python 任何时候都能找到它；`--no-deps` 表示不自动装依赖（讲义一已装齐，自动装反而可能改乱版本）。

✅ **检查点 2**：

```bash
python -c "import hierarchical_kick; print('安装 OK')"
```

预期输出：`安装 OK`。

## 3. 第三步：冒烟测试（只跑 2 轮迭代，确认全链路能转）

```bash
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/train.py \
  --num_envs 64 --max_iterations 2 --run_name smoke
```

> 这是**一条完整命令**（反斜杠 `\` 是换行续行符，整段一起复制粘贴，再回车）。

会发生什么：先打印几十行仿真器启动日志（约 10~60 秒），然后你应该看到类似：

```
[hierarchical_kick] PPO config: num_steps_per_env=24  gamma=0.99  ...
[hierarchical_kick] 最终奖励权重表: goal_scored=10.0 ... survival=0.25 ...
                          Learning iteration 0/2
                            Mean reward: 0.14
                          Learning iteration 2/2
```

✅ **检查点 3**：确认模型文件已生成：

```bash
ls logs/rsl_rl/upper_ukick_cmd/*smoke/
```

你应该看到 `model_0.pt`、`model_1.pt` 和 `params/` 文件夹。**看到这些 = 全链路（仿真器→环境→低层→上层→训练器）已打通**。

> 注意冒烟命令**没有** `--kick_rw_overrides` 和 `--survival_weight 0`——所以它用的是默认权重（goal=10）。这只是测管道，不影响第四步。

## 4. 第四步：正式训练（2000 轮，约 30 分钟）

```bash
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/train.py \
  --num_envs 4096 --seed 42 --max_iterations 2000 \
  --survival_weight 0 \
  --kick_rw_overrides '{"goal_scored": 500.0, "ball_distance": 0.05, "ball_distance_temperature": 1.0, "approach_ball_velocity": 0.30}' \
  --run_name my_first_kick
```

> ⚠️ `--kick_rw_overrides` 后面是一段 **JSON**：外层用**单引号**，内层用**双引号**，顺序不能反。这一段的含义：进球奖励 10→**500**（这是我们实验出的关键值，见 FAQ 6.4）、关掉生存底薪、距离引导改陡。**照抄即可。**

### 你应该看到什么

1. 启动时打印权重核对表，关键行：

```
goal_scored                = 500.0
survival                   = 0.0
[hierarchical_kick] 最终 PPO 面: num_steps_per_env=24  gamma=0.99  ...
```

2. 训练中（约 0.8 秒一轮），每隔 100 轮打印一次：

```
                          Learning iteration 500/2000
                            Mean reward: 8.44
                    Mean episode length: 224.00
```

3. **健康标志**（实测参考值）：
   - `Mean episode length` 在 **180~240** 之间——不是狗跑不全程，而是**进球即终止**（一局进球就结束开下一局，ep_len 短恰恰说明进球频繁）
   - `Mean reward` 在 **7~10** 之间——主要来自 +10 的进球奖励
   - 训练前 ~500 轮球的位置课程会从 0.5 米逐渐放开到 2.7 米（启动日志有 `d_max 调度` 打印）

✅ **检查点 4**：训练结束（打印 `Learning iteration 2000/2000`）后确认：

```bash
ls logs/rsl_rl/upper_ukick_cmd/*my_first_kick/ | grep model_1999
```

看到 `model_1999.pt` 即训练完成。中途随时 `Ctrl+C` 也不怕——每 100 轮存一次档，最新模型都在同目录。

## 5. 第五步：评估——球放多远都能进球吗？

评估脚本会让 64 只狗各踢 10 局，统计进球率。我们先评"默认位置"（球贴脚 0.1 米），再扫远档：

```bash
CKPT=$(ls -td logs/rsl_rl/upper_ukick_cmd/*my_first_kick | head -1)/model_1999.pt
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/eval_kick.py \
  --checkpoint "$CKPT" --num_envs 64 --episodes 10 --out eval_0.1m.json
```

> 第一行把最新模型的路径存进 `CKPT` 变量，第二行引用它。

然后逐档拉远（每档约 1 分钟）：

```bash
for D in 0.5 1.0 2.0 2.7; do
  OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/eval_kick.py \
    --checkpoint "$CKPT" --num_envs 64 --episodes 10 --ball_dist $D \
    --out eval_${D}m.json
done
```

### 你应该看到什么（实测参考值）

| `--ball_dist` | goal_rate（进球率） | kick_count（场均触球） | 说明 |
|---|---|---|---|
| 0.1 | ~1%（已知最弱档） | ~18（高频碰球但球压在脚下） | 球贴脚时踢击姿态受限，课程已知弱项 |
| 0.5 | ~74% | ~4 | |
| 1.0 | ~92% | ~1 | |
| 2.0 | ~93% | ~1 | |
| 2.7 | ~87% | ~1 | 最远档轻微回落 |

json 文件里有逐集明细；快速看总数：

```bash
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('eval_*.json')):
    d = json.load(open(f))
    print(f, d['overall']['goal_rate'])
EOF
```

## 6. 可选：看训练曲线

```bash
tensorboard --logdir logs/rsl_rl/upper_ukick_cmd --port 6006
```

浏览器打开 `http://localhost:6006`。关心的两条曲线：`Mean reward`（应升到 8~10 后平稳）与 `Episode Length`（180~240 波动=进球频繁）。

---

## 8. CyberDog2 本体：两段式训练（cyberdog2 分支）

> 本节在 **`cyberdog2` 分支**上（`git checkout cyberdog2`）。目标本体：小米 CyberDog2（铁蛋2，12 关节四足）。流程分两段：**先训速度跟踪低层（学会走路），再训踢球上层**——两段都已实测跑通。

### 8.1 第一段：cyberdog2 速度跟踪低层

```bash
conda activate env_isaaclab
cd ~/kicking && export PYTHONPATH=$HOME/kicking
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/train_velocity.py   --robot cyberdog2 --num_envs 4096 --seed 42 --max_iterations 3000   --run_name cyber_walk
```

会发生什么：与讲义一同款启动日志，随后训练约 25~40 分钟。

**你应该看到**：`Mean reward` 从负值一路升到 **300+**（约 3000 轮收敛，实测 327.5），`Mean episode length` 满值 1000。

✅ **检查点 8a**：行走探针（会走 = v_body 均值 ≥0.25 m/s + 步态振荡）：

```bash
RUN=$(ls -td logs/rsl_rl/hk_velocity_flat_cyberdog2/*cyber_walk | head -1)
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/probe_velocity_walk.py   --robot cyberdog2 --ckpt $RUN/model_2999.pt --num_envs 1 --steps 250 --cmd_vx 0.4
```

实测参考：`v_body 末 50 步均值 0.290（有球）/ 0.391（无球）`、`峰值 0.554`、`步态振荡 PASS`。

过门后把模型放进缺省位（踢球训练会自动加载它）：

```bash
cp $RUN/model_2999.pt hierarchical_kick/weights/velocity_flat_official/cyberdog2/model_walk.pt
```

### 8.2 第二段：cyberdog2 踢球训练

```bash
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/train.py   --robot cyberdog2 --num_envs 4096 --seed 42 --max_iterations 2000   --survival_weight 0   --kick_rw_overrides '{"goal_scored": 500.0, "ball_distance": 0.05, "ball_distance_temperature": 1.0, "approach_ball_velocity": 0.30}'   --run_name cyber_kick
```

**你应该看到**：`goal 终止占比` 从个位数一路爬到 **~77%** 并稳定（实测 77.3%@1999，无塌陷）。

✅ **检查点 8b**：评估（同第 5 步，加 `--robot cyberdog2`）：

```bash
CKPT=$(ls -td logs/rsl_rl/upper_ukick_cmd/*cyber_kick | head -1)/model_1999.pt
OMNI_KIT_ACCEPT_EULA=YES python hierarchical_kick/scripts/eval_kick.py   --robot cyberdog2 --checkpoint "$CKPT" --num_envs 64 --episodes 10 --out eval_cyber.json
```

**实测参考值**：`kick_count ≈ 8.3/局`、`goal_rate ≈ 43.9%`（球距 1.0 m 档 **84%**、球速峰 2.17 m/s）。

### 8.3 CyberDog2 注意事项

- **摔倒高度阈值已按本体标定为 0.17 m**（实测纯步态波谷最低 0.184 m，调高会误杀正常步态）
- **低层计算频率为每环境步一次（50 Hz）**；Go2 保持每仿真子步（200 Hz）历史语义，两者由 cfg 旗标自动分派
- **关节序（PhysX）**：`FR,FL,RR,RL × abad→hip→knee`——与训练面自洽，不要在装配层做置换
- 换本体时执行器/初始关节角**必须与低层训练配置同参**（本包 profile 已内置，勿手改）

---

## 9. 常见问题 FAQ（报错先来这里对号入座）

### 9.1 启动与环境类

| 症状 | 原因与解决 |
| --- | --- |
| `ModuleNotFoundError: No module named 'hierarchical_kick'` | 没装包或 PYTHONPATH 丢了。重跑第 2 步的 `pip install -e . --no-deps`；或每开新终端执行 `cd ~/kicking && export PYTHONPATH=$HOME/kicking` |
| 卡在 `Do you accept the EULA? (Yes/No)` 不动 | 见 0.4 第二坑。`Ctrl+C` 退出，命令前加 `OMNI_KIT_ACCEPT_EULA=YES ` 重跑 |
| `CUDA out of memory` / 显存不足 | `--num_envs 4096` 改小：`1024`（8GB 卡）或 `2048`（16GB 卡）。训练轮数不变 |
| 报 `no kernel image is available for execution` | 显卡太老或 torch 不是 CUDA 版。回讲义一 4.3 节自检 |
| 想用第二块显卡 | 命令前加 `CUDA_VISIBLE_DEVICES=1 `（注意末尾有空格） |

### 9.2 训练行为类

| 症状 | 原因与解决 |
| --- | --- |
| `Mean episode length` 只有 20~30 | **正常**——进球即终止（球就在门线附近布局），ep_len 短=进球快 |
| `Mean reward` 冲到 50~100+ | **正常且是好事**——每个进球 +10（有效值），进球越多奖励越高 |
| 训练全程 `Mean reward` < 2、goal 从不出现 | 九成是 `--kick_rw_overrides` 没照抄对（尤其 `goal_scored: 500.0`）。检查启动时打印的权重核对表第一行是不是 500.0 |
| 想从头重新训练 | 换一个 `--run_name`（如 `my_second_kick`），旧数据自动保留 |
| 训练中断了想接着练 | 加 `--resume --checkpoint logs/rsl_rl/upper_ukick_cmd/<你的run目录>/model_xxx.pt` |

### 9.3 评估类

| 症状 | 原因与解决 |
| --- | --- |
| `eval_kick.py` 找不到 checkpoint | `$CKPT` 变量没设置（每开新终端要重设）。重跑第 5 步第一段 |
| 所有档 goal_rate 都是 0 | 确认评估的是 `model_1999.pt`（训练完成的终档），且用的是 `--run_name my_first_kick` 那个目录 |

---

## 8. 产物在哪、下一步玩什么

| 产物 | 位置 |
|---|---|
| 训练模型（每 100 轮一存） | `logs/rsl_rl/upper_ukick_cmd/<日期>_<run_name>/model_*.pt` |
| 训练配置快照 | 同目录 `params/` |
| 评估逐集明细 | `eval_*.json` |

想深入玩的旋钮（都在 `hierarchical_kick/sim/ball_env_cfg.py`，改完重跑即可）：

- 奖励权重：`KickRewardWeightsCfg`（进球/过程引导/惩罚，共 9 项）
- 课程节奏：`KickRandomizationCfg` 的 `warmup_iters`（课程拉满的快慢）与 `d_max_final`（最远球距）
- 球出生随机幅度：`ball_xy_range`（横向散布）

祝你踢得开心。
