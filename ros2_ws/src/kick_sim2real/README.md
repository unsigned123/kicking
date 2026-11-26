# kick_sim2real — 上层策略实机部署（ROS2）

把 `hierarchical_kick` 训练出的**上层 3 维速度命令策略**（55 维 obs → `[vx, vy, wz]`）
部署到 CyberDog2 实机：上层策略在 PC/NX 上以 50Hz 运行，速度命令发给**出厂自带速度
控制器**（不是我们训练的冻结低层），球位置来自动捕。

```
/joint_states ─┐
/imu/data     ─┤
/odom         ─┼→ obs_builder(55) → policy(MLP) → [vx,vy,wz] → 后端
动捕 ball_pose ─┘                                             ├─ twist: /cmd_vel (ROS2)
                                                              └─ lcm:  robot_control_cmd (出厂运控)
```

## 1. 安装

```bash
# ROS2 环境（humble/jazzy 任选，与机器人侧一致）
pip install numpy torch pyyaml

# 本包（本仓库内）
cd ros2_ws
colcon build --packages-select kick_sim2real
source install/setup.bash

# （仅 lcm 后端需要）拉取出厂 LCM 消息类
bash src/kick_sim2real/scripts/fetch_lcm_msgs.sh
```

## 2. 导出策略

```bash
python -m kick_sim2real.export \
  --src hierarchical_kick/logs/rsl_rl/upper_ukick_cmd/<run>/model_1999.pt \
  --dst ~/kick_policy.pt
```

## 3. 配置

编辑 `config/kick_sim2real.yaml`：

- `policy_ckpt`：上一步产物
- `backend`：`twist`（ROS2 话题）或 `lcm`（出厂运控直连）
- `goal_center_xy`：球门中心（世界系）
- `mocap_frame_offset_xy` / `mocap_yaw_offset`：动捕系 → 里程计系外参
- `dry_run: true` 先联调（只打印命令不发布）

## 4. 运行

```bash
# 联调（dry_run）
ros2 launch kick_sim2real kick_sim2real.launch.py \
  params:=$PWD/config/kick_sim2real.yaml

# 正式
# 1) 先在 yaml 里 dry_run: false
# 2) 确认 /joint_states /imu/data /odom /mocap/ball_pose 均有数据
# 3) 急停: ros2 topic pub /kick_sim2real/estop std_msgs/Bool "data: true"
```

## 5. 三种输出后端

### A. `motion_servo`（缺省，与 Machine-Dog 模板同款）

发布 `protocol.msg/MotionServoCmd` 到 `/<dog_name>/motion_servo_cmd`：
`motion_id` 按速度自动选步态（**303 slow / 308 mid / 305 fast**，阈值表与模板
`move_local` 同源）、`cmd_type=1`、`value=2`、`vel_des=[vx,vy,wz]`、
`step_height`。**这就是"出厂自带速度控制器"的 ROS2 发布面。**

### B. `twist`

发布 `geometry_msgs/Twist` 到 `cmd_vel_topic`（备用桥接形态）。

### C. `lcm`（出厂运控文档接口，兜底）

直连 MiRoboticsLab 文档的高层运控：

| 项 | 值 |
|---|---|
| 命令通道 | `udpm://239.255.76.67:7671` → `robot_control_cmd` |
| 行走模式 | `mode=11 (LOCOMOTION)` |
| 步态 | `gait_id=26` 自变频（10=FAST / 3=MEDIUM / 27=SLOW） |
| 速度 | `vel_des=[vx, vy, wz]`（出厂上限 y≤1.6, wz≤2.5） |
| 生效语义 | `duration=0`（持续到下一帧）+ `life_count` 心跳 |
| 超时保护 | 500ms 无帧 → 出厂控制器自行趴下 |
| 状态 | `robot_control_response`（7670, 50Hz）`switch_status`：0=Done 2=ESTOP 3=EDAMP 4=LIFTED |
| 里程计 | `global_to_robot`（7667, 50Hz）`xyz/vBody/omegaBody/rpy` |

### 与 Machine-Dog 模板的集成

- **Skill 封装**：`kick_sim2real/skill.py::KickPolicySkill` 可直接挂进模板的
  `skill_manager`（`self.skills = {5: KickPolicySkill(self, move_skill, ckpt)}`），
  内部把策略输出写入 `move_local` 的 `target_vx/vy/wz`，由模板既有的
  限幅+选步态+发布链路统一输出。
- **数据面完全复用模板话题**：`/absolute_pose/<dog_name>`（狗世界位姿，odom+VRPN
  5s 校准）、`/absolute_pose/ball`（球世界位姿，VRPN）、`/<dog_name>/odom_out`
  （机体速度）、`/vrpn/<obj>/pose|twist|accel`（原始动捕）。**无需自己写动捕驱动**。
- **模板没有的两路输入**：`/joint_states` 与 `/imu/data`。**VP 协议已确认**：
  Imu 与 Odometry 是官方能力（有消息源）；**JointState 不在能力清单**（详见 §6.2
  的消融方案 a）。IMU 来源可从官方 Imu 消息接入，joint 需按 §6.2 处理。

## 5.5 坐标系（odom / 动捕）怎么处理

55 维 obs 全是**机体系相对量**（`ball_rel_b`/`goal_rel_b`/`push_geometry_b` 为 body 系
相对，`vBody/omegaBody/重力/关节` 本就是机体系），因此**任何一个世界系都可以**——
全局平移+偏航（对狗/球/门同步施加）在相对运算里完全抵消（有单测
`test_global_transform_invariance` 逐位验证）。

两条正确做法：

| 方案 | 数据源 | 需要的变换 |
|---|---|---|
| **A（缺省，推荐）** | 狗 `/vrpn/<dog_tracker>/pose`、球 `/vrpn/ball/pose`、门 `/vrpn/goal_*/pose` | **零变换**（天然同系 world） |
| B（模板内建） | 模板 `absolute_coordination`（odom 位姿 + **每 5 s** VRPN 校准）→ `/absolute_pose/*` | 零（该节点已做对齐；需先启动它） |

> 实机现状（2026-10 实测）：`dog_coordination` / `absolute_coordination` **未在运行**，
> `/absolute_pose/*` 不存在 → 缺省走方案 A（`pose_source: vrpn`），
> 机体速度仍取 `/<dog>/odom_out`（body 系，与坐标系无关）。

**B 的标定方法**：把狗放到动捕系一个已知点 (x_m, y_m) 且已知朝向 θ_m，读此刻 odom
位姿 (x_o, y_o, θ_o)，则 `odom_to_world_yaw = θ_m − θ_o`，
`odom_to_world_xy = [x_m, y_m] − R(θ_m−θ_o)·[x_o, y_o]`。

**注意**：`mocap_frame_offset_xy/mocap_yaw_offset` 只作用于**球**，仅在"球与狗不同源"
时微调用；方案 A 下保持 0。

## 6. ⚠️ 已知缺失与前置条件（联调前必读）

1. **`last_action`（12 维）在实机上没有直接对应量**：训练时该通道是冻结低层的
   12 维动作输出；出厂控制器内部动作不可读。当前实现用
   `(q − q_default)/0.5` 作代理（`last_action_source: joint_deviation`）。若行为
   异常，建议在仿真侧做一次「去 last_action 通道」的消融重训，再部署无耦合版本。
2. **关节状态：高层接口无、底层通道有（已验证）**。高层接口（`cyberdog_loco_cn`
   §2）共 5 个 LCM 通道——robot_control_cmd / robot_control_response /
   motor_temperature / user_gait_result / global_to_robot——**不含关节反馈**；
   VP 协议能力清单亦无 JointState。但底层通道 ``spi_data``（udpm 7667）含
   q/qd/tau（字段序见 `lcm_sources.py` 头注），且**只订阅不发布 `motor_ctrl`
   即可安全读取**（进入电机模式只取决于是否发布 motor_ctrl）。
   → 本包已实现 `joint_source: lcm_spi` + `odom_source: lcm_localization`
   （只读线程），**无需重训即可拿到完整 55 维 obs**。
   备选：仿真侧「19 维部署观测消融」（去 q/qd/last_action）——若不想依赖底层通道。
3. **动捕面已按模板对齐（VRPN）**：`/absolute_pose/ball` 来自模板的
   `absolute_coordination`（VRPN → 世界系）。仍需做一次**场地标定**：设定
   `goal_center_xy`（球门在世界系坐标）并核对狗/球坐标一致。
4. **机器人自身位姿**：用出厂里程计（腿式里程计，50Hz，会缓慢漂移）。长时间运行
   建议给狗也贴动捕 marker 替代 odom。
5. **域间隙**：上层策略训练时的低层是我们自己的速度跟踪器（kp25/d0.5 DCMotor
   语义），出厂控制器动力学不同——尤其低速/转向响应。**联调第一步应做"速度命令
   直通"对照**（直接发固定 vx=0.3 看实际速度），确认出厂控制器的响应曲线。
6. **朝向约定是硬约束**：训练中狗固定出生并面朝进攻轴（+y → 球门）。实机摆放
   必须保持「狗背对自家半场、面朝球门、球在狗与门之间」的几何，否则 obs 语义错位。
7. **时间对齐**：各源频率不同（关节 200–500Hz / odom 50Hz / 动捕 100–300Hz），
   当前用「最新值 + 超时门」。更精细的时间戳插值对齐是后续工作。
8. **安全**：除本节点看门狗（关节/里程计/动捕超时 → 零速）外，必须保留物理急停与
   场地软限位（训练中的 `robot_out_of_field` 判据在实机上没有对应，建议加激光/
   动捕围栏触发 `/kick_sim2real/estop`）。

## 7. 推荐联调顺序

1. 台架/吊起：dry_run 跑 10 分钟，确认 obs 数值合理（`projected_gravity≈(0,0,-1)`、
   `ball_rel_b` 与实际球位一致）
2. 速度直通：绕过策略，手发 `vx=0.2/0.5`，标定出厂控制器响应与死区
3. 低速策略：`vel_limit` 暂时降到 `[0.4, 0.2, 0.6]` 上线，观察是否"贴球不动"
4. 全场：恢复训练 vel_limit，记录进球率与失效模式
