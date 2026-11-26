# 实机接口清单 —— CyberDog2（cyberdog2-69）

> 实测日期 2026-10-10，机器 `cyberdog2-69`（10.0.0.69，用户 `mi`，主机名 `mi-desktop`，
> Jetson aarch64）。本文档记录本机 **ROS2 话题 / LCM 通道 / VRPN 动捕** 的完整接口面，
> 供 sim2real 部署与后续开发使用。所有频率均为实测值。

## 0. 环境

| 项 | 值 |
|---|---|
| ROS2 | Galactic（`/opt/ros2/galactic`）+ 小米扩展（`/opt/ros2/cyberdog`） |
| 环境文件 | `source /etc/mi/ros2_env.conf`（ROS_DOMAIN_ID=**42**，RMW=**cyclonedds**，`CYCLONEDDS_URI=file:///etc/mi/cyclonedds.xml`） |
| 机器人命名空间 | **`/gugugaga`**（见 §6：被上一手硬编码，出厂逻辑应为 `mi_desktop_48_b0_2d_5f_be_3a`） |
| 出厂 bringup | systemd `cyberdog_bringup.service` + `cyberdog_sudo.service`（`main.launch.py` + `cyberdog_sudo.launch.py`） |
| 动捕 | VRPN 服务器 `10.0.0.252:3883`，frame=`world`，约 100–130 Hz |
| 动捕接收节点 | `vrpn_listener`（源码 `/home/mi/ros2_ws/src/demo_python_pkg/vrpn_client_ros2/src/vrpn_listener`；启动：`source .../src/install/local_setup.bash && ros2 launch vrpn_listener sync_entity_state.launch`） |

## 1. ROS2 话题（实测）

| 话题 | 类型 | 频率 | 发布/订阅 | 用途 |
|---|---|---|---|---|
| `/gugugaga/cmd_vel` | `geometry_msgs/msg/Twist` | —（命令面） | 订阅：`velocity_adaptor` | **速度命令入口**（Twist → MotionServoCmd 桥） |
| `/gugugaga/motion_servo_cmd` | `protocol/msg/MotionServoCmd` | —（命令面，6 个发布者） | 订阅：`motion_manager` | **运控伺服命令**（模板同款） |
| `/gugugaga/motion_servo_response` | `protocol/msg/MotionServoResponse` | 随命令 | 发布：`motion_manager` | 命令回执 |
| `/gugugaga/motion_status` | `protocol/msg/MotionStatus` | **~9.7 Hz** | 发布：`motion_manager` | 当前 motion_id / contact / switch_status / 错误 |
| `/gugugaga/odom_out` | `nav_msgs/msg/Odometry` | **47 Hz** | 发布：`leg_odom_publisher` | 腿式里程计（`twist.linear` 为**机体系**速度） |
| `/gugugaga/joint_states` | `sensor_msgs/msg/JointState` | **无发布者** | 订阅：`robot_state_publisher` | 仅类型占位——本机无节点发布 |
| `/gugugaga/scan` | `sensor_msgs/msg/LaserScan` | — | — | 激光雷达 |
| `/gugugaga/dog_pose` | `geometry_msgs/msg/PoseStamped` | — | — | 位姿（未采样到发布者） |
| `/gugugaga/pose_filtered` / `pose_enable` / `tracking_pose_transformed` | `PoseStamped` / `Bool` / `PoseStamped` | — | — | 位姿滤波/开关（当前无发布者） |
| `/gugugaga/dog_sn` | `std_msgs/msg/String` | — | — | 序列号 |
| `/vrpn/<name>/pose|twist|accel` | `PoseStamped` / `TwistStamped` / `AccelStamped` | **~120 Hz** | 发布：`vrpn_listener` | 动捕（见 §5） |

ROS2 服务（相关）：`/gugugaga/motion_custom_cmd`、`/gugugaga/motion_manager/*`（参数服务）。

## 2. LCM 通道（实测，UDP 组播 `udpm://239.255.76.67:<port>?ttl=255`）

| 端口 | 通道 | 频率 | payload | 内容 |
|---|---|---|---|---|
| 7667 | **`leg_control_data`** | **50.0 Hz** | 392 B | **关节 q[12]/qd[12]** + 足端 p[12]/v[12] + tau_est/force_est/force_desired + 限位 |
| 7667 | `external_imu` | **500.7 Hz** | 100 B | IMU（高速） |
| 7667 | `global_to_robot` | 50.0 Hz | 76 B | 里程计（xyz/vxyz/rpy/omegaBody/vBody/timestamp） |
| 7667 | `robot_control_response` | 50.0 Hz | 64 B | 运控状态（switch_status 等） |
| 7667 | `motion_control_cmd` | 50.0 Hz | 124 B | ROS2→运控指令桥 |
| 7667 | `main_visualization` | 60.0 Hz | 100 B | 可视化 |
| 7667 | `motor_temperature` | 1.0 Hz | 56 B | 电机温度 |
| 7669 | `state_estimator` | 50.3 Hz | 192 B | 状态估计（IMU/接触/速度） |
| 7670 | `robot_control_response` | 50.3 Hz | 64 B | 同上（镜像） |
| 7671 | —（静默） | — | — | 发送命令时才出现（robot_control_cmd / user_gait_result） |

> 注意：底层电机模式专用通道 `spi_data` / `motor_ctrl_state` / `motor_ctrl` 在本机
> **常规运控模式下不广播（0 Hz）**——关节请用 `leg_control_data`。

## 3. 关键消息字段

### 3.1 `protocol/msg/MotionServoCmd`（ROS2 速度伺服命令）

```
int32     motion_id      # 步态 id：303=slow(0.65/0.3/1.25) 308=mid(1.0/0.3/1.25) 305=fast(1.6/0.55/2.5)
int32     cmd_type       # 0=SERVO_START 1=SERVO_DATA 2=SERVO_END
int32     cmd_source     # 0=APP 1=AUDIO ...
int32     value
float32[] vel_des        # [vx, vy, wz]  m/s, m/s, rad/s
float32[] rpy_des        # [roll, pitch, yaw] rad
float32[] pos_des        # 质心位置偏移 m
float32[] acc_des; float32[] ctrl_point; float32[] foot_pose; float32[] step_height
```

### 3.2 `protocol/msg/MotionStatus`（~9.7 Hz 状态）

```
int32 motion_id; int8 contact; int8 order_process_bar; int8 switch_status;
int8 ori_error; int16 footpos_error; int32[] motor_error
# switch_status: 0=NORMAL 1=TRANSITIONING 2=ESTOP 3=EDAMP 4=LIFTED
#                5=BAN_TRANS 6=OVER_HEAT 7=LOW_BAT 8=ORI_ERR 9=FOOTPOS_ERR 10=STAND_STUCK
```

### 3.3 LCM `leg_control_data`（**关节反馈，50 Hz**）

布局：**8 B LCM 指纹 + 大端 float 字段**、末尾 3×int32 限位；腿序 **FR,FL,RR,RL**，
每腿 **(abad, hip, knee)**；符号惯例 **abad +，hip −，knee −**：

```
偏移  字段                说明
  8   float q[12]          关节角 (rad, 原始读取)
 56   float qd[12]         关节角速度 (rad/s)
104   float p[12]          足端位置 (m, 相对髋)
152   float v[12]          足端速度 (m/s)
200   float tau_est[12]    估计力矩 (Nm)
248   float force_est[12]  估计足力 (N)
296   float force_desired[12]
344   int32 q_abad_limit[4]; q_hip_limit[4]; q_knee_limit[4]
```

实测示例（狗静止站立）：`q ≈ [0.02,-1.13,2.57 | 0.02,-1.12,2.57 | -0.20,-1.14,2.57 | 0.25,-1.12,2.55]`，`qd ≈ 0`。

### 3.4 LCM `global_to_robot`（`localization_lcmt`，50 Hz）

```
float xyz[3]; vxyz[3]; rpy[3]; omegaBody[3]; vBody[3]; int64 timestamp
```
其中 `vBody/omegaBody` 为**机体系**速度/角速度（可直接喂策略 obs）。

### 3.5 LCM `robot_control_response`（50 Hz）

```
int8 mode; int8 gait_id; int8 contact; int8 order_process_bar; int8 switch_status;
int8 ori_error; int16 footpos_error; int32 motor_error[12]
```

### 3.6 LCM `state_estimator`（50 Hz，端口 7669）

```
float p[3]; vWorld[3]; vBody[3]; p_abs[3]; vWorld_abs[3]; vBody_abs[3]; vRemoter[3];
float rpy[3]; omegaBody[3]; omegaWorld[3]; quat[4]; aBody[3]; aWorld[3];
float contactEstimate[4]; int64 timestamp
```

## 4. 动捕（VRPN，`/vrpn/<tracker>/pose|twist|accel`，~120 Hz，frame=world）

| tracker | 说明 | 实测 (x, y, z) |
|---|---|---|
| **`GEXY2`** | **本机（cyberdog2-69）** | (-1.84, 1.41, 0.13) |
| `ball` | 球 | (-0.19, 4.58, 0.09) |
| `goal_left` | 球门（一端） | (-0.11, 4.83, 0.97) |
| `goal_right` | 球门（另一端） | (0.44, -4.25, 1.09) |
| 其他狗 | `GEXY1` / `Lizj_dog` / `wcy_dog` / `xll_dog` / `zbj_dog` / `Sue_dog` | — |
| 其他刚体 | `Rigid0/3/11/12` / `VRPN_Control` | — |

场地为**双门**（两门在场地两端，y≈+4.8 / −4.3）；部署时需选定进攻门。

## 5. 命名空间 `gugugaga` 的来历（重要）

出厂 `get_namespace()` 本应返回 `主机名_eth0 MAC`（转义非字母数字为 `_`）：

```python
# /opt/ros2/cyberdog/lib/python3.6/site-packages/mi/cyberdog_bringup/manual.py:166-171
def get_namespace():
    hostname = socket.getfqdn(socket.gethostname())
    mac = get_mac('eth0')
    namespace = hostname + '_' + mac
    namespace = re.sub('[^0-9a-zA-Z]+', '_', namespace)
    return "gugugaga"        # ← 出厂应为 return namespace；被上一手写死
```

- 本机的"正常值"应为 **`mi_desktop_48_b0_2d_5f_be_3a`**（hostname `mi-desktop` + eth0 MAC `48:b0:2d:5f:be:3a`）
- 两份拷贝都被改：`/opt/ros2/cyberdog/lib/python3.6/site-packages/mi/cyberdog_bringup/manual.py`（实际导入）与 `/opt/ros2/cyberdog/share/cyberdog_bringup/bringup/manual.py`
- 恢复 = 两处改回 `return namespace` + 重启 `cyberdog_bringup` / `cyberdog_sudo`；**所有话题会改名**，写死 `/gugugaga` 的脚本会断（本仓库部署代码用 `dog_name` 参数适配两者）

## 6. 枚举与解码方法（可复现）

```bash
# ROS2 话题清单（含类型）
source /etc/mi/ros2_env.conf
ros2 topic list -t

# LCM 通道枚举（UDP 组播嗅探；LCM 短包 = "LC02" + seqno + 通道名\0 + 8B 指纹 + 字段）
python3 - <<'EOF'
import socket, struct, time
G='239.255.76.67'
for port in (7667, 7669, 7670, 7671):
    s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('', port))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 struct.pack('4s4s', socket.inet_aton(G), socket.inet_aton('0.0.0.0')))
    s.settimeout(0.5); stats={}; t0=time.time()
    while time.time()-t0 < 3:
        try: data,_=s.recvfrom(70000)
        except socket.timeout: continue
        if data[:4]==b'LC02':
            nul=data.find(b'\x00',8); ch=data[8:nul].decode('utf-8','replace')
            stats[ch]=stats.get(ch,0)+1
    print(port, {k: round(v/3,1) for k,v in stats.items()})
EOF

# 关节解码（leg_control_data，大端）
#   见 ros2_ws/src/kick_sim2real/kick_sim2real/lcm_sources.py::decode_leg_control_data
```

## 7. 对 sim2real 部署的直接结论

| 需求 | 接口 | 配置 |
|---|---|---|
| 球 / 门 / 狗位姿（世界系） | `/vrpn/{ball,goal_*,GEXY2}/pose` ~120 Hz | `pose_source: vrpn`，`dog_tracker: GEXY2` |
| 关节 q/qd | LCM `leg_control_data` 50 Hz（只读） | `joint_source: lcm_leg` |
| 机体速度/陀螺 | `/gugugaga/odom_out`（47 Hz，机体系）或 LCM `global_to_robot`（50 Hz） | `odom_source: ros` |
| 速度命令输出 | `protocol/msg/MotionServoCmd` → `/{ns}/motion_servo_cmd`（或 Twist → `/{ns}/cmd_vel`） | `backend: motion_servo`，`dog_name: gugugaga` |
| 安全状态 | `robot_control_response.switch_status` / `motion_status` | 内建 FSM |

坐标系：全部世界量同取 VRPN（frame=`world`），**零变换**（obs 全为机体系相对量，
全局平移+偏航自动抵消，单测 `test_global_transform_invariance` 验证）。
