# Jetson 端部署（TensorRT 推理）

机器狗 NX/Jetson 上是 **python3.6 + TensorRT 8.0.1.6 + CUDA 10.2**，没有 torch。
策略（55→128→128→128→3 MLP）导出 ONNX 后在机上用 `trtexec` 转 FP16 引擎，
用本目录脚本推理（ctypes 调 libcuda + tensorrt python，无需 pycuda）。

## 1. 导出 ONNX（任意有 torch 的机器）

```bash
python3 - <<'PY'
import torch
from torch import nn
blob = torch.load('model_1999.pt', map_location='cpu', weights_only=False)
st = blob['actor_state_dict']
keys = sorted([k for k in st if k.startswith('mlp.') and k.endswith('.weight')],
              key=lambda k: int(k.split('.')[1]))
layers = []
for i, k in enumerate(keys):
    idx = int(k.split('.')[1])
    lin = nn.Linear(int(st[k].shape[1]), int(st[k].shape[0]))
    with torch.no_grad():
        lin.weight.copy_(st[k]); lin.bias.copy_(st[f'mlp.{idx}.bias'])
    layers.append(lin)
    if i < len(keys) - 1:
        layers.append(nn.ELU())
net = nn.Sequential(*layers).eval()
torch.onnx.export(net, torch.zeros(1, 55), 'kick_policy.onnx', opset_version=11,
                  input_names=['obs'], output_names=['cmd'], dynamo=False)
PY
scp kick_policy.onnx cyberdog2-69:/home/mi/
```

## 2. 机上构建引擎

```bash
/usr/src/tensorrt/bin/trtexec --onnx=/home/mi/kick_policy.onnx \
    --saveEngine=/home/mi/kick_policy_fp16.trt --fp16 --workspace=64
```
实测：87.9 KB 引擎，推理 0.097 ms/次（50 Hz 用不到 1% 算力）。

## 3. 运行

```bash
cd /home/mi
source /etc/mi/ros2_env.conf          # 必须：ROS_DOMAIN_ID=42 / CycloneDDS / rclpy
python3 kick_policy_jetson.py --dry-run            # 只看 obs/命令，不发布
python3 kick_policy_jetson.py --vel-limit 0.3,0.15,0.5   # 首试低速
```

参数：`--dog-tracker GEXY2`（本机）、`--ball-tracker ball`、
`--goal-tracker goal_left`（或 `--goal-xy x,y`）、`--dry-run`。

## 4. 安全前置清单（必须逐条确认）

1. **狗必须站立**（`odom_out` 的 z ≈ 0.23-0.28 m；趴下时 z ≈ 0.05-0.06 m 不要发速度命令）
2. **断开手机 APP 连接**（避免指令冲突）
3. **狗朝向进攻门**（训练中球门在正前方；实机摆位需一致）
4. **周围无人/无障碍**，有人站在狗侧后方可随时抓住或断电
5. 首试一律 `--vel-limit 0.3,0.15,0.5`，确认行为正常后再放开
6. 中途异常：Ctrl+C（脚本退出前会自动补发零速）/ 直接断开或断电

## 5. 已知校准项

- **动捕 tracker 安装偏置**：站立后检查 dry-run 输出的 `grav` 是否 ≈ `(0,0,-1)`；
  若存在恒定倾斜，需标定 tracker 安装旋转（或临时只取 yaw、置零 roll/pitch）
- **关节符号/腿序**：`leg_control_data` 实测 FR/FL/RR/RL × (abad,hip,knee)、符号 (+,-,-)；
  若首试发现异常（如站立时 q_rel 抖动异常），核对符号表
