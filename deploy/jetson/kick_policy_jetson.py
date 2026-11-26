#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kick_policy_jetson — 在 CyberDog2（Jetson, python3.6）上用 TensorRT 跑上层策略。

数据源
  - 关节 q/qd : LCM `leg_control_data`（udpm 7667, 50 Hz, 大端；只读）
  - 机体速度   : LCM `global_to_robot`（vBody / omegaBody, 50 Hz）
  - 狗/球/门位姿: ROS2 `/vrpn/<tracker>/pose`（PoseStamped, frame=world）
输出
  - 速度命令   : ROS2 `protocol/msg/MotionServoCmd` → `/{ns}/motion_servo_cmd`
                 （motion_id 303/308/305 按速度自动选步态）

安全
  - --dry-run 只打印不发布
  - 关节/位姿/球超时 → 零速；switch_status ∈ {2,3,4} → 零速
  - --vel-limit 限幅（缺省与训练命令范围一致 1.6/0.55/2.5）

用法（机器人上）
  python3 kick_policy_jetson.py --dry-run
  python3 kick_policy_jetson.py --vel-limit 0.4,0.2,0.6      # 首试低速
"""
import argparse
import ctypes
import struct
import threading
import time
from collections import deque

import numpy as np
import tensorrt as trt

# --------------------------------------------------------------------------- #
# 契约常量（与训练侧逐位一致；见 docs/SIM2REAL_INTERFACES.md）
# --------------------------------------------------------------------------- #
DIST_NORM = 3.0
SCALE_JOINT_VEL = 0.05
SCALE_BASE_ANG_VEL = 0.2
OFFICIAL_ACTION_SCALE = 0.5
NS = "gugugaga"

# 训练序（PhysX BFS）与 SDK 序（FR/FL/RR/RL × abad/hip/knee）
TRAIN_JOINTS = ["FR_abad", "FL_abad", "RR_abad", "RL_abad",
                "FR_hip", "FL_hip", "RR_hip", "RL_hip",
                "FR_knee", "FL_knee", "RR_knee", "RL_knee"]
SDK_JOINTS = ["FR_abad", "FR_hip", "FR_knee",
              "FL_abad", "FL_hip", "FL_knee",
              "RR_abad", "RR_hip", "RR_knee",
              "RL_abad", "RL_hip", "RL_knee"]
PERM = [SDK_JOINTS.index(n) for n in TRAIN_JOINTS]          # train[i]=sdk[PERM[i]]
LEG_SIGN = [1.0, -1.0, -1.0] * 4                           # SDK: abad +, hip -, knee -
DEFAULT_Q = np.array([-0.10, 0.10, -0.10, 0.10,            # 训练序，cyberdog2 快照
                      0.80, 0.80, 1.00, 1.00,
                      -1.50, -1.50, -1.50, -1.50], dtype=np.float64)

GAITS = {"slow": {"id": 303, "lim": (0.65, 0.30, 1.25)},
         "mid":  {"id": 308, "lim": (1.00, 0.30, 1.25)},
         "fast": {"id": 305, "lim": (1.60, 0.55, 2.50)}}


def select_gait(vx, vy, wz):
    ax, ay, az = abs(vx), abs(vy), abs(wz)
    name = "fast"
    for cand in ("slow", "mid"):
        lim = GAITS[cand]["lim"]
        if ax <= lim[0] and ay <= lim[1] and az <= lim[2]:
            name = cand
            break
    lim = GAITS[name]["lim"]
    return name, np.array([np.clip(vx, -lim[0], lim[0]),
                           np.clip(vy, -lim[1], lim[1]),
                           np.clip(wz, -lim[2], lim[2])])


# --------------------------------------------------------------------------- #
# 几何（与训练侧同义；quat xyzw）
# --------------------------------------------------------------------------- #
def quat_apply_inverse(q, v):
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    ix, iy, iz, iw = -x, -y, -z, w
    tx = 2.0 * (iy * v[2] - iz * v[1])
    ty = 2.0 * (iz * v[0] - ix * v[2])
    tz = 2.0 * (ix * v[1] - iy * v[0])
    return np.array([v[0] + iw * tx + (iy * tz - iz * ty),
                     v[1] + iw * ty + (iz * tx - ix * tz),
                     v[2] + iw * tz + (ix * ty - iy * tx)])


def yaw_from_quat(q):
    x, y, z, w = np.asarray(q, dtype=np.float64) / (np.linalg.norm(q) + 1e-8)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def ball_rel_b(robot_xy, ball_xy, quat):
    d = np.array([ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1], 0.0])
    rel = quat_apply_inverse(quat, d)[:2] / DIST_NORM
    return np.array([rel[0], rel[1], float(np.linalg.norm(rel))])


def goal_rel_b(robot_xy, quat, goal_xy):
    d = np.array([goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1], 0.0])
    rel = quat_apply_inverse(quat, d)[:2] / DIST_NORM
    ang = float(np.arctan2(rel[1], rel[0]))
    return np.array([rel[0], rel[1], np.sin(ang), np.cos(ang)])


def push_geometry_b(robot_xy, ball_xy, quat, goal_xy):
    btog = np.array([goal_xy[0] - ball_xy[0], goal_xy[1] - ball_xy[1]])
    btog = btog / (np.linalg.norm(btog) + 1e-8)
    tob = np.array([ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1]])
    tob = tob / (np.linalg.norm(tob) + 1e-8)
    yaw = yaw_from_quat(quat)
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    return np.array([float(np.dot(tob, btog)),
                     float(fwd[0] * btog[1] - fwd[1] * btog[0]),
                     float(np.dot(fwd, btog))])


def build_obs(q_train, qd_train, gyro, quat, last_action, v_body,
              robot_xy, ball_xy, goal_xy):
    q = np.asarray(q_train, dtype=np.float64)
    qd = np.asarray(qd_train, dtype=np.float64)
    return np.concatenate([
        q - DEFAULT_Q,
        qd * SCALE_JOINT_VEL,
        np.asarray(gyro, dtype=np.float64) * SCALE_BASE_ANG_VEL,
        quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0])),
        np.asarray(last_action, dtype=np.float64),
        np.asarray(v_body, dtype=np.float64),
        ball_rel_b(robot_xy, ball_xy, quat),
        goal_rel_b(robot_xy, quat, goal_xy),
        push_geometry_b(robot_xy, ball_xy, quat, goal_xy),
    ]).astype(np.float32)


# --------------------------------------------------------------------------- #
# LCM 关节/里程计（只读，不切电机模式）
# --------------------------------------------------------------------------- #
def decode_leg_control_data(data):
    q = struct.unpack_from(">12f", data, 8)
    qd = struct.unpack_from(">12f", data, 8 + 48)
    q_s = [q[i] * LEG_SIGN[i] for i in range(12)]
    qd_s = [qd[i] * LEG_SIGN[i] for i in range(12)]
    return (np.array([q_s[i] for i in PERM]),
            np.array([qd_s[i] for i in PERM]))


class LcmReader(object):
    def __init__(self, on_joints, on_odom):
        import lcm
        self._lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self._on_joints = on_joints
        self._on_odom = on_odom
        self._lcm.subscribe("leg_control_data", self._jcb)
        self._lcm.subscribe("global_to_robot", self._ocb)
        t = threading.Thread(target=self._run)
        t.daemon = True
        t.start()

    def _run(self):
        while True:
            try:
                self._lcm.handle()
            except Exception:
                time.sleep(0.1)

    def _jcb(self, ch, data):
        try:
            q, qd = decode_leg_control_data(data)
            self._on_joints(q, qd)
        except Exception:
            pass

    def _ocb(self, ch, data):
        # localization_lcmt: xyz(3) vxyz(3) rpy(3) omegaBody(3) vBody(3) ts
        try:
            off = 8
            xyz = struct.unpack_from(">3f", data, off)
            rpy = struct.unpack_from(">3f", data, off + 24)
            omega = struct.unpack_from(">3f", data, off + 36)
            vbody = struct.unpack_from(">3f", data, off + 48)
            self._on_odom(np.array(xyz), np.array(vbody), np.array(omega),
                          np.array(rpy))
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# TensorRT 推理（ctypes 调 libcuda）
# --------------------------------------------------------------------------- #
class TrtPolicy(object):
    def __init__(self, engine_path):
        self._cuda = ctypes.CDLL("libcuda.so.1")
        c = self._cuda
        c.cuInit.argtypes = [ctypes.c_uint]
        c.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        c.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.c_uint, ctypes.c_int]
        c.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_size_t]
        c.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t]
        c.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_size_t]
        for fn in ("cuInit", "cuDeviceGet", "cuCtxCreate_v2", "cuMemAlloc_v2",
                   "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2"):
            getattr(c, fn).restype = ctypes.c_int

        def ck(ret):
            if ret != 0:
                raise RuntimeError("CUDA error %d" % ret)

        ck(c.cuInit(0))
        dev = ctypes.c_int()
        ck(c.cuDeviceGet(ctypes.byref(dev), 0))
        ctx = ctypes.c_void_p()
        ck(c.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_binding_name(i)
                      for i in range(self.engine.num_bindings)]
        self.ptrs = {}
        for i, name in enumerate(self.names):
            shape = tuple(self.engine.get_binding_shape(i))
            size = int(np.prod(shape)) * 4
            p = ctypes.c_void_p()
            ck(c.cuMemAlloc_v2(ctypes.byref(p), size))
            self.ptrs[name] = (p, size)
        self.bindings = [int(self.ptrs[n][0].value) for n in self.names]
        self.in_name = self.names[0]
        self.out_name = self.names[1]
        self._obs = np.zeros((1, 55), dtype=np.float32)
        self._out = np.zeros((1, 3), dtype=np.float32)

    def act(self, obs):
        self._obs[0] = obs
        c = self._cuda
        c.cuMemcpyHtoD_v2(self.ptrs[self.in_name][0],
                          self._obs.ctypes.data_as(ctypes.c_void_p),
                          self._obs.nbytes)
        self.context.execute_v2(self.bindings)
        c.cuMemcpyDtoH_v2(self._out.ctypes.data_as(ctypes.c_void_p),
                          self.ptrs[self.out_name][0], self._out.nbytes)
        return self._out[0].astype(np.float64)


# --------------------------------------------------------------------------- #
# ROS2 节点
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", default="/home/mi/kick_policy_fp16.trt")
    p.add_argument("--dog-tracker", default="GEXY2")
    p.add_argument("--ball-tracker", default="ball")
    p.add_argument("--goal-tracker", default="goal_left")
    p.add_argument("--goal-xy", default=None,
                   help="固定门坐标 'x,y'（缺省用 VRPN goal-tracker）")
    p.add_argument("--vel-limit", default="1.6,0.55,2.5")
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--joint-timeout", type=float, default=0.10)
    p.add_argument("--pose-timeout", type=float, default=0.25)
    p.add_argument("--ball-timeout", type=float, default=0.30)
    a = p.parse_args()

    vel_limit = np.array([float(v) for v in a.vel_limit.split(",")])
    fixed_goal = None
    if a.goal_xy:
        fixed_goal = tuple(float(v) for v in a.goal_xy.split(","))

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from protocol.msg import MotionServoCmd, MotionStatus
    from rclpy.node import Node

    st = {"q": None, "qd": None, "t_joint": 0.0,
          "robot_xy": None, "quat": None, "v_body": np.zeros(3),
          "gyro": np.zeros(3), "t_odom": 0.0, "t_pose": 0.0,
          "ball": None, "t_ball": 0.0,
          "goal": np.array(fixed_goal) if fixed_goal else None, "t_goal": 0.0,
          "status": None}
    lock = threading.Lock()

    def on_joints(q, qd):
        st["q"], st["qd"], st["t_joint"] = q, qd, time.time()

    def on_odom(xyz, vbody, omega, rpy):
        st["v_body"] = vbody
        st["gyro"] = omega
        st["t_odom"] = time.time()

    LcmReader(on_joints, on_odom)

    rclpy.init()
    node = Node("kick_policy_jetson")
    pub = node.create_publisher(MotionServoCmd,
                                "/%s/motion_servo_cmd" % NS, 10)

    def pose_cb(key):
        def cb(msg):
            p, o = msg.pose.position, msg.pose.orientation
            now = time.time()
            with lock:
                st[key] = np.array([p.x, p.y])
                if key == "robot_xy":
                    st["quat"] = np.array([o.x, o.y, o.z, o.w])
                    st["t_pose"] = now
                elif key == "ball":
                    st["t_ball"] = now
                elif key == "goal":
                    st["t_goal"] = now

        return cb

    node.create_subscription(PoseStamped, "/vrpn/%s/pose" % a.dog_tracker,
                             pose_cb("robot_xy"), 10)
    node.create_subscription(PoseStamped, "/vrpn/%s/pose" % a.ball_tracker,
                             pose_cb("ball"), 10)
    if fixed_goal is None:
        node.create_subscription(PoseStamped, "/vrpn/%s/pose" % a.goal_tracker,
                                 pose_cb("goal"), 10)

    def status_cb(msg):
        st["status"] = int(msg.switch_status)

    node.create_subscription(MotionStatus, "/%s/motion_status" % NS,
                             status_cb, 10)

    policy = TrtPolicy(a.engine)
    print("[kick] engine loaded: %s (bindings=%s)" % (a.engine,
                                                      policy.names))
    print("[kick] dry_run=%s vel_limit=%s dog=%s ball=%s goal=%s" %
          (a.dry_run, vel_limit.tolist(), a.dog_tracker, a.ball_tracker,
           a.goal_tracker if fixed_goal is None else fixed_goal))
    print("[kick] 等待数据（关节 LCM / VRPN 位姿）...")

    warned = {"t": 0.0, "p": 0.0}

    def tick():
        now = time.time()
        with lock:
            q, qd = st["q"], st["qd"]
            robot_xy, quat = st["robot_xy"], st["quat"]
            ball = st["ball"]
            goal = st["goal"]
            t_j = st["t_joint"]
            t_o = min(st["t_odom"], st["t_pose"] or st["t_odom"])
            t_b = st["t_ball"]
            v_body, gyro = st["v_body"].copy(), st["gyro"].copy()
            status = st["status"]

        def block():
            if q is None or now - t_j > a.joint_timeout:
                return "joints stale"
            if robot_xy is None or quat is None or now - t_o > a.pose_timeout:
                return "pose/odom stale"
            if ball is None or now - t_b > a.ball_timeout:
                return "ball stale"
            if goal is None:
                return "goal missing"
            if status in (2, 3, 4):
                return "factory switch_status=%s" % status
            return None

        reason = block()
        if reason:
            if now - warned["t"] > 2.0:
                print("[kick] hold zero: %s" % reason)
                warned["t"] = now
            return None
        last_action = (q - DEFAULT_Q) / OFFICIAL_ACTION_SCALE
        obs = build_obs(q, qd, gyro, quat, last_action, v_body, robot_xy,
                        ball, goal)
        cmd = np.clip(policy.act(obs), -vel_limit, vel_limit)
        if a.dry_run and now - warned["p"] > 0.5:
            warned["p"] = now
            print("[dry-run] q_rel[:3]=%s qd_sc[:3]=%s grav=%s "
                  "ball=%s goal=%s cmd=%s" %
                  (np.round(obs[0:3], 3).tolist(),
                   np.round(obs[12:15], 3).tolist(),
                   np.round(obs[27:30], 3).tolist(),
                   np.round(obs[45:48], 3).tolist(),
                   np.round(obs[48:52], 3).tolist(),
                   np.round(cmd, 3).tolist()))
        return cmd

    def on_timer():
        cmd = tick()
        if cmd is None or a.dry_run:
            return
        gait, limited = select_gait(cmd[0], cmd[1], cmd[2])
        msg = MotionServoCmd()
        msg.motion_id = GAITS[gait]["id"]
        msg.cmd_type = 1
        msg.value = 2
        msg.vel_des = [float(limited[0]), float(limited[1]),
                       float(limited[2])]
        msg.step_height = [0.05, 0.05]
        pub.publish(msg)

    node.create_timer(1.0 / a.rate, on_timer)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if not a.dry_run:
                msg = MotionServoCmd()
                msg.motion_id = GAITS["slow"]["id"]
                msg.cmd_type = 1
                msg.value = 2
                msg.vel_des = [0.0, 0.0, 0.0]
                msg.step_height = [0.05, 0.05]
                pub.publish(msg)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
