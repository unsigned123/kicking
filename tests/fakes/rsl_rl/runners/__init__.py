"""fake rsl_rl.runners —— 5.1.0 OnPolicyRunner 的 import/构造面桩。

仅覆盖 ``LatentDecoderRunner``（models/decoder_runner.py）离线 import 与
构造所需：``__init__`` 存 env/cfg/device 属性（**不建 alg/logger**——
learn 循环体不在本地测，A100 真身跑）。
"""
from __future__ import annotations


class OnPolicyRunner:
    """OnPolicyRunner 占位（构造面：属性存储；无 alg/logger 构建）。"""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None,
                 device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.alg = None
        self.logger = None
        self.current_learning_iteration = 0
        self.is_distributed = False
        self.gpu_world_size = 1
        self.gpu_local_rank = 0
        self.gpu_global_rank = 0
