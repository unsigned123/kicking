"""离线测试用的 rsl_rl 5.1.0 纯 torch 桩化覆盖（本地 pytest 专用，不上生产）。

服务器（env_isaaclab）有真实 rsl_rl 5.1.0（含 tensordict 依赖），本包只在
**本地离线**（无 tensordict / 无 rsl_rl 时）通过 conftest 插到 sys.path 供
``import rsl_rl.*`` 解析。桩实现忠实复刻 5.1.0 被本工程使用的接口（源码逐个
核对：rsl_rl 5.1.0 site-packages），仅剥掉 tensordict 依赖。

覆盖的接口面：
  - modules: MLP / EmpiricalNormalization / Distribution / GaussianDistribution
  - models:  MLPModel（forward/get_latent/update_normalization/_get_obs_dim 等）
  - algorithms: PPO（__init__ 超参 + update 依赖的属性，不含 rollout 循环）
  - runners: OnPolicyRunner（仅 import/构造面属性存储；learn 循环不在本地测）
  - utils: resolve_callable / resolve_optimizer / resolve_nn_activation /
           get_param / unpad_trajectories / check_nan / ...
"""
from __future__ import annotations