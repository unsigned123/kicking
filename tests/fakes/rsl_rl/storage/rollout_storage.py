"""fake rsl_rl.storage —— 5.1.0 RolloutStorage.Transition 复刻 + 存储外壳。

离线测试由 test_sonic_ppo_loss.py 自建 mini-batch 假存储；这里只提供
``RolloutStorage.Transition``（SonicPPO 构造时 ``self.transition = ...`` 用到）
与最小外壳。
"""
from __future__ import annotations


class RolloutStorage:
    """5.1.0 存储外壳（离线桩：只实现 update() 依赖的属性与 Transition）。"""

    class Transition:
        """Storage for a single state transition（5.1.0 字段面复刻）。"""

        def __init__(self) -> None:
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.distribution_params = None
            self.privileged_actions = None
            self.hidden_states = (None, None)

        def clear(self) -> None:
            self.__init__()

    def __init__(self, *args, **kwargs) -> None:
        # 构造签名对齐（num_envs / num_transitions_per_env / obs / ...）
        # 离线测试不真正填充存储，mini_batch_generator 由测试侧伪存储替换。
        pass

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        raise NotImplementedError(
            "fake RolloutStorage.mini_batch_generator 未被使用："
            "test_sonic_ppo_loss.py 提供言 mini-batch 伪存储")

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        raise NotImplementedError(
            "fake RolloutStorage.recurrent_mini_batch_generator 未被使用")

    def clear(self) -> None:
        pass