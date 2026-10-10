"""hierarchical_kick.sim：kick-flat 上层 RL 的 isaaclab 面。

vendor 源：官方 IsaacLab v2.3.0 velocity 包谱系（逐文件 diff 记录见
各文件头）。

文件谱系：
  - ``mdp/``            官方 velocity mdp 最小集（rewards/curriculums
                        verbatim）+ upper 自建 func（survival/height 终止
                        + ball_mdp 球 env 全套）；
  - ``velocity_env_cfg.py``  官方 velocity_env_cfg.py verbatim（仅 mdp
                        import 面一处适配，diff 见文件头）；
  - ``flat_env_cfg.py`` 官方 go2 rough/flat 谱系 + Upper 适配层；
  - ``ball_env_cfg.py`` U-Kick 球 env（训练面 + oracle 面 + cmd 面）；
  - ``math_utils.py``   离线数学单源（quat 纯 torch；ball_mdp 降级路径）。

gym 注册面在 ``hierarchical_kick.tasks``（原 register.py 迁移）。
本子包 mdp/__init__ 顶层拉起 isaaclab——离线测试只触 core/ 与注册字符串
面（AST/常量/按路径加载），不拉起 mdp 子包。
"""
