# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .cartpole_env_cfg import CartpoleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    # IMPORTANT: 这个kwargs参数会传递给register，其内部会调用EnvSpec配置类保存所有的参数，具体到kwargs会通过**kwargs先解包再传递给EnvSpec
    # IMPORTANT: env_cfg_entry_point是必须要有的参数，因为在加载环境时，parse_env_cfg函数会根据这个参数来加载环境配置
    kwargs={
        # 配置入口点可以是YAML文件或者python配置类
        "env_cfg_entry_point": CartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # 如果使用rsl-rl的话，至少在这个isaac lab中，必须要用rsl_rl_cfg_entry_point这个键同时其值必须是一个配置类
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
