# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

# IMPORTANT: mdp模块对应于马尔科夫决策过程的实现，其中就定义了action、observation、reward、termination、event等概念
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg: # ActionsCfg好像并不需要继承于任何东西
    """Action specifications for the environment."""

    # mdp模块中定义了action的一些配置，除了effort之外，还有速度和位置的控制
    # IMPORTANT: 注意，这里的asset_name并不是prim_path中的内容，而是基于ArticulationCfg生成的asset_name，如在CartpoleSceneCfg中定义的robot
    # 其实还是可以定义其他的action，可以将一个机器人的控制分成多个控制组，这样可以更灵活地控制机器人
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup): # 观测组的配置需要继承于ObservationGroupCfg，这里是因为将ObservationGroupCfg重命名为ObsGroup
        """Observations for policy group."""

        # observation terms (order preserved)，自定义的观测项
        # 观测项的定义要通过ObservationTermCfg，这里是因为ObservationTermCfg重命名为ObsTerm
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) # 通过ObsTerm定义观测项，其中又用到了mdp模块中的函数
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None: # dataclass中定义的特殊方法，在初始化(运行__init__)之后自动运行，-> None表示返回值为None，不返回任何值
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    # IMPORTANT: 官方文档提到，观测组的名称必须要有一个名为policy的观测组，因为这个观测组是被wrappers（环境包装器）必须要访问的
    # NOTE: 环境包装器wrappers好像只会访问名为policy的观测组，其他的观测组不会被访问
    policy: PolicyCfg = PolicyCfg() # 仅定义了一个名为policy的观测组，其实还可以定义其他的观测组


# Event事件负责对应模拟环境中的事件，包括场景初始化、随机化物理特性以及改变视觉特性等
@configclass
class EventCfg:
    """Configuration for events."""
    # IMPORTANT: 可以看出，所有的Event都是通过EventTermCfg定义的，这里是因为EventTermCfg重命名为EventTerm
    # 在EventTermCfg内置了三种mode，分别是startup、reset和interval，分别对应于环境初始化、重置和间隔
    # reset是环境终止或重置时发生的事件
    # interval是每隔一定步数执行一次的事件，应该是要配合interval_range_s参数一起用吧
    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={ # IMPORTANT: 这里的params是传递给上面定义的func函数的参数，基于键值对的形式
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg): # 设置总的环境管理配置类，这个类需要继承于ManagerBasedEnvCfg
    # NOTE: 传入这里面的也都是各种配置类
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=32, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self): # dataclass中定义的特殊方法，在初始化(运行__init__)之后自动运行
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0] # 配置场景相机
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        # decimation参数应该是表示多少个sim步长更新一次控制动作
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        # 这里设置物理仿真的事件步长
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    # NOTE: envs.ManagerBasedEnv类没有任何终止概念，因此用户需要为环境定义终止/重置条件
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg() # 实例化总的环境管理配置类
    env_cfg.scene.num_envs = args_cli.num_envs # 可以通过命令行来修改环境的个数
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg) # 创建环境管理器，将之前定义的环境管理配置类对象传入

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode(): # 是Pytorch用于禁用梯度计算和其他开销的上下文管理器，是torch.no_grad()的更高效版本
            # reset
            if count % 300 == 0:
                count = 0
                env.reset() # 重置所有的环境并返回初始观测值
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts) # 这里的step应该会自动按照设定好的decimation参数来更新控制动作
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
