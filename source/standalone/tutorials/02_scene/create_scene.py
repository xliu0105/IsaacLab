# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip


# 场景是由一系列带有自己配置的实体组成，这些都在继承于scene.InteractiveSceneCfg的配置类内指定，然后将配置类传给scene.InteractiveScene以创建场景
# InteractiveScene就是交互式场景管理接口
# IMPORTANT: 配置类中的变量名称用于从scene.InteractiveScene对象访问对应实体的键值。如可以用scene["cartpole"]访问cartpole实体
# 在下面这个配置的中，ground和dome_light是AssetBaseCfg类型，是不可交互的；而cartpole是ArticulationCfg类型，是执行器类型，是可交互的
# IMPORTANT: 在Scene配置类中，使用绝对路径的如地面，光源是不会被克隆到多个环境的，同时其也不能指定{ENV_REGEX_NS}这样的相对路径，我试了会报错，可能在系统底层，这些东西就是不允许被克隆的
# IMPORTANT: 而对于实体来说，如这里的cartpole，是可以被克隆到多个环境的，因此其prim_path可以指定{ENV_REGEX_NS}这样的相对路径，其会自动根据后面设置的环境个数处理成"/World/envs/env_{i}"这样的路径
# IMPORTANT: 如果在配置类中没有使用{ENV_REGEX_NS}这样的符号，在创建InteractiveScene对象指定多环境时，会报错
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    # 通过AssetBaseCfg创建对象，在01_assets/run_rigid_object.py中，通过RigidObjectCfg创建对象，这里的AssetBaseCfg是RigidObjectCfg的父类
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg( # 按照InteractiveSceneCfg注释的说法，这个light推荐是最后添加的吧
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # 这里的replace是dataclasses的一个方法，用于创建现有实例的副本，并可以选择性的修改某些属性，在configclass修饰类中基于dataclasses的replace自定义了replace方法
    # IMPORTANT: 这里的prim_path="{ENV_REGEX_NS}/Robot"是一个占位符，表示在创建InteractiveScene对象时，会根据num_envs参数的值替换这个占位符
    # 任何带有{ENV_REGEX_NS}变量的实体的prim路径在每个环境中都会被克隆，路径会被场景对象替换为"/World/envs/env_{i}"，其中i是环境的索引
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["cartpole"]  # IMPORTANT: 可以使用InteractiveScene对象从字典中获取场景元素
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_link_pose_to_sim(root_state[:, :7])
            robot.write_root_com_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            # write_joint_state_to_sim有四个参数，前两个是关节位置和速度，后两个是joint_ids和env_ids，用来指定要写入的关节和环境索引
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()  # IMPORTANT: reset函数有参数为Sequence[int]类型，用于指定要重置的环境索引，如果不传参数，则默认重置所有环境
            # IMPORTANT: 如果要在scene.reset()部分环境，则在write_root_state_to_sim和write_joint_state_to_sim中也要指定相同的环境索引以及对应shape的对局
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)  # 设置关节力矩是针对Articulation对象的方法
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    # 根据上面自定义的继承于InteractiveSceneCfg的CartpoleSceneCfg配置类创建场景
    # InteractiveSceneCfg配置类内有参数，如num_envs指定了环境副本数量，env_spacing指定了环境之间的间距
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)  # 将配置类传给InteractiveScene以创建场景
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
