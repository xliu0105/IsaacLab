# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/add_sensors_on_robot.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
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
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    # IMPORTANT: 传感器也是定义在InteractiveSceneCfg的派生类中的，也是场景的一部分
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # IMPORTANT: 所有的传感器配置都继承于omni.isaac.lab.sensors中的传感器配置类，这些配置类都是SensorBaseCfg的派生类
    # IMPORTANT: 可以看出，camera的prim_path是在base基础上又有了一个front_cam，而RayCasterCfg和ContactSensorCfg都是写到了robot现有的实体上，如base或者.*_FOOT
    # sensors
    camera = CameraCfg(
        # IMPORTANT: 相机传感器会在场景中有一个对应的prim，因此会在指定的prim_path下生成一个相机的prim，而某些传感器会直接附加在现有的prim上，不会生成传感器对应的prim
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam", # NOTE: 传感器的prim_path
        update_period=0.1, # NOTE: 传感器的更新周期
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"], # camera的datatypes可以有多种类型，具体可以在camera.py文件中查看
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ), # IMPORTANT: 这里的spawn是一个PinholeCameraCfg的实例，生成相机的配置，主要是相机的具体参数
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        # NOTE: offset是相对于父prim的偏移量，这个offset的定义需要依赖于各个传感器的配置类内部的子类去定义，如CameraCfg.OffsetCfg或者RayCasterCfg.OffsetCfg
    )
    height_scanner = RayCasterCfg(
        # IMPORTANT: 高度扫描传感器不会生成对应的prim，而是直接附加在现有的prim上
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True, # 只关注高度信息，因此不需要考虑机器人的滚动和俯仰，因此这个设为true
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # 设置光线的属性
        debug_vis=True, # 是否要可视化光纤击中网格的点
        mesh_prim_paths=["/World/defaultGroundPlane"], # 射线要投射的mesh列表，目前只支持单个静态mesh
    )
    # NOTE: 接触传感器依赖于PhysX的接触报告，因此需要在资产配置中将activate_contact_sensors设为true，即要在ArticulationCfg设置中将activate_contact_sensors设为true
    # 接触传感器不会生成对应的prim，而是直接附加在现有的prim上，在这里是附加在机器人的脚上
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True,
        filter_prim_paths_expr = ["/World/defaultGroundPlane"]
    )# 储存历史信息


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")
        print(scene["height_scanner"])
        print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())
        # print("-------------------------------",scene["contact_forces"].data.net_forces_w.shape)
        # print("--------------------------------",scene["contact_forces"].data.force_matrix_w.shape)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset() # 仅当播放模拟的时候才会初始化传感器的缓冲区和物理句柄，因此调用sim.reset()是很重要的
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
