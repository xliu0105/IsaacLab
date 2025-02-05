# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")  # 创建一个命令行参数处理器
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)  # 设置物理和渲染时间步长为0.01s，通过SimulationCfg创建对象
    sim = SimulationContext(sim_cfg)  # 通过上一行代码创建的SimulationCfg对象，使用SimulationContext创建模拟上下文实例
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])  # 设置主相机

    # Play the simulator
    sim.reset()  # 播放时间轴并初始化模拟器中的物理处理。在第一次步进模拟器之前必须调用此方法，否则模拟器不能正确初始化。注意区分其和sim.SimulationContext.play()方法，这个只播放时间轴但不初始化
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():  # 如果模拟器还在运行
        # perform step
        sim.step()  # step一次，即为步进一次


if __name__ == "__main__":
    # run the main function
    main()  # 调用上面定义的main函数
    # close sim app
    simulation_app.close()  # 调用omni.isaac.kit.SimulationApp.close()方法停止模拟并关闭窗口
