# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
# 实例化一个获取命令行参数的对象
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch # 这里导入了Pytorch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
# IMPORTANT: 使用RigidObject将一个刚体添加到场景中，可以通过RigidObjectCfg类配置刚体的属性
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]] # 列表
    for i, origin in enumerate(origins):  # 通过for循环+enumerate，遍历origins列表，创建4个Xform原语，偏移为origins列表中定义好的
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    # 在文档中，RigidObjectCfg继承于AssetBaseCfg类，在实例化的时候，有些参数是AssetBaseCfg类的参数，有些是RigidObjectCfg类的参数
    cone_cfg = RigidObjectCfg( # 刚体的配置参数
        prim_path="/World/Origin.*/Cone",  # NOTE： 资源的路径，.*是通配符，.表示匹配任意字符，*表示匹配前面的元素多次，.*组合表示匹配任意长度的任意字符序列
        spawn=sim_utils.ConeCfg(  # 生成资源的配置参数
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),  # 初始状态
    )
    # IMPORTANT: 当将配置参数传递给RigidObject类时，当模拟播放时，会生成对象
    cone_object = RigidObject(cfg=cone_cfg)  # 将上面配置好的刚体参数传入RigidObject类中

    # return the scene information
    scene_entities = {"cone": cone_object}  # 这里设置一个字典，key是cone，value是cone_object
    return scene_entities, origins  # return这个字典和origins，origins应该是代表生成物体的位置

# 这里的函数定义中包含参数类型注解，是在3.5中引入的，用于标注函数的参数类型，但是不会对参数进行类型检查，也就是如果我传错了参数类型，也不会报错
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"]  # 从字典中取出key为cone的value，也就是在design_scene函数中定义的cone_object
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()  # 获取物理时间步长
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:  # 每250次循环重置一次
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = cone_object.data.default_root_state.clone()  # cone_object是RigidObject类的实例，default_root_state属性获取刚体的默认根状态
            # IMPORTANT: default_root_state返回的是一个tensor，shape是(num_instances, 13)，num_instances是刚体的数量，13是刚体的状态信息
            # 默认根属性可以在assets.RigidObjectCfg.init_state中配置
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins  # root_state[:, :3]表示root_state所有行的前三列
            root_state[:, :3] += math_utils.sample_cylinder(  # 从圆柱体表面采样随机点，返回的数据是一个(size,3)的tensor，根据device指定在CPU还是GPU上
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation
            # 将计算出来的root_state写入到模拟器中，覆盖掉原来的root_state，root_state是一个tensor
            cone_object.write_root_pose_to_sim(root_state[:, :7])
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            cone_object.reset()  # 重置所选环境的所有内置缓冲区，如果没有传入参数，将重置所有缓冲区
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        # 这一段代码需要在sim.step()之前执行，将数据写入到模拟器中，以便在模拟器中执行
        cone_object.write_data_to_sim()  # 将其他数据，比如外部力写入模拟缓冲区，但在这里没有添加任何外部力，同时此方法不是必须的
        # perform step
        sim.step()  # 模拟执行一次步进
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt)  # 更新刚体对象内部缓冲区，以反映其在assets.RigidObject.data 属性中的新状态，应该是为了让刚体对象的状态和模拟器中的状态保持一致，又可能是去处理物理交互事件
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)  # 将scene_origins转换为tensor，scene_origins应该是代表生成物体的位置
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
