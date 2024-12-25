# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()  # GroundPlaneCfg 配置了一个类似网格的地面平面，其外观和大小等属性可修改
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light 
    cfg_light_distant = sim_utils.DistantLightCfg(  # 这应该是设置一个配置类，后面使用func才是实例化对象吧
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    # NOTE：prim_utils.create_prim的注解见文档：https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#module-omni.isaac.core.utils.prims
    prim_utils.create_prim("/World/Objects", "Xform")  # 创建一个Xform基本体(原语)，这应该是用来分组的，在同一个Xform原语进行任何变换操作，都会同时作用于其下的所有子原语
    # 所以USD的模型文件也是类似于Mujoco的，是一种层级结构，看上面的isaac sim文档，可以看到可以直接使用prim_utils.create_prim创建各种类型的物体，如cube，sphere等，也可以导入usd文件
    # Xform作为组织和管理变换的节点，会将几何体放在Xform下；然而一般不会将Xform或者其他几何体再放在几何体的节点下面，因为这样会导致几何体的变换不正确
    # spawn a red cone
    # IMPORTANT: 注意，各类模型配置类都是定义在omni.isaac.lab.sim.spawners.shapes下面的，但由于在omni.isaac.lab.sim的__init__.py中导入了很多子功能，因此可以直接在omni.isaac.lab.sim下访问
    cfg_cone = sim_utils.ConeCfg(  # 这个锥的配置类只启用视觉元素，不启用物理属性
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))  # 配置类.func()应该是用来实例化对象的吧
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(  # 这个锥的配置类启用了物理属性
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 可以通过设置RigidBodyPropertiesCfg中的参数kinematic_enabled为true将这个物体设为运动学物体，不受物理影响
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # spawn a blue cuboid with deformable body
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
