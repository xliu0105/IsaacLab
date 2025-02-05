# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.actuators import ActuatorBaseCfg
from omni.isaac.lab.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .articulation import Articulation


@configclass  # NOTE: @configclass修饰器用于将类转化为配置类，可以理解为自动生成初始化方法(__init__)
class ArticulationCfg(AssetBaseCfg):
    """Configuration parameters for an articulation."""

    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):
        """Initial state of the articulation."""

        # root velocity
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Linear velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Angular velocity of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

        # joint state
        joint_pos: dict[str, float] = {".*": 0.0}
        """Joint positions of the joints. Defaults to 0.0 for all joints."""
        joint_vel: dict[str, float] = {".*": 0.0}
        """Joint velocities of the joints. Defaults to 0.0 for all joints."""

    ##
    # Initialize configurations.
    ##

    class_type: type = Articulation

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the articulated object. Defaults to identity pose with zero velocity and zero joint state."""

    soft_joint_pos_limit_factor: float = 1.0
    """
    soft_joint_pos_limit_factor参数将实际的自由度位置限制范围缩放一下，如果原来的位置限制范围是-1.0到1.0，这个参数设为0.9的话，缩放后的实际位置限制范围就是-0.9到0.9。
    Fraction specifying the range of DOF position limits (parsed from the asset) to use. Defaults to 1.0.

    The joint position limits are scaled by this factor to allow for a limited range of motion.
    This is accessible in the articulation data through :attr:`ArticulationData.soft_joint_pos_limits` attribute.
    """

    actuators: dict[str, ActuatorBaseCfg] = MISSING
    """Actuators for the robot with corresponding joint names."""
