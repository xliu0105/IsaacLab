# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


# IMPORTANT: 直接工作流不适用Action和Observation管理器，因此任务配置需要定义环境的动作和观察数
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    # 注意，以下这一个是创建场景scene的配置类，这个配置类中没有定义robot属性，因此需要在_setup_scene函数中手动向scene中添加robot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005


class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg  # 这种表达方法仅仅是一个注解，而不是真正的类型声明

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)  # 调用父类的构造函数

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    # 创建场景
    # NOTE: 创建场景的函数必须名为_setup_scene(self)，因为在父类中这个名称已经被定义为抽象方法了
    # _setup_scene会在父类的构造函数中被调用
    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)  # 创建一个Articulation对象，这句话会直接在仿真环境中创建一个机器人
        # # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())  # 使用isaac lab提供的spawn_ground_plane函数创建一个地面
        # # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)  # 在父类的构造中定义了self.scene
        self.scene.filter_collisions(global_prim_paths=[])  # NOTE: 设置不同环境之间的碰撞被过滤
        # # add articultion to scene
        """
            注意，场景(scene)是已通过InteractiveSceneCfg配置, 由InteractiveScene类创建的.
            如果在InteractiveSceneCfg配置类中设置了robot, 如自定义了类继承于InteractiveSceneCfg, 并且其中定义了如
                robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            则创建的scene对象中的articulations属性(字典)中会有一个名为"robot"的Articulation对象
            但是在这个文件中, 并不是这么做的, InteractiveSceneCfg中并没有定义robot属性(Articulation), 因此需要通过下面的一行代码手动给scene添加robot
        """
        self.scene.articulations["cartpole"] = self.cartpole  # 手动向scene中添加robot(articulation)，如果没有向场景中添加robot，那么就无法控制并观察这个robot，但在仿真环境中这个robot是存在的
        # # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # action的step之前的预处理，在父类中是抽象方法，必须要重写实现
    # 这个预处理在每次step前只会被调用一次
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    # IMPORTANT: _apply_action函数是要将动作写到Articulation对象的buffer中，用的就是set_joint_effort_target函数，
    # 执行这一步之后还要有一个scene.write_data_to_sim()的操作才算是把action动作写进仿真器中，这个操作在父类的step函数已经实现了
    # 这个函数在仿真中会被调用decimation次，因为每一次step意味着在仿真中执行decimation次物理步骤，因此这一步的动作会被执行decimation次
    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}  # NOTE: 将观测值存在字典中，键必须是"policy"，因为在父类中已经用这个键来提取观测值了
        return observations

    # NOTE: 计算奖励的函数必须名为_get_rewards(self)，因为在父类中这个名称已经被定义为抽象方法了
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    # _reset_idx其实是不必要重写的，因为父类中的_reset_idx函数已经实现了对环境的重置，但这里是为了在reset的时候能够设置机器人的初始状态
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)  # 调用了父类的_reset_idx函数

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
