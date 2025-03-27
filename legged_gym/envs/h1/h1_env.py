from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import HumanoidTerrain, Terrain


class h1Env(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 15] =  noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15: 25] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[25: 35] = 0.  # previous actions
        noise_vec[35: 38] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[38: 40] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def _get_walk_mask(self):
        self.feet_phase[:, 0] = self.phase[:, 0]
        self.feet_phase[:, 1] = torch.fmod(self.phase[:, 0] + 0.5, 1.0)
        walk_mask = self.feet_phase < 0.55
        return walk_mask

    def step(self, actions):
        # # dynamic randomization
        # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self):
        phase_sin = torch.sin(2 * torch.pi * self.phase)
        phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.privileged_obs = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,

            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target) * 10.,
            self.contacts,
        ), dim=-1)

        self.obs = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,
        ), dim=-1)

        if self.add_noise:
            self.obs += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_level

        self.obs_history.append(self.obs)
        self.critic_history.append(self.privileged_obs)

        self.obs_buf = torch.cat([self.obs_history[i] for i in range(-self.cfg.env.frame_stack, 0)], dim=1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        self.obs_history_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.o_h_frame_stack)], dim=1)


    # ================================================ Rewards ================================================== #
    #####################     foot-pos    #################################################################
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0,1,5,6]] - self.default_dof_pos[:, [0,1,5,6]]), dim=-1)

    def _reward_ankle_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [4,9]] - self.default_dof_pos[:, [4,9]]), dim=-1)

    def _reward_feet_contact(self):
        walk_mask = self._get_walk_mask()
        reward = 1. * (self.contacts == walk_mask)
        return torch.sum(reward, dim=1)

