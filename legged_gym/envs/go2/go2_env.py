from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import HumanoidTerrain, Terrain


class go2Env(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = -0.28
        self.feet_height = torch.zeros((self.num_envs, self.feet_num), device=self.device)
        self.feet_phase = torch.zeros((self.num_envs, self.feet_num), device=self.device)
        self.max_feet_air_time = torch.zeros_like(self.feet_air_time)
        self.env_obs_buf = torch.zeros((self.num_envs, self.cfg.env.num_env_obs), device=self.device)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 17] =  noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[17: 29] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[29: 41] = 0.  # previous actions
        noise_vec[41: 44] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[44: 47] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def _get_contact_mask(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        return contact_mask

    def _get_walk_mask(self):
        self.feet_phase[:, 0] = self.phase[:, 0]
        self.feet_phase[:, 3] = self.phase[:, 0]
        self.feet_phase[:, 1] = torch.fmod(self.phase[:, 0] + 0.5, 1.0)
        self.feet_phase[:, 2] = torch.fmod(self.phase[:, 0] + 0.5, 1.0)
        walk_mask = self.feet_phase < 0.55
        return walk_mask

    def step(self, actions):
        # # dynamic randomization
        # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def post_physics_step(self):
        super().post_physics_step()
        self.extras['privileged_obs'] = self.get_privileged_observations()
        self.extras['obs_history'] = self.get_observations_history()
        self.extras['env_obs'] = self.get_env_observations()

    def foot_positions_in_base_frame(self):
        feet_indices = self.feet_indices
        feet_states = self.rigid_state[:, feet_indices, :]
        assert feet_states.shape == (self.num_envs, 2, 13), f"feet state shape is {feet_states.shape}"
        Lfoot_positions_local = quat_rotate_inverse(self.base_quat, feet_states[:, 0, :3] - self.root_states[:, :3])
        Rfoot_positions_local = quat_rotate_inverse(self.base_quat, feet_states[:, 1, :3] - self.root_states[:, :3])
        self.feet_pos = torch.concat((Lfoot_positions_local, Rfoot_positions_local), dim=-1)
        return self.feet_pos

    def compute_observations(self):
        contact_mask = self._get_contact_mask()
        phase_sin = torch.sin(2 * torch.pi * self.phase)
        phase_cos = torch.cos(2 * torch.pi * self.phase)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        self.privileged_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target) * 10.,
            contact_mask,
        ), dim=-1)

        heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights
        self.env_obs = torch.cat((
            heights,    # 15
        ), dim=-1)

        self.obs = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            q,
            dq,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity * self.obs_scales.quat,
        ), dim=-1)

        self.critic_obs = torch.cat((
            self.obs,
            self.privileged_obs,
        ), dim=-1)

        if self.add_noise:
            self.obs = self.obs.clone() + (2 * torch.rand_like(self.obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            self.obs = self.obs.clone()

        self.obs_history.append(self.obs)
        self.critic_history.append(self.critic_obs)

        self.obs_buf = torch.cat([self.obs_history[i] for i in range(-self.cfg.env.frame_stack, 0)], dim=1)
        self.critic_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        self.obs_history_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.o_h_frame_stack)], dim=1)

    def get_privileged_observations(self):
        self.privileged_obs_buf = self.privileged_obs.clone()
        return self.privileged_obs_buf

    def get_observations_history(self):
        return self.obs_history_buf

    def get_env_observations(self):
        self.env_obs_buf = self.env_obs.clone()
        return self.env_obs_buf


    # ================================================ Rewards ================================================== #

    #####################     foot-pos    #################################################################
    def _reward_hip_pos(self):
        error = 0.
        error += self.sqrdexp(10. * (self.dof_pos[:, 0] - self.default_dof_pos[:, 0]))
        error += self.sqrdexp(10. * (self.dof_pos[:, 3] - self.default_dof_pos[:, 3]))
        error += self.sqrdexp(10. * (self.dof_pos[:, 6] - self.default_dof_pos[:, 6]))
        error += self.sqrdexp(10. * (self.dof_pos[:, 9] - self.default_dof_pos[:, 9]))
        return error / 4.

    def _reward_feet_contact(self):
        contact_mask = self._get_contact_mask()
        walk_mask = self._get_walk_mask()
        reward = 1. * (contact_mask == walk_mask)
        reward = torch.sum(reward, dim=1)
        return reward

    def _reward_feet_air_time(self):
        contact_mask = self._get_contact_mask()
        contact_filt = torch.logical_or(contact_mask, self.last_contacts)
        self.last_contacts = contact_mask
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        error = torch.sum((self.feet_air_time - self.cfg.rewards.target_air_time) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return torch.square(error)

    def _reward_feet_height(self):
        contact_mask = self._get_contact_mask()
        walk_mask = self._get_walk_mask()
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.root_states[:, 2].unsqueeze(1)
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        error = torch.sum((self.feet_height - self.cfg.rewards.target_feet_height) * (1 - (1. * walk_mask)), dim=1)
        self.feet_height *= ~contact_mask
        return torch.square(error)


    ################################# balance ##################################################

    def _reward_base_height(self):
        feet_heights = self.rigid_state[:, self.feet_indices, 2]
        feet_heights_f = feet_heights[:, 0]
        feet_heights_b = feet_heights[:, 2]
        base_feet_heights =  torch.where(feet_heights_f >= feet_heights_b, feet_heights_b, feet_heights_f)
        base_height = self.root_states[:, 2] - (base_feet_heights - 0.02)
        return torch.square(base_height - self.cfg.rewards.base_height_target)