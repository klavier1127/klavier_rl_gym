from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import HumanoidTerrain, Terrain


class h1Env(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = -0.91 # -0.85
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.max_feet_air_time = torch.zeros_like(self.feet_air_time)
        self.feet_phase = torch.zeros((self.num_envs, 2), device=self.device)

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

    def _get_contact_mask(self):
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        return contact_mask

    def step(self, actions):
        # dynamic randomization
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self):
        contact_mask = self._get_contact_mask()
        phase_sin = torch.sin(2 * torch.pi * self.phase)
        phase_cos = torch.cos(2 * torch.pi * self.phase)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        push_robot = torch.norm(self.rand_push_force[:, :2] - self.last_rand_push_force[:, :2], dim=1) > 0.1
        self.last_rand_push_force[:, :2] = self.rand_push_force[:, :2]

        self.privileged_obs_buf = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,

            self.base_lin_vel * self.obs_scales.lin_vel,
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target,
            contact_mask,
            push_robot.unsqueeze(1),
            self.rew_buf.unsqueeze(1),
        ), dim=-1)

        self.obs_buf = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            q,
            dq,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if self.add_noise:
            self.obs_buf = self.obs_buf.clone() +(2 * torch.rand_like(self.obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            self.obs_buf = self.obs_buf.clone()

        self.obs_history.append(self.obs_buf)
        self.critic_history.append(self.privileged_obs_buf)

        self.obs_buf = torch.cat([self.obs_history[i] for i in range(-self.cfg.env.frame_stack, 0)], dim=1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        self.obs_history_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.o_h_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        self.max_feet_air_time[env_ids, :] = 0.
        super().reset_idx(env_ids)


    # ================================================ Rewards ================================================== #
    #####################     foot-pos    #################################################################
    def _reward_hip_pos(self):
        error = 0.
        error += self.sqrdexp(10. * self.dof_pos[:, 0])
        error += self.sqrdexp(10. * self.dof_pos[:, 1])
        error += self.sqrdexp(10. * self.dof_pos[:, 5])
        error += self.sqrdexp(10. * self.dof_pos[:, 6])
        return error / 4.

    def _reward_ankle_pos(self):
        error = 0
        error += self.sqrdexp(5. * (self.dof_pos[:, 4] - self.default_dof_pos[:, 4]))
        error += self.sqrdexp(5. * (self.dof_pos[:, 9] - self.default_dof_pos[:, 9]))
        return error / 2.

    def _reward_feet_contact(self):
        contact_mask = self._get_contact_mask()
        walk_mask = self._get_walk_mask()
        reward = 1. * (contact_mask == walk_mask)
        reward = torch.sum(reward, dim=1)
        return reward

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

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
    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_base_height(self):
        feet_heights = self.rigid_state[:, self.feet_indices, 2]
        feet_heights_l = feet_heights[:, 0]
        feet_heights_r = feet_heights[:, 1]
        base_feet_heights =  torch.where(feet_heights_l >= feet_heights_r, feet_heights_r, feet_heights_l)
        base_height = self.root_states[:, 2] - (base_feet_heights - 0.07)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)