from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.envs import LeggedRobot


class titaEnv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 13] =  noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[13: 21] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[21: 29] = 0.  # previous actions
        noise_vec[29: 32] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[32: 35] = noise_scales.quat * self.obs_scales.quat  # proj_grav
        return noise_vec

    def step(self, actions):
        # dynamic randomization
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def post_physics_step(self):
        super().post_physics_step()
        self.extras['privileged_obs'] = self.get_privileged_observations()
        self.extras['obs_history'] = self.get_observations_history()

    def _compute_torques(self, actions):
        # pd controller
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0
        self.dof_vel[:, self.wheel_indices] = -self.cfg.control.wheel_speed
        actions_scaled = actions * self.cfg.control.action_scale
        torques = self.p_gains * (actions_scaled + dof_err) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        self.dof_pos[:, self.wheel_indices] = 0
        privileged_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            # (self.feet_pos[:, :, 2] - self.root_states[:, 2].unsqueeze(1) + self.cfg.rewards.base_height_target) * 5.0,
            # self.contacts,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights
            privileged_obs = torch.cat((
            privileged_obs,
            torch.clip(heights, -1, 1) * 5.0,    # 15
        ), dim=-1)

        obs = torch.cat((
            torch.sin(2 * torch.pi * self.phase),
            torch.cos(2 * torch.pi * self.phase),
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity * self.obs_scales.quat,
        ), dim=-1)

        critic_obs = torch.cat((
            obs,
            privileged_obs,
        ), dim=-1)

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) -1) * self.noise_scale_vec * self.cfg.noise.noise_level

        self.obs_history.append(obs)
        self.critic_history.append(critic_obs)

        self.obs_buf = torch.cat([self.obs_history[i] for i in range(-self.cfg.env.frame_stack, 0)], dim=1)
        self.critic_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
        self.privileged_obs_buf = privileged_obs.clone()
        self.obs_history_buf = torch.cat([self.obs_history[i] for i in range(self.cfg.env.o_h_frame_stack)], dim=1)

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def get_observations_history(self):
        return self.obs_history_buf


    # ================================================ Rewards ================================================== #
    #####################     foot-pos    #################################################################
    def _reward_hip_pos(self):
        error = 0.
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 0] - self.default_dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 4] - self.default_dof_pos[:, 4]) / self.cfg.normalization.obs_scales.dof_pos)
        return error / 2.
