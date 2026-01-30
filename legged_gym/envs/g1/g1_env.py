from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from legged_gym.envs import LeggedRobot
import csv


class g1Env(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # # ================== 日志相关（最简版） ==================
        # self.log_step = 0
        # self.log_file = open("euler.csv", "w", newline="")
        # self.log_writer = csv.writer(self.log_file)

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

    def _get_walk_mask(self):
        self.feet_phase[:, 0] = self.phase[:, 0]
        self.feet_phase[:, 1] = torch.fmod(self.phase[:, 0] + 0.5, 1.0)
        walk_mask = self.feet_phase <= 0.55
        return walk_mask

    def step(self, actions):
        # # dynamic randomization
        # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def post_physics_step(self):
        super().post_physics_step()
        self.extras['privileged_obs'] = self.get_privileged_observations()
        self.extras['obs_history'] = self.get_observations_history()
        # # ========= 边跑边写 CSV =========
        # env_id = 0
        # # self.log_writer.writerow([
        # #     # float(self.torques[env_id, 3]),  # 左膝
        # #     # float(self.torques[env_id, 9]),  # 右膝
        # #     # float(torch.norm(self.base_euler_xyz[env_id, :2])),
        # # ])

    def compute_observations(self):
        heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights

        privileged_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.feet_pos[:, :, 2] - self.root_states[:, 2].unsqueeze(1) + self.cfg.rewards.base_height_target) * 5.0,
            self.contacts,
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

        if self.cfg.terrain.measure_heights:
            # privileged_obs = torch.cat((
            #     privileged_obs,
            #     torch.clip(heights, -1, 1) * 5.0,    # 15
            # ), dim=-1)

            obs = torch.cat((
                obs,
                torch.clip(heights, -1, 1) * 5.0,  # 15
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
    # def _reward_hip_pos(self):
    #     error = self.sqrdexp(10. * (self.dof_pos[:, 1]))
    #     error += self.sqrdexp(10. * (self.dof_pos[:, 2]))
    #     error += self.sqrdexp(10. * (self.dof_pos[:, 7]))
    #     error += self.sqrdexp(10. * (self.dof_pos[:, 8]))
    #     return error / 4.

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]] - self.default_dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_ankle_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [4, 5, 10, 11]] - self.default_dof_pos[:, [4, 5, 10, 11]]), dim=1)

    def _reward_feet_contact(self):
        walk_mask = self._get_walk_mask()
        reward = (self.contacts == walk_mask).float()
        return torch.mean(reward, dim=1)

    # def _reward_feet_height(self):
    #     contact_foot = torch.min(self.feet_pos[:, 0, 2], self.feet_pos[:, 1, 2])
    #     swing_foot = torch.max(self.feet_pos[:, 0, 2], self.feet_pos[:, 1, 2])
    #     reward = (swing_foot - contact_foot - self.cfg.rewards.target_feet_height) > -0.01
    #     return reward

