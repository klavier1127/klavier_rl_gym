from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import torch
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import HumanoidTerrain, Terrain


class go2wEnv(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.feet_height = torch.zeros((self.num_envs, self.feet_num), device=self.device)
        self.feet_phase = torch.zeros((self.num_envs, self.feet_num), device=self.device)
        self.max_feet_air_time = torch.zeros_like(self.feet_air_time)
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 21] =  noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[21: 37] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[37: 53] = 0.  # previous actions
        noise_vec[53: 56] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[56: 58] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def step(self, actions):
        # dynamic randomization
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

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
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 1] - self.default_dof_pos[:, 1]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 5] - self.default_dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 9] - self.default_dof_pos[:, 9]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 13] - self.default_dof_pos[:, 13]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 2] - self.default_dof_pos[:, 2]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 6] - self.default_dof_pos[:, 6]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 10] - self.default_dof_pos[:, 10]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            10. * (self.dof_pos[:, 14] - self.default_dof_pos[:, 14]) / self.cfg.normalization.obs_scales.dof_pos)
        return error / 8.

    ################################# balance ##################################################
    def _reward_orientation(self):
        error = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        error += torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return error / 2

    # def _reward_base_height(self):
    #     feet_heights = self.rigid_state[:, self.feet_indices, 2]
    #     feet_heights_f = feet_heights[:, 0]
    #     feet_heights_b = feet_heights[:, 2]
    #     base_feet_heights =  torch.where(feet_heights_f >= feet_heights_b, feet_heights_b, feet_heights_f)
    #     base_height = self.root_states[:, 2] - (base_feet_heights - 0.09)
    #     error = torch.abs(base_height - self.cfg.rewards.base_height_target)
    #     return torch.exp(-10. * error)

    def _reward_base_height(self):
        error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        return torch.exp(-10. * error)