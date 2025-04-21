from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs import LeggedRobot
import torch
from legged_gym.algo.ppo_amp.datasets.motion_loader import AMPLoader

COM_OFFSET = torch.tensor([0.011611, 0.004437, 0.000108])
HIP_OFFSETS = torch.tensor([
    [0.1881, 0.04675, 0.],
    [0.1881, -0.04675, 0.],
    [-0.1881, 0.04675, 0.],
    [-0.1881, -0.04675, 0.]]) + COM_OFFSET

class go2AMPEnv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

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

    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.213
        l_low = 0.213
        l_hip = 0.072 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions

    def step(self, actions):
        # # dynamic randomization
        # actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def post_physics_step(self):
        super().post_physics_step()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.extras['reset_env_ids'] = env_ids
        self.extras['terminal_amp_states'] = self.get_amp_observations()[env_ids]
        self.extras['privileged_obs'] = self.get_privileged_observations()
        self.extras['obs_history'] = self.get_observations_history()
        self.extras['amp_obs'] = self.get_amp_observations()

    def compute_observations(self):
        phase_sin = torch.sin(2 * torch.pi * self.phase)
        phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.privileged_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.root_states[:, 2].unsqueeze(1) - self.feet_pos[:, :, 2] - self.cfg.rewards.base_height_target) * 10.,
            self.contacts,
        ), dim=-1)
        heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights
        if self.cfg.terrain.measure_heights:
            self.privileged_obs = torch.cat((
            self.privileged_obs,
            heights,    # 15
        ), dim=-1)

        self.obs = torch.cat((
            phase_sin,
            phase_cos,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity * self.obs_scales.quat,
        ), dim=-1)

        self.critic_obs = torch.cat((
            self.obs,
            self.privileged_obs,
        ), dim=-1)

        if self.add_noise:
            self.obs += (2 * torch.rand_like(self.obs) -1) * self.noise_scale_vec * self.cfg.noise.noise_level

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

    def get_amp_observations(self):
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        self.amp_obs_buf = torch.cat((
            self.dof_pos,   # 12
            foot_pos,       # 12
            self.base_lin_vel,  # 3
            self.base_ang_vel,  # 3
            self.dof_vel,  # 12
            self.root_states[:, 2:3]),   # 1
        dim=-1)
        return self.amp_obs_buf

    def _reset_dofs_amp(self, env_ids, frames):
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames).to(torch.float32)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames).to(torch.float32)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))