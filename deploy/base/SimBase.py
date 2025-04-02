import math
import time
import torch
import numpy as np
from collections import deque


class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break


class SimBase(object):
    def __init__(self, _cfg, _policy):
        self.run_thread = None
        self.run_flag = True
        self.cfg = _cfg
        self.policy = _policy

        self.timestamp_ms = 0
        # joint target
        self.target_q = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action_filter = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.num_actions = self.cfg.env.num_actions
        self.default_dof_pos = self.cfg.robot_config.default_dof_pos
        self.wb_pos = self.cfg.robot_config.wb_pos
        # obs
        self.hist_obs = deque()
        for _ in range(self.cfg.env.frame_stack):
            self.hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))
        self.obs_history = deque()
        for _ in range(self.cfg.env.o_h_frame_stack):
            self.obs_history.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))

    def ref_trajectory(self, cnt_pd_loop):
        leg_l = math.sin(2 * math.pi * cnt_pd_loop * 0.001 / self.cfg.control.cycle_time)  # x * 0.001, ms -> s
        leg_r = math.sin(2 * math.pi * cnt_pd_loop * 0.001 / self.cfg.control.cycle_time)  # x * 0.001, ms -> s
        ref_dof_pos = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        ref_dof_pos[2] = self.wb_pos[2] + leg_l * 0.17
        ref_dof_pos[3] = self.wb_pos[3] - leg_l * 0.34
        ref_dof_pos[4] = self.wb_pos[4] + leg_l * 0.17

        ref_dof_pos[7] = self.wb_pos[7] + leg_r * 0.17
        ref_dof_pos[8] = self.wb_pos[8] - leg_r * 0.34
        ref_dof_pos[9] = self.wb_pos[9] + leg_r * 0.17
        return ref_dof_pos

    def run(self):
        pass

    def get_action(self, obs):
        self.hist_obs.append(obs)
        self.hist_obs.popleft()
        self.obs_history.append(obs)
        self.obs_history.popleft()
        policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
        for i in range(self.cfg.env.frame_stack):
            policy_input[0, i * self.cfg.env.num_single_obs: (i + 1) * self.cfg.env.num_single_obs] = self.hist_obs[i][0, :]
        policy_input_history = np.zeros([1, self.cfg.env.num_obs_history], dtype=np.float32)
        for i in range(self.cfg.env.o_h_frame_stack):
            policy_input_history[0, i * self.cfg.env.num_single_obs: (i + 1) * self.cfg.env.num_single_obs] = self.obs_history[i][0, :]
        self.action[:] = self.policy(torch.tensor(policy_input), torch.tensor(policy_input_history)).detach().numpy()
        self.action = np.clip(self.action, -20.0, 20.0)
        self.action_filter = 0.7 * self.action_filter + 0.3 * self.action
        self.target_q = self.action_filter * self.cfg.control.action_scale
        return self.action * self.cfg.control.action_scale + self.default_dof_pos[:10]

    def init_robot(self):
        pass

