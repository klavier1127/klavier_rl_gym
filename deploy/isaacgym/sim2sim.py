import math
import time
import torch
import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm
from deploy import DEPLOY_ROOT_DIR
from deploy_config import deploy_config
from deploy.base.SimBase import SimBase
from deploy.base.SimBase import NanoSleep
from deploy.utils.math_utils import quat_to_grav


class Sim2Sim(SimBase):
    def __init__(self, _cfg, _policy):
        super().__init__(_cfg, _policy)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.data.qpos[7:] = self.default_dof_pos[:10]
        self.cmd = np.array([0.0, 0.0, 0.0])
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def get_obs(self):
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        omega = self.data.qvel[3:6]
        quat = self.data.qpos[3:7]
        quat = [quat[1], quat[2], quat[3], quat[0]]
        proj_grav = quat_to_grav(quat)
        return q, dq, omega, proj_grav

    def set_sim_target(self, target_q):
        q = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        dq = self.data.qvel.astype(np.double)[-self.cfg.env.num_actions:]
        tau = (target_q - q) * self.cfg.robot_config.kps - dq * self.cfg.robot_config.kds
        self.data.ctrl = np.clip(tau, -self.cfg.robot_config.tau_limit[-self.cfg.env.num_actions:],
                      self.cfg.robot_config.tau_limit[-self.cfg.env.num_actions:])  # Clamp torques
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def run(self):
        cnt_pd_loop = 0
        phase = 0
        pbar = tqdm(range(int(self.cfg.env.run_duration / 0.001)), desc="x02 Simulating...")  # x * 0.001, ms -> s
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            # Obtain an observation
            q, dq, omega, proj_grav = self.get_obs()

            cycle_time = 0.6
            dt_phase = self.cfg.sim_config.dt / cycle_time
            phase = phase + dt_phase
            phase_sin = np.sin(2 * math.pi * phase)
            phase_cos = np.cos(2 * math.pi * phase)
            # 1000hz -> 100hz
            if cnt_pd_loop % self.cfg.control.decimation == 0:
                obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
                obs[0, 0] = phase_sin
                obs[0, 1] = phase_cos
                obs[0, 2] = self.cmd[0]
                obs[0, 3] = self.cmd[1]
                obs[0, 4] = 0
                obs[0, 5:15] = q - self.default_dof_pos[:10]
                obs[0, 15:25] = dq * 0.05
                obs[0, 25:35] = self.action
                obs[0, 35:38] = omega * 0.25
                obs[0, 38:41] = proj_grav
                # obs[0, 42:45] = proj_grav
                self.target_q = self.get_action(obs)  # 策略推理
                # self.target_q = self.ref_trajectory(cnt_pd_loop)  # 参考轨迹可视化，需要将xml中的<freejoint/>注释掉，将机器人挂起来
                now = time.perf_counter()
                pbar.set_postfix(
                    calculateTime=f"{(now - start_time) * 1000:.3f}ms",  # 计算用时，单位毫秒
                    runTime=f"{(now - start):.3f}s"  # 运行时间，单位秒
                )
            self.set_sim_target(self.target_q)
            cnt_pd_loop += 1
        self.viewer.close()

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        timer = NanoSleep(1)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            for idx in range(self.cfg.env.num_actions):
                self.target_q[idx] = s0 * q0[idx] + s1 * qd[idx]
            self.set_sim_target(self.target_q)
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def init_robot(self):
        final_goal = self.default_dof_pos
        self.joint_plan(1, final_goal)
        for idx in range(self.cfg.env.num_actions):
            self.target_q[idx] = final_goal[idx]

if __name__ == '__main__':
    mode_path = f"{DEPLOY_ROOT_DIR}/logs/x2/exported/policies/policy_rma.pt"
    policy = torch.jit.load(mode_path)
    mybot = Sim2Sim(deploy_config, policy)
    mybot.init_robot()
    mybot.run()
