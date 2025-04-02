import math
import time
import copy
import torch
import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm
from deploy_config import deploy_config
from deploy.base.SimBase import SimBase
from deploy.base.SimBase import NanoSleep
from deploy.utils.math import quat_to_euler


class Sim2Sim(SimBase):
    def __init__(self, _cfg, _policy):
        super().__init__(_cfg, _policy)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.data.qpos[7:] = self.default_dof_pos[:10]
        self.cmd = np.array([2.0, 0.0, 0.0])
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def get_obs(self):
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        quat = [quat[1], quat[2], quat[3], quat[0]]
        omega = self.data.qvel[3:6]
        eu_ang = quat_to_euler(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        return q, dq, omega, eu_ang

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
        pbar = tqdm(range(int(self.cfg.env.run_duration / 0.001)),
                    desc="x02 Simulating...")  # x * 0.001, ms -> s
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            # Obtain an observation
            q, dq, omega, euler = self.get_obs()

            speed_command = np.linalg.norm(self.cmd[:2])
            stand_mask = speed_command < 0.2
            cycle_time = 0.7
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
                obs[0, 38:40] = euler[:2]
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
        final_goal = self.wb_pos
        self.joint_plan(1, final_goal)
        for idx in range(self.cfg.env.num_actions):
            self.target_q[idx] = final_goal[idx]

    def show(self):
        m = self.model
        name = m.names.decode('utf-8').split('\x00')
        print("\033[32m>>The robot: %s with %d dof, DofProperties information as follow:\033[0m" % (name[0], m.njnt))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print(
            "\033[33m|     Joint names    | type | idx |   default_pos  | damping | stiffness |   limits_lower   |   limits_upper   | margin |\033[0m")
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        for i in range(0, m.njnt):
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(
                "| %-19s|  %2d  |  %2d | %5.2f(%7.2f) | %5.1f   | %6.1f    | %7.4f(%7.2f) | %7.4f(%7.2f) |  %5.2f |" % (
                    jointName, m.jnt_type[i], i,
                    m.qpos0[i], np.rad2deg(m.qpos0[i]),
                    m.dof_damping[i],
                    m.jnt_stiffness[i],
                    m.jnt_range[i][0], np.rad2deg(m.jnt_range[i][0]),
                    m.jnt_range[i][1], np.rad2deg(m.jnt_range[i][1]),
                    m.jnt_margin[i]))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print("\033[32m>>The robot: %s with %d actuators/controls(ctrl) informations:\033[0m" % (name[0], m.nu))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[33m|     Joint names    | actuator | id | limit | c_lower | c_upper |\033[0m")
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        for i in range(0, m.nu):
            joint_id = m.actuator_trnid[i]  # 获取关节 ID
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id[0])  # 获取关节名称
            actuatorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print("| %-19s|   %-6s | %2d |   %d   | %7.2f | %7.2f |" % (
                jointName, actuatorName, i,
                m.actuator_ctrllimited[i],
                m.actuator_ctrlrange[i][0],
                m.actuator_ctrlrange[i][1]))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[32m>>The robot: %s with %d sensors informations:\033[0m" % (name[0], m.nsensor))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[33m| sensor | id | dim | address |\033[0m")
        print("\033[32m+--------+----+-----+---------+\033[0m")
        for j in range(0, m.nsensor):
            SensorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, j);
            print("| %-6s | %2d |   %d |    %2d   |" % (
                SensorName, j,
                m.sensor_dim[j],
                m.sensor_adr[j]))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[32m>> the end of joint and sensor information !\033[0m")


if __name__ == '__main__':
    mode_path = "/home/droid/IssacGym-projects/klavier_rl_gym/logs/x2/exported/policies/policy_lstm.pt"
    policy = torch.jit.load(mode_path)
    mybot = Sim2Sim(deploy_config, policy)
    mybot.show()
    mybot.init_robot()
    mybot.run()
