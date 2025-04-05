import math
import time
import torch
import numpy as np
from tqdm import tqdm
from deploy import DEPLOY_ROOT_DIR
from deploy_config import deploy_config
from deploy.base.SimBase import SimBase
from deploy.base.SimBase import NanoSleep
from grpc import insecure_channel
from deploy.base.DroidGrpcClient import DroidGrpcClient


class Sim2Real(SimBase, DroidGrpcClient):
    def __init__(self, _cfg, _policy, _grpc_channel):
        SimBase.__init__(self, _cfg, _policy)
        DroidGrpcClient.__init__(self, _grpc_channel)
        self.robotCommand.cmd_enable = 2  # joint control mode

        for i in range(10):
            self.robotCommand.kp[i] = self.cfg.robot_config.kps[i]
            self.robotCommand.kd[i] = self.cfg.robot_config.kds[i]
            self.robotCommand.max_torque[i] = self.cfg.robot_config.tau_limit[i]

    def init_robot(self):
        self.joint_plan(1, self.default_dof_pos)
        timer = NanoSleep(self.cfg.control.decimation)  # 创建一个1毫秒的NanoSleep对象
        self.get_robot_state()
        temp_tic = self.robotState.system_tic
        while self.robotState.rc_keys[0] > 64:
            if self.robotState.system_tic - temp_tic > 1000:
                temp_tic = self.robotState.system_tic
            start_time = time.perf_counter()
            self.get_robot_state()
            timer.waiting(start_time)
        print("单击CH6开始, CH8右滑到底急停", self.robotState.imu_euler)

        while (self.robotState.rc_keys[3] == 0) and (self.run_flag == True):  # CH6
            start_time = time.perf_counter()
            self.get_robot_state()
            print("单击CH6开始, CH8右滑到底急停", self.robotState.imu_euler)

            if self.robotState.rc_keys[0] > 64:
                print("紧急停止！！！")
                exit()
            timer.waiting(start_time)

    def get_obs(self):
        q = np.array(self.robotState.position)
        q = q[:10]
        dq = np.array(self.robotState.velocity)
        dq = dq[:10]
        omega = np.array(self.robotState.imu_gyro)
        euler = np.array(self.robotState.imu_euler)
        euler[euler > math.pi] -= 2 * math.pi
        return q, dq, omega, euler

    def run(self):
        pre_tic = 0
        cnt_pd_loop = 0
        timer = NanoSleep(self.cfg.control.decimation)
        pbar = tqdm(range(int(self.cfg.env.run_duration / (self.cfg.control.decimation * 0.001))), desc="x02 running...")
        start = time.perf_counter()

        for _ in pbar:
            start_time = time.perf_counter()
            self.get_robot_state()

            cycle_time = 0.6
            phase = torch.tensor(cnt_pd_loop * self.cfg.sim_config.dt / cycle_time)
            phase_sin = np.sin(2 * math.pi * phase)
            phase_cos = np.cos(2 * math.pi * phase)
            q, dq, omega, eu_ang = self.get_obs()

            obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
            obs[0, 0] = phase_sin
            obs[0, 1] = phase_cos
            obs[0, 2] = 1.5 * self.robotState.rc_du[2] * self.cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = 0.5 * self.robotState.rc_du[1] * self.cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = self.robotState.rc_du[3] * 0.5
            obs[0, 5:15] = q - self.default_dof_pos[:10]
            obs[0, 15:25] = dq * 0.05
            obs[0, 25:35] = self.action
            obs[0, 35:38] = omega * 0.25
            obs[0, 38:40] = eu_ang[:2]

            self.target_q = self.get_action(obs)
            # self.target_q[0]=0
            # self.target_q[5]=0

            if self.robotState.rc_keys[0] > 64:
                for idx in range(10):
                    self.robotCommand.kp[idx] = self.cfg.robot_config.standkps[idx]
                    self.robotCommand.kd[idx] = self.cfg.robot_config.standkds[idx]

                self.robotCommand.position[:10] = self.default_dof_pos[:10]
            else:
                for idx in range(10):
                    self.robotCommand.kp[idx] = self.cfg.robot_config.kps[idx]
                    self.robotCommand.kd[idx] = self.cfg.robot_config.kds[idx]
                    self.robotCommand.position[idx] = self.target_q[idx]

            self.set_robot_command()
            pbar.set_postfix(
                realCycle=f"{self.robotState.system_tic - pre_tic}ms",  # 实际循环周期，单位毫秒
                calculateTime=f"{(time.perf_counter() - start_time) * 1000:.3f}ms",  # 计算用时，单位毫秒
                runTime=f"{(time.perf_counter() - start):.3f}s"  # 运行时间，单位秒
            )
            pre_tic = self.robotState.system_tic
            timer.waiting(start_time)
            cnt_pd_loop += self.cfg.control.decimation
        self.joint_plan(1, self.cfg.robot_config.default_dof_pos)


if __name__ == '__main__':
    mode_path = f"{DEPLOY_ROOT_DIR}/logs/x2/exported/policies/policy_lstm.pt"
    channel = insecure_channel('192.168.55.12:50051')
    policy = torch.jit.load(mode_path)
    mybot = Sim2Real(deploy_config, policy, channel)
    mybot.init_robot()
    mybot.fox_run()
    mybot.run()
    mybot.fox_stop()
