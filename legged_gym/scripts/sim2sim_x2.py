import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.x2.x2_config import x2Cfg
from legged_gym.utils import quat_to_euler
import torch


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def smooth_sqr_wave(phase, cycle_time):
    p = 2.*np.pi*phase * 1. / cycle_time
    return np.sin(p) / (2*np.sqrt(np.sin(p)**2. + 0.2**2.)) + 1./2.

def get_obs(data):
    q = data.qpos[7:]
    dq = data.qvel[6:]
    quat = data.qpos[3:7]
    quat = [quat[1], quat[2], quat[3], quat[0]]
    omega = data.qvel[3:6]
    eu_ang = quat_to_euler(quat)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi
    return q, dq, omega, eu_ang

def pd_control(default_dof_pos, target_q, q, kp, target_dq, dq, kd):
    return (target_q - q + default_dof_pos) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    default_pos = [0, 0, 0.3, -0.6, 0.3,     0, 0, 0.3, -0.6, 0.3]
    data.qpos[7:] = default_pos
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    obs_history = deque()
    for _ in range(cfg.env.o_h_frame_stack):
        obs_history.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    phase = 0
    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        # Obtain an observation
        q, dq, omega, euler = get_obs(data)

        # 1000hz -> 100hz
        force = [0, 0, 0]
        vx, vy, dyaw = 0.0, 0.0, 0.0
        cmd = np.array([[vx, vy, dyaw]], dtype=np.float32)

        cycle_time = 0.6
        dt_phase = cfg.sim_config.dt / cycle_time
        phase = phase + dt_phase

        if count_lowlevel % cfg.sim_config.decimation == 0:
            phase_sin = np.sin(2 * math.pi * phase)
            phase_cos = np.cos(2 * math.pi * phase)
            # phase_sin = np.where(stand_mask, 0, phase_sin)
            # phase_cos = np.where(stand_mask, 0, phase_cos)

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            obs[0, 0] = phase_sin
            obs[0, 1] = phase_cos
            obs[0, 2:5] = cmd[0] * 2
            obs[0, 5:15] = (q - default_pos)
            obs[0, 15:25] = dq * 0.05 # dq * 0.05
            obs[0, 25:35] = action
            obs[0, 35:38] = omega*0.25
            obs[0, 38:40] = euler[:2]

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            hist_obs.append(obs)
            hist_obs.popleft()
            obs_history.append(obs)
            obs_history.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            policy_input_history = np.zeros([1, cfg.env.num_obs_history], dtype=np.float32)
            for i in range(cfg.env.o_h_frame_stack):
                policy_input_history[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = obs_history[i][0, :]
            action = policy(torch.tensor(policy_input), torch.tensor(policy_input_history)).detach().numpy()

            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            target_q = action * cfg.control.action_scale
            # target_q[:, 0] = 0
            # target_q[:, 6] = 0

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(default_pos, target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        data.ctrl = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name="pelvis")

        data.xfrc_applied[robot_body_id, :3] += force
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    # parser.add_argument('--load_model', type=str, required=True,
    #                     help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg(x2Cfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/scene.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20 # 50hz
        class robot_config:
            kps = np.array([200, 200, 200, 200, 30,     200, 200, 200, 200, 30], dtype=np.double)
            kds = np.array([  4,   4,   4,   4,  4,       4,   4,   4,   4,  4], dtype=np.double)
            tau_limit = np.array([30, 45, 60, 60, 30,    30,  45,  60,  60,  30], dtype=np.double)


    model_path = "../logs/x2/exported/policies/policy_lstm.pt"
    policy = torch.jit.load(model_path)
    run_mujoco(policy, Sim2simCfg())
