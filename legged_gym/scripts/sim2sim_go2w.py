import math
from pynput import keyboard
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_config import go2Cfg
from legged_gym.envs.go2w.go2w_config import go2wCfg
from legged_gym.utils import quat_to_euler, quat_to_grav, euler_to_grav

import torch



class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

vx, vy, dyaw = 0.0, 0.0, 0.0
def on_press(key):
    global vx, vy, dyaw
    try:
        if key.char == '2':  # 向前
            vx += 0.1
        elif key.char == '3':  # 向后
            vx -= 0.1
        elif key.char == '4':  # 向左
            vy += 0.1
        elif key.char == '5':  # 向右
            vy -= 0.1
        elif key.char == '6':  # 逆时针旋转
            dyaw += 0.1
        elif key.char == '7':  # 顺时针旋转
            dyaw -= 0.1
        elif key.char == '`':
            vx = 0.0
            vy = 0.0
            dyaw = 0.0

        # 限制速度范围
        vx = np.clip(vx, -1.0, 2.0)
        vy = np.clip(vy, -0.5, 0.5)
        dyaw = np.clip(dyaw, -1.0, 1.0)
    except AttributeError:
        pass

# 启动键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

def smooth_sqr_wave(phase, cycle_time):
    p = 2.*np.pi*phase * 1. / cycle_time
    return np.sin(p) / (2*np.sqrt(np.sin(p)**2. + 0.2**2.)) + 1./2.

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos[-16:]
    dq = data.qvel[-16:]
    quat = data.qpos[3:7]
    quat = [quat[1], quat[2], quat[3], quat[0]]
    omega = data.qvel[3:6]
    eu_ang = quat_to_euler(quat)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi
    return q, dq, omega, eu_ang

def pd_control(default_dof_pos, target_q, q, kp, target_dq, dq, kd):
    q_error = default_dof_pos- q
    q_error[[3, 7, 11, 15]] = 0.
    dq[[3, 7, 11, 15]] = 1.
    torques = (target_q + q_error) * kp + (target_dq - dq) * kd
    return torques

def run_mujoco(policy, cfg):
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    default_pos = [0.0, 0.67, -1.3, 0.0,   0.0, 0.67, -1.3, 0.0,    0.0, 0.67, -1.3, 0.0,    0.0, 0.67, -1.3, 0.0]
    data.qpos[-16:] = default_pos
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
        q[[3, 7, 11, 15]] = 0.
        # 1000hz -> 100hz
        force = [0, 0, 0]
        cmd = np.array([[vx, vy, dyaw]], dtype=np.float32)

        cycle_time = 0.5
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
            obs[0, 5:21] = q - default_pos
            obs[0, 21:37] = dq * 0.05
            obs[0, 37:53] = action
            obs[0, 53:56] = omega * 0.25
            obs[0, 56:58] = euler[:2]

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

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(default_pos, target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

        data.ctrl = tau
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


    class Sim2simCfg(go2wCfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/scene.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
        class robot_config:
            kps = np.array([50,  50,  50,  20,  50,  50,  50,  20,  50,  50,  50,  20,  50,  50,  50,  20], dtype=np.double)
            kds = np.array([1, 1, 1, 0.5,  1, 1, 1, 0.5, 1, 1, 1, 0.5,  1, 1, 1, 0.5], dtype=np.double)
            # tau_limit = np.array([23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55, 23.7, 23.7, 35.55], dtype=np.double)
            tau_limit = 50000. * np.ones(16, dtype=np.double)


    model_path = "../logs/go2w/exported/policies/policy_pia.pt"
    policy = torch.jit.load(model_path)
    run_mujoco(policy, Sim2simCfg())
