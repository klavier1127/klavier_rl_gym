import numpy as np
from deploy import DEPLOY_ROOT_DIR


class deploy_config:
    class env:
        # change the observation dim
        frame_stack = 15
        o_h_frame_stack = 25
        num_single_obs = 41
        num_observations = int(frame_stack * num_single_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)
        num_actions = 10
        run_duration = 5000.0 # 秒

    class cmd:
        vx = 0.0
        vy = 0.0
        yaw = 0.0

    class sim_config:
        dt = 0.001
        mujoco_model_path = f'{DEPLOY_ROOT_DIR}/resources/robots/x2/scene.xml'

    class control:
        action_scale = 0.25
        decimation = 20
        cycle_time = 0.6

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05

        clip_observations = 20.

    class robot_config:
        default_dof_pos = np.array([0, 0.05, 0.3, -0.6, 0.3,
                                          0, 0.05, 0.3, -0.6, 0.3,
                                                                   -0.3, 0.1, 0, 1.5,
                                                                   -0.3, 0.1, 0, 1.5], dtype=np.double)

        standkps = np.array([150, 150, 150, 150, 50, 150, 150, 150, 150, 50], dtype=np.double)
        standkds = np.array([  1,   3,   3,   3,  3,   1,   3,   3,   3,  3], dtype=np.double)
        kps = np.array([100, 200, 200, 200, 30, 100, 200, 200, 200, 30], dtype=np.double)
        kds = np.array([  2,   4,   4,   4,  4,   2,   4,   4,   4,  4], dtype=np.double)
        tau_limit = np.array([30, 45, 60, 60, 30,  30, 45, 60, 60, 30], dtype=np.double)

