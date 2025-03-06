from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class h1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        o_h_frame_stack = 25

        num_single_obs = 40
        single_num_privileged_obs = 48 # + 9
        num_observations = int(frame_stack * num_single_obs)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 10
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        knee_name = "knee"

        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["knee", "hip"]
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False # Humanoid is False , Dog is True

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        # curriculum = False
        # measure_heights = False

        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = False

        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.06]

        default_joint_angles = {
            'left_hip_yaw_joint' : 0. ,
            'left_hip_roll_joint' : 0,
            'left_hip_pitch_joint' : -0.1,
            'left_knee_joint' : 0.3,
            'left_ankle_joint' : -0.2,

            'right_hip_yaw_joint' : 0.,
            'right_hip_roll_joint' : 0,
            'right_hip_pitch_joint' : -0.1,
            'right_knee_joint' : 0.3,
            'right_ankle_joint' : -0.2,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_yaw': 150.0,
                     'hip_roll': 150.0,
                     'hip_pitch': 150.0,
                     'knee': 200.0,
                     'ankle': 40
                     }

        damping = {'hip_yaw': 3,
                   'hip_roll': 3,
                   'hip_pitch': 3,
                   'knee': 4,
                   'ankle': 2
                   }

        action_scale = 0.25
        decimation = 4

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

    class domain_rand:
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        max_push_ang_vel = 1.0
        dynamic_randomization = 0.02

        randomize_commands = True
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        randomize_all_mass = False
        rd_mass_range = [0.5, 1.5]
        randomize_all_com = False
        rd_com_range = [-0.03, 0.03]
        randomize_base_com = True
        added_com_range = [-0.10, 0.10]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = False
        Kd_factor_range = [0.8, 1.2]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]
        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0.01, 0.05]


    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        base_height_target = 1.03
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.10    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.8               # sec
        target_air_time = 0.4

        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.5
        max_contact_force = 400     # Forces above this value are penalized

        class scales:
            # feet pos
            hip_pos = 0.5
            ankle_pos = 0.3
            feet_air_time = -3.
            feet_height = -5.
            feet_contact = 0.6
            contact_no_vel = -0.3

            # vel tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            ang_vel_xy = -0.1
            lin_vel_z = -3.

            # base pos
            orientation = 1.
            base_height = 0.2

            # energy
            action_rate = -0.01
            torques = -1e-5
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.
            dof_pos_limits = -5.
            torque_limits = -1e-2
            alive = 0.5

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 20.
        clip_actions = 20.


class h1CfgPPO(LeggedRobotCfgPPO):
    # OnPolicyRunner  EstOnPolicyRunner  RNNOnPolicyRunner
    # DWLOnPolicyRunner PIAOnPolicyRunner
    runner_class_name = 'PIAOnPolicyRunner'

    class policy:
        # # only for 'OnPolicyRunner':
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [768, 256, 128]

        # # only for 'EstOnPolicyRunner':
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [768, 256, 128]
        # state_estimator_dims = [256, 128, 64]

        # only for 'RNNOnPolicyRunner', 'DWLOnPolicyRunner' and 'PIAOnPolicyRunner':
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        schedule = 'adaptive'
        entropy_coef = 0.01
        gamma = 0.99
        lam = 0.95
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'    # ActorCritic,  ActorCriticRecurrent,  ActorCriticPIA
        algorithm_class_name = 'PPO'
        num_steps_per_env = 25  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'h1'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt