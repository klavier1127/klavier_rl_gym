from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class g1WBCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        o_h_frame_stack = 25

        num_single_obs = 46 + 33
        single_num_privileged_obs = 54 + 33
        num_observations = int(frame_stack * num_single_obs)
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 23
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_23dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["knee", "hip", "shoulder", "elbow", "wrist"]
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

        # mesh_type = 'trimesh'
        # curriculum = True
        # measure_heights = False

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
        pos = [0.0, 0.0, 0.8]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0,
            'left_hip_yaw_joint' : 0. ,
            'left_knee_joint' : 0.3,
            'left_ankle_pitch_joint' : -0.2,
            'left_ankle_roll_joint' : 0,

            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0,
            'right_hip_yaw_joint' : 0.,
            'right_knee_joint' : 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint' : 0,

            'waist_yaw_joint' : 0.,
            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint': 0.,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_joint': 0.,
            'left_wrist_roll_joint': 0.,
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint': 0.,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_joint': 0.,
            'right_wrist_roll_joint': 0.,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 100,
                     'shoulder': 100,
                     'elbow': 100,
                     'wrist': 40,

                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 2,
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]

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
        randomize_joint_friction = False
        joint_friction_range = [0.01, 1.15]
        randomize_joint_damping = False
        joint_damping_range = [0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0.008, 0.03]


    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        base_height_target = 0.78
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        target_feet_height = 0.06 # m
        cycle_time = 0.8 # sec
        target_air_time = 0.4

        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.5
        max_contact_force = 300     # Forces above this value are penalized

        class scales:
            waist_pos = 0.3
            arm_mirror = 0.5
            shoulder_pos = 0.2
            elbow_pos = 0.2
            wrist_pos = 0.1
            # feet pos
            hip_pos = 0.5
            ankle_pos = 0.3
            feet_contact = 0.6
            feet_air_time = -3.
            feet_height = -5.
            contact_no_vel = -0.3

            # vel tracking
            tracking_lin_vel = 2.
            tracking_ang_vel = 1.
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


class g1WBCfgPPO(LeggedRobotCfgPPO):
    # OnPolicyRunner  EstOnPolicyRunner  RNNOnPolicyRunner
    # DWLOnPolicyRunner PIAOnPolicyRunner SymOnPolicyRunner
    runner_class_name = 'PIAOnPolicyRunner'

    class policy:
        # # only for 'OnPolicyRunner' and 'SymOnPolicyRunner':
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
        experiment_name = 'g1_wb'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt