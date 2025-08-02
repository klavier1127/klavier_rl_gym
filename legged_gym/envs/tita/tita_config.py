from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class titaCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        o_h_frame_stack = 25

        num_single_obs = 35
        num_privileged_obs = 3
        num_single_critic_obs = num_single_obs + num_privileged_obs
        num_observations = int(frame_stack * num_single_obs)
        num_critic_observations = int(c_frame_stack * num_single_critic_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 8


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tita/urdf/tita.urdf'
        name = "tita"
        foot_name = "leg_4"
        wheel_name = ["leg_4"]
        penalize_contacts_on = ["leg_1", "leg_2", "leg_3"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        replace_cylinder_with_capsule = False   # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False          # Some .obj meshes must be flipped from y-up to z-up

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

        # mesh_type = 'trimesh'
        # curriculum = True
        # measure_heights = False

        measured_points_x = [-0.1, 0., 0.1] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.1, 0., 0.1]
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 8  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]
        restitution = 0.

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
        pos = [0.0, 0.0, 0.4]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'joint_left_leg_1': 0,
            'joint_right_leg_1': 0,

            'joint_left_leg_2': 0.8,
            'joint_right_leg_2': 0.8,

            'joint_left_leg_3': -1.5,
            'joint_right_leg_3': -1.5,

            'joint_left_leg_4': 0,
            'joint_right_leg_4': 0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'leg_1': 40, 'leg_2': 40, 'leg_3': 40, "leg_4": 20}  # [N*m/rad]
        damping = {'leg_1': 2, 'leg_2': 2, 'leg_3': 2, "leg_4": 1}  # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4
        wheel_speed = 1

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.3
        dynamic_randomization = 0.02

        randomize_commands = True
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        randomize_all_mass = False
        rd_mass_range = [0.5, 1.5]
        randomize_all_com = False
        rd_com_range = [-0.03, 0.03]
        randomize_base_com = False
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
        joint_friction_range = [0.03, 0.3]
        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]
        randomize_joint_armature = True
        joint_armature_range = [0.01, 0.03]


    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.5, 3.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0., 0.]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        base_height_target = 0.35   # 0.25
        base_feet_height = 0.035
        target_feet_height = 0.06       # m
        cycle_time = 0.5                # sec
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.5
        max_contact_force = 100     # Forces above this value are penalized

        class scales:
            hip_pos = 0.5
            feet_contact = 0.
            feet_air_time = 0.0   # 1.
            feet_height = 0.0    # 0.2
            foot_mirror_up = -0.
            feet_slip = -0.
            # vel tracking
            tracking_lin_vel = 1.    # 1.
            tracking_ang_vel = 0.5   # 0.5
            ang_vel_xy = -0.1     # -0.05
            lin_vel_z = -2      # -2.
            # base pos
            orientation = -2.0
            base_height = -1.0
            # energy
            action_rate = -0.01      # -0.5
            torques = -1e-4
            dof_vel = -1e-3
            dof_acc = -2e-7
            collision = -1.  # -1.
            dof_pos_limits = -5.    # -10.
            torque_limits = -1e-2
            termination = -0.  # -100.   # -100
            alive = 0.2

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


class titaCfgPPO(LeggedRobotCfgPPO):
    # OnPolicyRunner  EstOnPolicyRunner  RNNOnPolicyRunner
    # DWLOnPolicyRunner PIAOnPolicyRunner
    runner_class_name = 'RNNOnPolicyRunner'

    class policy:
        # # only for 'OnPolicyRunner':
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [768, 256, 128]

        # # only for 'EstOnPolicyRunner':
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [768, 256, 128]
        # state_estimator_dims = [256, 128, 64]

        # # only for 'RNNOnPolicyRunner' and 'DWLOnPolicyRunner':
        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1

        # only for 'PIAOnPolicyRunner':
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
        experiment_name = 'tita'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt