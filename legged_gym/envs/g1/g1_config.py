from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class g1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 1
        c_frame_stack = 1
        o_h_frame_stack = 25

        num_single_obs = 47
        num_privileged_obs = 7 + 15
        num_single_critic_obs = num_single_obs + num_privileged_obs
        num_observations = int(frame_stack * num_single_obs)
        num_critic_observations = int(c_frame_stack * num_single_critic_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 12

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        knee_name = "knee"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["knee", "hip"]
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        # curriculum = False
        # measure_heights = False

        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True

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

            'torso_joint' : 0.
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz

    class domain_rand:
        push_robots = True
        push_interval_s = 9
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.5
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
        randomize_base_com = False
        added_com_range = [-0.06, 0.06]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = False
        Kd_factor_range = [0.8, 1.2]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]
        randomize_joint_friction = False
        joint_friction_range = [0.01, 0.03]
        randomize_joint_damping = False
        joint_damping_range = [0.1, 0.3]
        randomize_joint_armature = True
        joint_armature_range = [0.01, 0.03]


    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False

        base_height_target = 0.78
        base_feet_height = 0.035
        target_feet_height = 0.08 # m
        cycle_time = 0.8 # sec
        target_air_time = 0.4 # sec
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.25
        max_contact_force = 500     # Forces above this value are penalized

        class scales:
            # feet pos
            hip_pos = -2.0
            ankle_pos = -0.0
            feet_contact = 0.5
            feet_air_time = -0.0
            feet_height = -10.0
            contact_no_vel = -0.2
            contact_forces = -0.0

            # vel tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.05
            lin_vel_z = -1.0

            # base pos
            default_dof_pos = -0.03
            orientation = -1.0
            base_height = -1.0

            # energy
            action_rate = -0.01
            torques = -1e-5
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.0
            dof_pos_limits = -5.0
            torque_limits = -1e-2
            alive = 0.3

class g1CfgPPO(LeggedRobotCfgPPO):
    # OnPolicyRunner  RNNOnPolicyRunner  RMAOnPolicyRunner DWAQOnPolicyRunner LAPDOnPolicyRunner
    runner_class_name = 'LAPDOnPolicyRunner'

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 25  # per iteration
        max_iterations = 10000  # number of policy updates
        experiment_name = 'g1'
