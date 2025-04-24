from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class x2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 3
        o_h_frame_stack = 25

        num_single_obs = 41
        num_privileged_obs = 55
        num_single_critic_obs = num_single_obs + num_privileged_obs
        num_observations = int(frame_stack * num_single_obs)
        num_critic_observations = int(c_frame_stack * num_single_critic_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 10

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x2/x2.urdf'
        name = "x2"
        foot_name = "ankle_pitch"
        knee_name = "knee"
        terminate_after_contacts_on = ["pelvis", "hip", "knee"]
        penalize_contacts_on = ["knee", "hip"]
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False

        # mesh_type = 'trimesh'
        # curriculum = True
        # measure_heights = True

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
        pos = [0.0, 0.0, 0.90]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'L_hip_yaw': 0.,
            'L_hip_roll': 0.,
            'L_hip_pitch': 0.3,
            'L_knee_pitch': -0.6,
            'L_ankle_pitch': 0.3,

            'R_hip_yaw': 0.,
            'R_hip_roll': 0.,
            'R_hip_pitch': 0.3,
            'R_knee_pitch': -0.6,
            'R_ankle_pitch': 0.3,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 200,
                     'ankle': 30,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 4,
                     'hip_roll': 4,
                     'hip_pitch': 4,
                     'knee': 4,
                     'ankle': 4,
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
        randomize_all_mass = True
        rd_mass_range = [0.5, 1.5]
        randomize_all_com = True
        rd_com_range = [-0.03, 0.03]
        randomize_base_com = True
        added_com_range = [-0.06, 0.06]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]
        randomize_joint_friction = True
        joint_friction_range = [0.01, 0.1]
        randomize_joint_damping = True
        joint_damping_range = [0.1, 0.3]
        randomize_joint_armature = True
        joint_armature_range = [0.01, 0.03]


    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False

        base_height_target = 0.86
        base_feet_height = 0.09
        target_feet_height = 0.08 # m
        cycle_time = 0.6 # sec
        target_air_time = 0.3
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.25
        max_contact_force = 400     # Forces above this value are penalized

        class scales:
            # feet pos
            hip_pos = -2.0
            ankle_pos = -1.0
            feet_contact = 0.5
            feet_air_time = -0.0
            feet_height = -10.
            contact_no_vel = -0.2
            contact_forces = -0.005

            # vel tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.05
            lin_vel_z = -1.0

            # base pos
            default_dof_pos = -0.03
            orientation = -1.
            base_height = -1.

            # energy
            action_rate = -0.03
            torques = -1e-5
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.0
            dof_pos_limits = -5.0
            torque_limits = -1e-2
            alive = 0.3

class x2CfgPPO(LeggedRobotCfgPPO):
    # OnPolicyRunner  RNNOnPolicyRunner  RMAOnPolicyRunner
    runner_class_name = 'RMAOnPolicyRunner'

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 10000  # number of policy updates
        experiment_name = 'x2'
