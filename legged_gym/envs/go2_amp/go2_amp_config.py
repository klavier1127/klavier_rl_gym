from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import glob
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
motion_dir = os.path.join(current_dir, "datasets/mocap_motions")
MOTION_FILES = [f for f in glob.glob(os.path.join(motion_dir, "*")) if os.path.isfile(f)]


class go2AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 15
        c_frame_stack = 15
        o_h_frame_stack = 25

        num_single_obs = 47
        num_privileged_obs = 11
        num_single_critic_obs = num_single_obs + num_privileged_obs
        num_observations = int(frame_stack * num_single_obs)
        num_critic_observations = int(c_frame_stack * num_single_critic_obs)
        num_obs_history = int(o_h_frame_stack * num_single_obs)

        num_actions = 12
        amp_motion_files = MOTION_FILES


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        replace_cylinder_with_capsule = True   # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True          # Some .obj meshes must be flipped from y-up to z-up

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
        pos = [0.0, 0.0, 0.35]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': -0. ,  # [rad]
            'RR_hip_joint': -0.,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'joint': 40}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz

    class domain_rand:
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.3
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
            lin_vel_x = [-1.0, 2.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.57, 1.57]

    class rewards:
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False

        base_height_target = 0.32   # 0.25
        base_feet_height = 0.028
        target_feet_height = 0.06       # m
        cycle_time = 0.5               # sec
        target_air_time = 0.25
        # tracking reward = exp(error*sigma)
        tracking_sigma = 0.25
        max_contact_force = 100     # Forces above this value are penalized

        class scales:
            # feet pos
            hip_pos = 0.3
            # base pos
            default_dof_pos = -0.03
            orientation = -1.
            base_height = -1.
            # vel tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.05
            lin_vel_z = -1.
            # energy
            action_rate = -0.01
            torques = -1e-4
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.0
            dof_pos_limits = -5.0
            torque_limits = -1e-2
            alive = 0.3


class go2AMPCfgPPO(LeggedRobotCfgPPO):
    # AMPOnPolicyRunner
    runner_class_name = 'AMPOnPolicyRunner'

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 30000  # number of policy updates
        experiment_name = 'go2_amp'

        amp_reward_coef = 0.1
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_discr_hidden_dims = [512, 256]

