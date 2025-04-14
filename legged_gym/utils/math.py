import math
import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def get_scale_shift(range):
    # scale = 2. / (range[1] - range[0])
    scale = 2. / (range[1] - range[0]) if range[1] != range[0] else 1.
    shift = (range[1] + range[0]) / 2.
    return scale, shift

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

def quat_to_euler(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def quat_to_grav(q):
    q = np.asarray(q)
    v = np.array([0, 0, -1], dtype=np.float32)
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = 2.0 * q_w[..., np.newaxis] * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=-1)[..., np.newaxis]
    return a - b + c


def euler_to_grav(euler):
    r = euler[0]
    p = euler[1]
    y = euler[2]
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    grav = np.dot(rot, np.array([0, 0, -1]))
    return grav

# @ torch.jit.script
def exp_avg_filter(x, avg, alpha):
    """
    Simple exponential average filter
    """
    avg = alpha*x + (1-alpha)*avg
    return avg

def apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff):
    # Create a Jacobian matrix and move it to the same device as input tensors
    J = torch.eye(q.shape[-1]).to(q.device)
    J[4, 3] = 0.2
    J[9, 8] = 0.2

    # Perform transformations using Jacobian
    q = torch.matmul(q, J.T)
    qd = torch.matmul(qd, J.T)
    q_des = torch.matmul(q_des, J.T)
    qd_des = torch.matmul(qd_des, J.T)

    # Inverse of the transpose of Jacobian
    J_inv_T = torch.inverse(J.T)

    # Compute feed-forward torques
    tau_ff = torch.matmul(J_inv_T, tau_ff.T).T

    # Compute kp and kd terms
    kp = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kp, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    kd = torch.diagonal(
        torch.matmul(
            torch.matmul(J_inv_T, torch.diag_embed(kd, dim1=-2, dim2=-1)),
            J_inv_T.T
        ),
        dim1=-2, dim2=-1
    )

    # Compute torques
    torques = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff
    torques = torch.matmul(torques, J)

    return torques

def kalman_filter(p, x, z):
    q = 1e-5
    r = 0.3
    x_pred = x
    p_pred = p + q
    k = p_pred / (p_pred + r)
    x_new = x_pred + k * (z - x_pred)
    p_new = (1 - k) * p_pred
    return x_new, p_new