import numpy as np

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

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[-1]
    q_vec = q[:3]
    # a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    a = v * np.expand_dims(2.0 * q_w ** 2 - 1.0, axis=-1)
    # b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    b = np.cross(q_vec, v) * np.expand_dims(q_w, axis=-1) * 2.0
    # c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    c = q_vec * np.expand_dims(np.sum(q_vec * v, axis=-1), axis=-1) * 2.0
    return a - b + c

def quat_to_grav(q):
    q = np.asarray(q)
    v = np.array([0, 0, -1], dtype=np.float32)
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., np.newaxis]
    b = 2.0 * q_w[..., np.newaxis] * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.sum(q_vec * v, axis=-1)[..., np.newaxis]
    return a - b + c

def smooth_sqr_wave(phase, cycle_time):
    p = 2.*np.pi*phase * 1. / cycle_time
    return np.sin(p) / (2*np.sqrt(np.sin(p)**2. + 0.2**2.)) + 1./2.

