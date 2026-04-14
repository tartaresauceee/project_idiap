import numpy as np
from scipy.spatial.transform import Rotation as R

from unpack_data.victor_io_zarr import EpisodeData

# Physical parameters
MASS_KG = 0.0312  # Tool mass in kg
G_WORLD = np.array([0.0, 0.0, -9.81])  # Gravitational acceleration
Q0_REF = np.array([-0.37382076, 0.03296935, -0.02435134, -0.92659488]) # Reference quaternion
P_SENSOR_TO_TIP = np.array([0.0, -0.104, 0.002325]) # Sensor origin to spatula tip (m)

# Force estimation parameters
STIFFNESS_MATRIX = 150 * np.eye(3)  # ZFT stiffness (N/m)
CONTACT_HEIGHT_THRESHOLD = 0.045  # Height threshold for contact (m)
CONTACT_HEIGHT_THRESHOLD_TIGHT = 0.04  # Tighter threshold (m)

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x u = v [cross] u."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])

def sensor2tool(sensor):
    perm = [0, 1, 2] # Aligned
    sign = [-1, -1, 1]
    tool = sensor[:, perm] * sign
    assert(sensor.all() == tool.all())
    return tool

def world2tool(world, quat, inverse=False):
    """
    Rotate force vector between world and tool frames.
    
    Parameters
    ----------
    world : np.ndarray
        Force in world frame, shape (3,) or (N, 3).
    quat : np.ndarray
        Quaternion [x, y, z, w], shape (4,) or (N, 4).
    inverse : bool, optional
        If True, rotate from tool to world. Default is False (world to tool).
    
    Returns
    -------
    np.ndarray
        Rotated force, same shape as input.
    
    Notes
    -----
    Quaternion convention: [x, y, z, w] (scalar last).
    """
    rot = R.from_quat(quat)
    tool = rot.apply(world, inverse=inverse)
    return tool

def gravity_compensation(force_tool, q, m=MASS_KG, torque=False):
    g0 = world2tool(G_WORLD, Q0_REF, inverse=False)
    gi = world2tool(G_WORLD, Q0_REF, inverse=False)

    F_g = m * (gi - g0)
    F = force_tool - F_g
    return F

def preprocess_force(episode: EpisodeData, world=True):

    force_sensor = episode.wrenches[:,:3]
    quat = episode.states[:, -4:]


    force_tool = gravity_compensation(force_sensor, quat)

    if not world:
        return force_tool

    force_world = world2tool(force_tool, quat, inverse=True)

    return force_world

def compute_force(episode: EpisodeData):
    force_world = preprocess_force(episode)
    quat = episode.states[:,-4:]
    force_tool = world2tool(force_world, quat)
    tau_tool = episode.wrenches[:,3:6]
    
    
    # Compute torque from force
    l = 0.104
    r_tool = np.array([0.0, -l, 0.0])
    r_tool = np.broadcast_to(r_tool, force_tool.shape)
    # tau_tool_pred = np.cross(r_tool, force_tool)

    # Compute force from torque (assuming direction)
    force_unit = force_tool / np.linalg.norm(force_tool, axis=1).reshape(-1, 1)
    v = np.cross(r_tool, force_unit)
    lambda_ = np.abs(np.vecdot(tau_tool, v)) / np.vecdot(v, v)
    force_pred = lambda_[:, np.newaxis] * force_unit


    return world2tool(force_pred, quat, inverse=True), force_world

def compute_force_james(episode: EpisodeData):

    # Preprocess force
    force_tool = preprocess_force(episode, world=False)
    tau_tool = episode.wrenches[:,3:6]

    # Compute position vector skew matrix
    r_skew = skew(P_SENSOR_TO_TIP)

    # Closed form optimal force estimation
    A = np.eye(r_skew.shape[0]) + r_skew.T @ r_skew
    B = force_tool + tau_tool @ r_skew
    F_tip_est_tool = B @ np.linalg.inv(A).T

    quat = episode.states[:,-4:]
    F_tip_est_world = world2tool(F_tip_est_tool, quat, inverse=True)

    return F_tip_est_world

def compute_zft(position, force, K=150*np.eye(3)):
    """
    Compute Zero-Force Trajectory (ZFT) from position and estimated force.
    
    The ZFT represents the spatial location where force would be zero,
    computed from a linear spring model: F = K(z - position).
    
    Parameters
    ----------
    position : np.ndarray
        Spatial position, shape (N, 3).
    force : np.ndarray
        Estimated force, shape (N, 3).
    K : np.ndarray, optional
        Diagonal stiffness matrix, shape (3, 3). Default is 150*I.
    
    Returns
    -------
    np.ndarray
        Zero-force trajectory, shape (N, 3).
    """
    zft = force @ np.linalg.inv(K).transpose() + position

    return zft

def get_contact_idx_with_height(height: np.ndarray, thresh: float):
    idx = np.where(height < thresh)[0]
    first_idx = idx[0] if idx.size > 0 else None  # or None

    return first_idx