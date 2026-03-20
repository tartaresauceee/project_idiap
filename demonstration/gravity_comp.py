import numpy as np
import matplotlib.pyplot as plt
from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from data_analysis import world2tool


G_WORLD = np.array([0.0, 0.0, -9.81])

def average_static_trial(episode: EpisodeData):
    """
    Average a static trial over time.
    Returns mean wrench (6,) and mean quaternion.
    """

    wrenches = episode.wrenches
    quats = episode.states[:,-4:]

    mean_wrench = np.mean(wrenches,axis=0)
 
    # Average quaternions: use SVD method (handles sign flips)
    Q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    # Flip quats that are on the wrong hemisphere relative to the first one
    signs = np.sign(Q @ Q[0])
    signs[signs == 0] = 1
    Q = Q * signs[:, None]

    _, _, Vt = np.linalg.svd(Q)
    mean_q = Vt[0]
    mean_q /= np.linalg.norm(mean_q)
 
    return mean_wrench, mean_q

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x u = v [cross] u."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])

def calibration_system(
    trials: list[tuple[np.ndarray, np.ndarray]],  # list of (mean_wrench, R_i)
    q0: np.ndarray,                                # rotation at tare pose
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the stacked system b=Ax from N calibration trials.
 
    x = [m, p]  where p = mc  (first moment)
 
    Each trial contributes a 6x4 block to A and a 6-vector to b.
    """

    g0 = world2tool(G_WORLD, q0, inverse=False)

    A_list = []
    b_list = []

    for mean_wrench_i, qi in trials:
        gi = world2tool(G_WORLD, qi, inverse=False)
        di = gi - g0

        # Measurement model
        H = np.zeros((6,4))
        H[:3,0] = di
        H[3:,1:] = -skew(di)

        A_list.append(H)
        b_list.append(mean_wrench_i)

    A = np.vstack(A_list)
    b = np.concatenate(b_list)

    return A, b

def estimate_gravity_params(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solve b = A x via least squares.
    Returns (mass, first_moment_p, center_of_mass_c).
    """

    x_star, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
 
    m_star = x_star[0]
    p_star = x_star[1:]
    c_star = p_star / m_star
 
    print(f"Estimated mass:           {m_star:.4f} kg")
    print(f"Estimated CoM (sensor f): {c_star} m")
    print(f"Matrix rank:              {rank} / {A.shape[1]}")
 
    return m_star, p_star, c_star



def main():
    dataset_name = "calibration"

    episode_n = [0, 1, 2, 3]

    episodes = extract_episode(dataset_name, episode_n)

    # Tare episode
    _, q0 = average_static_trial(episodes[0])

    print("q0: ", q0)

    # Gravity parameters identification
    avg_trials = []
    for episode in episodes[1:]:
        avg_trials.append(average_static_trial(episode))
    
    print(len(avg_trials))

    A, b = calibration_system(avg_trials, q0)
    m, p, c = estimate_gravity_params(A, b)


    


if __name__ == "__main__":
    main()