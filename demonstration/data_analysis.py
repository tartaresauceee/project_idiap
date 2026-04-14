import numpy as np
import matplotlib.pyplot as plt
import argparse


from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from scipy.spatial.transform import Rotation as R

Mass = 0.0312 # kg
G_WORLD = np.array([0.0, 0.0, -9.81])
q0 = np.array([-0.37382076, 0.03296935, -0.02435134, -0.92659488])

def get_contact_time(episode: EpisodeData, baseline_samples=50, drift_factor=0.5, threshold_factor=5.0):
    """Detect contact time using CUSUM on torque Tx."""
    torque = episode.wrenches[:, 3]
    times = episode.times

    # Estimate baseline from first samples (assumed non-contact)
    baseline = torque[:baseline_samples]
    mu0 = np.mean(baseline)
    sigma = np.std(baseline)

    drift = drift_factor * sigma
    threshold = threshold_factor * sigma

    # Two-sided CUSUM
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, len(torque)):
        s_pos = max(0.0, s_pos + (torque[i] - mu0) - drift)
        s_neg = max(0.0, s_neg - (torque[i] - mu0) - drift)
        if s_pos > threshold or s_neg > threshold:
            return i, times[i]

    return -1, None

def get_contact_idx_with_height(height: np.ndarray, thresh: float):
    idx = np.where(height < thresh)[0]
    first_idx = idx[0] if idx.size > 0 else None  # or None

    return first_idx

def plot_overlap(episodes: EpisodeData):

    fig, ax = plt.subplots(2, 2, figsize=(10,5))

    for episode in episodes:

        contact_idx = get_contact_idx_with_height(episode.states[:,2], thresh=0.04)
        if contact_idx is None:
            print("No contact detected")
            return
        
        range_ = range(contact_idx-50, contact_idx+300)

        times = episode.times
        times = (times - times[0])[range_]

        f_world_tau, f_world_f = compute_force(episode)

        zft = compute_zft(episode.states[:,:3], -f_world_tau)

        ax[0, 0].plot(times, episode.states[range_,2], label='Z')
        ax[0, 0].plot(times, zft[range_,2], label='ZFT')
        ax[0, 0].legend()

        ax[0, 1].plot(times, f_world_tau[range_,2], label='Fz (from moment)')
        ax[0, 1].plot(times, f_world_f[range_,2], label='Fz (from force)')
        ax[0, 1].legend()

        ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[range_,3:6], axis=1), label='|Moment|')
        ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[range_,:3], axis=1), label='|Force|')
        ax[1, 0].legend()

        ax[1, 1].plot(times, np.linalg.norm(f_world_tau[range_], axis=1), label='|F| from torque')
        ax[1, 1].plot(times, np.linalg.norm(f_world_f[range_], axis=1), label='|F| from force')
        ax[1, 1].legend()

def plot_single(episode: EpisodeData, index=0):
    times = episode.times - episode.times[0]

    f_world_tau, f_world_f = compute_force(episode)

    zft = compute_zft(episode.states[:,:3], -f_world_tau)

    fig, ax = plt.subplots(2, 2, figsize=(10,5), sharex=True)

    ax[0, 0].plot(times, episode.states[:,2], label='Z')
    ax[0, 0].plot(times, zft[:,2], label='ZFT')
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("Z-axis position (m)")
    ax[0, 0].set_title("Vertical Position VS Estimated ZFT")
    ax[0, 0].legend(loc='upper right')

    ax[0, 1].plot(times, f_world_tau[:,2], label='Fz')
    # OLD
    # ax[0, 1].plot(times, f_world_tau[:,2], label='Fz (tau est)')
    # ax[0, 1].plot(times, f_world_f[:,2], label='Fz (raw force)')
    # #ax[0, 1].plot(times, f_world_james[:,2], label='Fz (james est)')
    # ax[0, 1].plot(times, f_world_james_open[:,2], label='Fz (james est open)')
    ax[0, 1].set_xlabel("time (s)")
    ax[0, 1].set_ylabel("Z-axis force (N)")
    ax[0, 1].set_title("Estimated Vertical Force")
    ax[0, 1].legend(loc='upper right')

    ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[:,3:6], axis=1), label='|t|')
    ax[1, 0].plot(times, np.linalg.norm(f_world_f, axis=1), label='|F|')
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].set_ylabel("Magnitude (N & Nm)")
    ax[1, 0].set_title("Raw Wrench")
    ax[1, 0].legend(loc='upper right')

    ax[1, 1].plot(times, np.linalg.norm(f_world_tau, axis=1), label='|F|')
    # OLD
    # ax[1, 1].plot(times, np.linalg.norm(f_world_tau, axis=1), label='|F| tau est')
    # ax[1, 1].plot(times, np.linalg.norm(f_world_f, axis=1), label='|F| raw force')
    # ax[1, 1].plot(times, np.linalg.norm(f_world_james, axis=1), label='|F| james est')
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylabel("Magnitude (N)")
    ax[1, 1].set_title("Estimated Force Magnitude")
    ax[1, 1].legend(loc='upper right')

    contact_idx = get_contact_idx_with_height(episode.states[:,2], thresh=0.04)

    if contact_idx is not None:
        t_contact = times[contact_idx]
        for a in ax.ravel():
            a.axvline(x=t_contact, linestyle='--', color='red', linewidth=1.5)

    fig.tight_layout()
    fig.savefig("results.png", dpi=1200)

# def compute_force_james(episode: EpisodeData):

#     # Preprocess force
#     force_tool = preprocess_force(episode, world=False)
#     tau_tool = episode.wrenches[:,3:6]

#     # Compute position vector skew matrix
#     r_skew = skew(r)

#     # Closed form optimal force estimation
#     A = np.eye(r_skew.shape[0]) + r_skew.T @ r_skew
#     B = force_tool + tau_tool @ r_skew
#     F_tip_est_tool = B @ np.linalg.inv(A).T

#     quat = episode.states[:,-4:]
#     F_tip_est_world = world2tool(F_tip_est_tool, quat, inverse=True)

#     return F_tip_est_world

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




def preprocess_force(episode: EpisodeData, world=True):

    force_sensor = episode.wrenches[:,:3]
    quat = episode.states[:, -4:]


    force_tool = gravity_compensation(force_sensor, quat, Mass)

    if not world:
        return force_tool

    force_world = world2tool(force_tool, quat, inverse=True)

    return force_world

def compute_zft(position, force):

    K = 150 * np.eye(3)
    zft = force @ np.linalg.inv(K).transpose() + position

    return zft

def plot_episodes(episodes: list[EpisodeData], episode_n, overlap: bool = False):
    if not overlap:
        for episode, n in zip(episodes, episode_n):
            plot_single(episode, index=n)

    else:
        plot_overlap(episodes, episode_n)
    
    plt.show()

def plot_episodes_cal(episodes: list[EpisodeData], episode_n, overlap: bool = False):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

    for episode, n in zip(episodes, episode_n):
        episode.wrenches[:,:3] *= np.array([-1, -1, -1])
        plot_episode_cal(episode, index=n, ax=ax)

    for a in ax.ravel():
        a.legend()

    fig.suptitle("Calibration: Raw vs Gravity-Compensated Force (overlapped trials)", y=1.02)
    plt.tight_layout()
    plt.show()

def sensor2tool(sensor):
    perm = [0, 1, 2] # Aligned
    sign = [-1, -1, 1]
    tool = sensor[:, perm] * sign
    assert(sensor.all() == tool.all())
    return tool

def world2tool(world, quat, inverse=False):
    """
    - world: force in world frame
    - quat: quaterion of transformation from world to tool
    - inverse: compute rotation from tool to world if True
    """
    rot = R.from_quat(quat)
    tool = rot.apply(world, inverse=inverse)
    return tool

def gravity_compensation(force_tool, q, m, torque=False):
    g0 = world2tool(G_WORLD, q0, inverse=False)
    gi = world2tool(G_WORLD, q, inverse=False)

    F_g = m * (gi - g0)
    F = force_tool - F_g
    return F

def main():
    parser = argparse.ArgumentParser(description="Analyze episode data.")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g. metal, sponge, vial)")
    parser.add_argument("episode_n", type=int, nargs="+", help="Episode index/indices (e.g. 20 or 1 2 3)")
    parser.add_argument("--overlap", action="store_true", help="Enable overlap plotting")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    episode_n = args.episode_n
    overlap = args.overlap

    # For metal these indicies are failed [11, 12, 13, 14, 18, 19, 20, 25, 26, 29]
    # For sponge these indicies are failed [0, 2, 5, 6, 8, 10, 14, 17, 22, 27, 28, 29]

    episodes = extract_episode(dataset_name, episode_n)

    if episodes is None:
        print("Episode extraction: Something went wrong")
        return

    # Normalize to list so plotting logic is uniform
    if isinstance(episodes, EpisodeData):
        episodes = [episodes]
        episode_n = [episode_n]

    if dataset_name == "calibration":
        plot_episodes_cal(episodes, episode_n, overlap=overlap)

    else:
        plot_episodes(episodes ,episode_n, overlap=overlap)
    


if __name__ == "__main__":
    main()