import numpy as np
import matplotlib.pyplot as plt
import argparse


from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from scipy.spatial.transform import Rotation as R

Mass = 0.0042 # kg

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

    fig, ax = plt.subplots(2, 2, figsize=(10,5))

    ax[0, 0].plot(times, episode.states[:,2], label='Z')
    ax[0, 0].plot(times, zft[:,2], label='ZFT')
    ax[0, 0].legend()

    ax[0, 1].plot(times, f_world_tau[:,2], label='Fz (from moment)')
    ax[0, 1].plot(times, f_world_f[:,2], label='Fz (from force)')
    ax[0, 1].legend()

    ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[:,3:6], axis=1), label='|Moment|')
    ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[:,:3], axis=1), label='|Force|')
    ax[1, 0].legend()

    ax[1, 1].plot(times, np.linalg.norm(f_world_tau, axis=1), label='|F| from torque')
    ax[1, 1].plot(times, np.linalg.norm(f_world_f, axis=1), label='|F| from force')
    ax[1, 1].legend()

    contact_idx = get_contact_idx_with_height(episode.states[:,2], thresh=0.04)

    if contact_idx is not None:
        t_contact = times[contact_idx]
        for a in ax.ravel():
            a.axvline(x=t_contact, linestyle='--', color='red', linewidth=1.5)

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


    plt.figure()
    # plt.plot(episode.times, tau_tool, label=['taux', 'tauy', 'taudz'])
    plt.plot(episode.times, v, label=['vx', 'vy', 'vz'])
    plt.plot(episode.times, lambda_, label='lambda')
    plt.legend()
    plt.show()

    return world2tool(force_pred, quat, inverse=True), force_world

# ...existing code...

def plot_force_fft(force: np.ndarray, times: np.ndarray | None = None, remove_dc: bool = True, max_freq: float | None = None):

    n = force.shape[0]
    dt = np.mean(np.diff(times))

    sig = force.copy()
    if remove_dc:
        sig = sig - np.mean(sig, axis=0, keepdims=True)

    freqs = np.fft.rfftfreq(n, d=dt)
    fft_mag = np.abs(np.fft.rfft(sig, axis=0)) / n

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ["Fx", "Fy", "Fz"]
    for i in range(3):
        ax[i].plot(freqs, fft_mag[:, i], label=labels[i])
        ax[i].set_ylabel("|FFT|")
        ax[i].legend()
        ax[i].grid(True, alpha=0.3)

    ax[-1].set_xlabel("Frequency [Hz]")
    if max_freq is not None:
        ax[-1].set_xlim(0, max_freq)
    fig.suptitle("Force FFT (frequency domain)")
    plt.tight_layout()
    plt.show()

# ...existing code...

def preprocess_force(episode: EpisodeData):

    force_sensor = episode.wrenches[:,:3]
    quat = episode.states[:, -4:]
    force_tool = sensor2tool(force_sensor)
    force_world = world2tool(force_tool, quat, inverse=True)
    force_world = gravity_compensation(force_world, Mass)
    force_tool = world2tool(force_world, quat, inverse=False)

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
        plot_overlap(episodes)
    
    plt.show()

def sensor2tool(sensor):
    perm = [0, 1, 2] # Aligned
    sign = [1, 1, 1]
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

def gravity_compensation(force, m):
    return force + np.array([0.0, 0.0, 9.81*m])

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

    # Normalize to list so plotting logic is uniform
    if isinstance(episodes, EpisodeData):
        episodes = [episodes]
        episode_n = [episode_n]

    plot_episodes(episodes ,episode_n, overlap=overlap)
    


if __name__ == "__main__":
    main()