import numpy as np
import matplotlib.pyplot as plt
import argparse


from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from scipy.spatial.transform import Rotation as R

Mass = 0.0312 # kg
G_WORLD = np.array([0.0, 0.0, -9.81])
q0 = np.array([-0.37382076, 0.03296935, -0.02435134, -0.92659488])
r = np.array([0.0, -0.104, 0.002325]) # vector from sensor origin to spatula tip 

force_drift = []

def plot_episode_cal(episode: EpisodeData, index: int = 0, ax=None):
    times = episode.times - episode.times[0]

    # Raw force from sensor (Fx, Fy, Fz)
    force_raw = episode.wrenches[:, :3]

    # Gravity-compensated force in tool frame
    quat = episode.states[:, -4:]
    force_gc = gravity_compensation(force_raw, quat, Mass)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        created_fig = True

    axes_labels = ["Fx", "Fy", "Fz"]

    # Top row: raw force (overlap episodes)
    for i in range(3):
        ax[0, i].plot(times, force_raw[:, i], label=f"ep {index}")
        ax[0, i].set_title(f"{axes_labels[i]} raw")
        ax[0, i].set_ylabel("Force (N)")
        ax[0, i].grid(True, alpha=0.3)

    # Bottom row: gravity-compensated force (overlap episodes)
    for i in range(3):
        ax[1, i].plot(times, force_gc[:, i], label=f"ep {index}")
        ax[1, i].set_title(f"{axes_labels[i]} gravity compensated")
        ax[1, i].set_xlabel("Time (s)")
        ax[1, i].set_ylabel("Force (N)")
        ax[1, i].grid(True, alpha=0.3)

    if created_fig:
        for a in ax.ravel():
            a.legend()
        fig.suptitle(f"Calibration Episode {index}: Raw vs Gravity-Compensated Force", y=1.02)
        plt.tight_layout()


def get_contact_idx_with_height(height: np.ndarray, thresh: float):
    idx = np.where(height < thresh)[0]
    first_idx = idx[0] if idx.size > 0 else None  # or None

    return first_idx

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]x such that [v]x u = v [cross] u."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])

def plot_overlap(episodes: list[EpisodeData], episode_n: list):

    fig, ax = plt.subplots(2, 2, figsize=(10,5))

    for episode, n in zip(episodes, episode_n):
        print(n)

        contact_idx = get_contact_idx_with_height(episode.states[:,2], thresh=0.045)
        if contact_idx is None:
            print("No contact detected")
            range_ = range(len(episode.times))
        else:
            range_ = range(contact_idx-50, contact_idx+300)

        times = episode.times
        times = (times - times[0])[range_]

        f_world_tau, f_world_f = compute_force(episode)

        zft = compute_zft(episode.states[:,:3], -f_world_tau)

        line_pos = ax[0, 0].plot(times, episode.states[range_,2], label='Z')
        ax[0, 0].plot(times, zft[range_,2], label='ZFT', linestyle='--', color=line_pos[0].get_color())
        ax[0, 0].set_xlabel("time (s)")
        ax[0, 0].set_ylabel("Z position (m)")
        ax[0, 0].set_title("Vertical Position VS Estimated ZFT")
        ax[0, 0].legend()

        line_F = ax[0, 1].plot(times, f_world_tau[range_,2], label='Fz (from moment)')
        ax[0, 1].plot(times, f_world_f[range_,2], label='Fz (from force)', linestyle='--', color=line_F[0].get_color())
        ax[0, 1].set_xlabel("time (s)")
        ax[0, 1].set_ylabel("Z force (N)")
        ax[0, 1].set_title("Estimated Vertical Forces")
        ax[0, 1].legend()

        lines_Fmag = ax[1, 0].plot(times, np.linalg.norm(episode.wrenches[range_,3:6], axis=1), label='|Moment|')
        ax[1, 0].plot(times, np.linalg.norm(f_world_f[range_,:3], axis=1), label='|Force|', linestyle='--', color=lines_Fmag[0].get_color())
        ax[1, 0].legend()
        ax[1, 0].set_xlabel("time (s)")
        ax[1, 0].set_ylabel("Force Norm (N)")
        ax[1, 0].set_title("Sensor Wrench Norm")

        ax[1, 1].plot(times, np.linalg.norm(f_world_tau[range_], axis=1), label='|F| from torque')
        ax[1, 1].plot(times, np.linalg.norm(f_world_f[range_], axis=1), label='|F| from force')
        ax[1, 1].legend()
        ax[1, 1].set_xlabel("time (s)")
        ax[1, 1].set_ylabel("Force Norm (m)")
        ax[1, 1].set_title("Estimated Forces Norm")
        ax[1, 1].legend()

        angles = R.from_quat(episode.states[range_,-4:]).as_euler('xyz', degrees=True)
        print("Average angle (xyz): ", np.mean(angles, axis=0))

        force_start = np.mean(episode.wrenches[:25, :3], axis=0)
        force_drift.append(np.linalg.norm(force_start))
        print("Force start: ", np.linalg.norm(force_start))

    _, axis = plt.subplots()
    axis.plot(episode_n, force_drift, 'bo')
    axis.set_xlabel("iter")
    axis.set_ylabel("Force (N)")
    axis.set_title("Average Force Norm at rest to detect drift")

def plot_single(episode: EpisodeData, index=0):
    times = episode.times - episode.times[0]

    f_world_tau, f_world_f = compute_force(episode)
    # f_world_james = compute_force_james(episode)
    # f_world_james_open = compute_force_james_open(episode)

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