import numpy as np
import matplotlib.pyplot as plt
import copy

from unpack_data.victor_io_zarr import EpisodeData

from physics_utils import MASS_KG, gravity_compensation, compute_force, compute_zft, get_contact_idx_with_height

def plot_episode_cal(episode: EpisodeData, index: int = 0, ax=None):
    times = episode.times - episode.times[0]

    # Raw force from sensor (Fx, Fy, Fz)
    force_raw = episode.wrenches[:, :3]

    # Gravity-compensated force in tool frame
    quat = episode.states[:, -4:]
    force_gc = gravity_compensation(force_raw, quat, MASS_KG)

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
        episode_copy = copy.deepcopy(episode)
        episode_copy.wrenches[:,:3] *= np.array([-1, -1, -1])
        plot_episode_cal(episode_copy, index=n, ax=ax)

    for a in ax.ravel():
        a.legend()

    fig.suptitle("Calibration: Raw vs Gravity-Compensated Force (overlapped trials)", y=1.02)
    plt.tight_layout()
    plt.show()