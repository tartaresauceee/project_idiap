import numpy as np
import matplotlib.pyplot as plt
from unpack_data.victor_io_zarr import extract_episode, EpisodeData

def plot_single(episode: EpisodeData, index=0):
    times = episode.times

    fig, axes = plt.subplots(2, 3, sharex=True,figsize=(12, 10))

    axes[0, 0].plot(times, episode.states[:,0], label="x")
    axes[0, 0].legend()

    axes[0, 1].plot(times, episode.states[:,1], label="y")
    axes[0, 1].legend()

    axes[0, 2].plot(times, episode.states[:,2], label="z")
    axes[0, 2].legend()

    axes[1, 0].plot(times, episode.wrenches[:,0], label="Fx")
    axes[1, 0].legend()

    axes[1, 1].plot(times, episode.wrenches[:,1], label="Fy")
    axes[1, 1].legend()

    axes[1, 2].plot(times, episode.wrenches[:,2], label="Fz")
    axes[1, 2].legend()

    fig.suptitle(f"Episode {index}")
    fig.tight_layout()

def main():
    dataset_name = "calibration"

    episode_n = [0, 1, 2, 3]

    episodes = extract_episode(dataset_name, episode_n)

    # Normalize to list so plotting logic is uniform
    if isinstance(episodes, EpisodeData):
        episodes = [episodes]
        episode_n = [episode_n]

    for episode in episodes:
        force = np.mean(episode.wrenches[50:-50, :3], axis=0)
        print(force.shape)
        mag = np.linalg.norm(force)
        print(mag)
        g = 9.81
        m = mag / g
        print("Mass = ", m)
        plot_single(episode)

    plt.show()

    


if __name__ == "__main__":
    main()