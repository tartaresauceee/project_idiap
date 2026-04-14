import numpy as np
import argparse

from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from plotting import plot_episodes, plot_episodes_cal

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