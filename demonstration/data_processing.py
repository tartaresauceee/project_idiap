import numpy as np
import argparse

from unpack_data.victor_io_zarr import extract_episode, EpisodeData
from plotting import plot_episodes, plot_episodes_cal

FAILED_EPISODES = {
    "metal": [11, 12, 13, 14, 18, 19, 20, 25, 26, 29],  # Contact detection issues
    "sponge": [0, 2, 5, 6, 8, 10, 14, 17, 22, 27, 28, 29],  # Low/no force
}

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize episode force data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_processing.py metal 20                    # Single episode
  python data_processing.py sponge 1 2 3 --overlap      # Multiple episodes
  python data_processing.py calibration 0 1 2           # Calibration episodes
        """
    )
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g. metal, sponge, vial)")
    parser.add_argument("episode_n", type=int, nargs="+", help="Episode index/indices (e.g. 0 or 1 2 3)")
    parser.add_argument("--overlap", action="store_true", help="Overlay multiple episodes on same plot")
    args = parser.parse_args()

    episodes = extract_episode(args.dataset_name, args.episode_n)

    if episodes is None:
        print("Episode extraction: Something went wrong")
        return

    # Normalize to list so plotting logic is uniform
    if isinstance(episodes, EpisodeData):
        episodes = [episodes]
        episode_n = [episode_n]

    if args.dataset_name == "calibration":
        plot_episodes_cal(episodes, args.episode_n, overlap=args.overlap)

    else:
        plot_episodes(episodes , args.episode_n, overlap=args.overlap)
    


if __name__ == "__main__":
    main()