from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zarr
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Allow running this file directly by adding the repo root to sys.path.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@dataclass
class EpisodeData:
    states: np.ndarray
    wrenches: np.ndarray
    T_world_vialBase: Optional[np.ndarray]
    T_world_vialLip: Optional[np.ndarray]
    T_world_mocap: Optional[np.ndarray]
    episode_ends: np.ndarray
    times: np.ndarray
    images: Optional[np.ndarray]
    images_times: Optional[np.ndarray]

def pose7_to_T(pose):
    pos = np.asarray(pose[:3], dtype=float)
    quat = np.asarray(pose[3:7], dtype=float)  # [qx,qy,qz,qw]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T

def T_to_pose7(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got shape {T.shape}")
    pos = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [qx,qy,qz,qw]
    return np.concatenate([pos, quat])

def index_range_for_times(t_sensor, t0, t1):
    """
    Return a numpy array of indices in t_sensor that fall between the
    closest samples to t0 and t1 (inclusive).
    """
    i0 = int(np.argmin(np.abs(t_sensor - t0)))
    i1 = int(np.argmin(np.abs(t_sensor - t1)))

    # ensure correct order
    if i0 > i1:
        i0, i1 = i1, i0

    # return index array
    return np.arange(i0, i1 + 1)

def import_data(
    zarr_root: Path,
    target_hz: float,
    lowpass_cutoff_pos: float,
    lowpass_cutoff_wrench:float,
    butter_order: int,
):
    zarr_data = zarr.open(zarr_root, mode="r")

    print(zarr_data.tree())

    episode_ends_raw = zarr_data["data/episode_ends"][:]  # shape: (N_trials,)

    states_raw = zarr_data["data/state"][:]  # shape: (N_mocap, 7)
    times_raw = zarr_data["data/time"][:]  # shape: (N_mocap, 1)

    has_wrench = "data/wrench" in zarr_data and "data/wrench_time" in zarr_data
    if has_wrench:
        wrench_raw = zarr_data["data/wrench"][:]  # shape: (N_mocap, 6)
        wrench_times_raw = zarr_data["data/wrench_time"][:]  # shape: (N_mocap, 1)
    else:
        wrench_raw = None
        wrench_times_raw = None
        print("data/wrench does not exist")

    # Optional world transforms (vial base/lip/mocap).
    T_world_vial_base_raw = (
        zarr_data["data/T_world_vial_base"][:] if "data/T_world_vial_base" in zarr_data else None
    )
    if "data/T_world_vial_base_times" in zarr_data:
        T_world_vial_base_times_raw = zarr_data["data/T_world_vial_base_times"][:]
    elif "data/T_world_vial_base_time" in zarr_data:
        T_world_vial_base_times_raw = zarr_data["data/T_world_vial_base_time"][:]
    else:
        T_world_vial_base_times_raw = None

    T_world_vial_lip_raw = (
        zarr_data["data/T_world_vial_lip"][:] if "data/T_world_vial_lip" in zarr_data else None
    )
    if "data/T_world_vial_lip_times" in zarr_data:
        T_world_vial_lip_times_raw = zarr_data["data/T_world_vial_lip_times"][:]
    elif "data/T_world_vial_lip_time" in zarr_data:
        T_world_vial_lip_times_raw = zarr_data["data/T_world_vial_lip_time"][:]
    else:
        T_world_vial_lip_times_raw = None

    T_world_mocap_raw = zarr_data["data/T_world_mocap"][:] if "data/T_world_mocap" in zarr_data else None
    if "data/T_world_mocap_times" in zarr_data:
        T_world_mocap_times_raw = zarr_data["data/T_world_mocap_times"][:]
    elif "data/T_world_mocap_time" in zarr_data:
        T_world_mocap_times_raw = zarr_data["data/T_world_mocap_time"][:]
    else:
        T_world_mocap_times_raw = None

    if "data/image" in zarr_data:
        images_raw = zarr_data["data/image"]  # shape: (N_camera, H, W, 3)
        images_times_raw = zarr_data["data/image_times"]  # shape: (N_camear, H, W, 3)
    else:
        images_raw = None
        images_times_raw = None
        print("data/image does not exist")

    if "data/dynamixel" in zarr_data:
        dynamixel_raw = zarr_data["data/dynamixel"][:, 1]  # shape: (N_dynamixel, 1) angle in deg
        dynamixel_times_raw = zarr_data["data/dynamixel_time"][:]  # shape: (N_dynamixel, 1)
    else:
        print("data/dynamixel does not exist")

    # Ensure time arrays are 1-D for interp1d and slerp
    times_raw = np.asarray(times_raw).ravel()
    if has_wrench:
        wrench_times_raw = np.asarray(wrench_times_raw).ravel()
    if T_world_vial_base_times_raw is not None:
        T_world_vial_base_times_raw = np.asarray(T_world_vial_base_times_raw)
        if T_world_vial_base_times_raw.ndim > 1:
            T_world_vial_base_times_raw = T_world_vial_base_times_raw[:, 0]
        else:
            T_world_vial_base_times_raw = T_world_vial_base_times_raw.ravel()

    if T_world_vial_lip_times_raw is not None:
        T_world_vial_lip_times_raw = np.asarray(T_world_vial_lip_times_raw)
        if T_world_vial_lip_times_raw.ndim > 1:
            T_world_vial_lip_times_raw = T_world_vial_lip_times_raw[:, 0]
        else:
            T_world_vial_lip_times_raw = T_world_vial_lip_times_raw.ravel()
    if T_world_mocap_times_raw is not None:
        T_world_mocap_times_raw = np.asarray(T_world_mocap_times_raw)
        if T_world_mocap_times_raw.ndim > 1:
            T_world_mocap_times_raw = T_world_mocap_times_raw[:, 0]
        else:
            T_world_mocap_times_raw = T_world_mocap_times_raw.ravel()
    if "data/dynamixel" in zarr_data:
        dynamixel_times_raw = np.asarray(dynamixel_times_raw).ravel()

    dt = 1.0 / float(target_hz)

    target_times_list = []
    states_list = []
    wrenches_list = []
    dynamixel_orientationes_list = []
    T_world_vial_base_list = []
    T_world_vial_lip_list = []
    T_world_mocap_list = []
    episode_ends_list = []
    n_list = 0

    for trial in range(episode_ends_raw.shape[0]):
        if trial == 0:
            dex_start = 0
        else:
            dex_start = episode_ends_raw[trial - 1]

        # (FIX) Indexing error some were need to fix later
        dex_end = episode_ends_raw[trial] - 1

        # Time window for this trial in mocap time base
        t0 = times_raw[dex_start]
        t1 = times_raw[dex_end]

        # Select dynamixel samples in this time window
        dex_states = np.arange(dex_start, dex_end)
        dex_wrench = None
        if has_wrench:
            dex_wrench = index_range_for_times(wrench_times_raw, t0, t1)
        if "data/dynamixel" in zarr_data:
            dex_dynamixel = index_range_for_times(dynamixel_times_raw, t0, t1)
        dex_vial_base = None
        dex_vial_lip = None
        dex_mocap = None
        if T_world_vial_base_raw is not None and T_world_vial_base_times_raw is not None:
            dex_vial_base = index_range_for_times(T_world_vial_base_times_raw, t0, t1)
        if T_world_vial_lip_raw is not None and T_world_vial_lip_times_raw is not None:
            dex_vial_lip = index_range_for_times(T_world_vial_lip_times_raw, t0, t1)
        if T_world_mocap_raw is not None and T_world_mocap_times_raw is not None:
            dex_mocap = index_range_for_times(T_world_mocap_times_raw, t0, t1)

        # Interpolate to a fixed sampling rate for pose (chop edges to avoid slerp issues)
        target_times = np.arange(times_raw[dex_start] + dt, times_raw[dex_end] - dt, dt)

        # Create empty states vector for the trial
        states = np.zeros((len(target_times), 7))
        wrenches = np.zeros((len(target_times), 6))
        T_world_vial_base = None
        T_world_vial_lip = None
        T_world_mocap = None
        if T_world_vial_base_raw is not None:
            T_world_vial_base = np.zeros((len(target_times), T_world_vial_base_raw.shape[1]))
        if T_world_vial_lip_raw is not None:
            T_world_vial_lip = np.zeros((len(target_times), T_world_vial_lip_raw.shape[1]))
        if T_world_mocap_raw is not None:
            T_world_mocap = np.zeros((len(target_times), T_world_mocap_raw.shape[1]))
        if "data/dynamixel" in zarr_data:
            dynamixel_orientationes = np.zeros((len(target_times)))

        for i in range(3):
            f_interp = interp1d(times_raw[dex_states], states_raw[dex_states, i], kind="cubic")
            states[:, i] = f_interp(target_times)

        rot_in = Rotation.from_quat(states_raw[dex_states, 3:7])
        slerp = Slerp(times_raw[dex_states], rot_in)
        states[:, 3:7] = slerp(target_times).as_quat()

        if has_wrench and dex_wrench is not None:
            for i in range(6):
                f_interp = interp1d(
                    wrench_times_raw[dex_wrench],
                    wrench_raw[dex_wrench, i],
                    kind="cubic",
                    bounds_error=False,
                    fill_value=(wrench_raw[dex_wrench[0], i], wrench_raw[dex_wrench[-1], i]),
                )
                wrenches[:, i] = f_interp(target_times)

        if "data/dynamixel" in zarr_data:
            f_interp = interp1d(
                dynamixel_times_raw[dex_dynamixel],
                dynamixel_raw[dex_dynamixel],
                kind="nearest",
                bounds_error=False,
                fill_value=(dynamixel_raw[dex_dynamixel[0]], dynamixel_raw[dex_dynamixel[-1]]),
            )

            dynamixel_orientationes = f_interp(target_times)

        if T_world_vial_base is not None and dex_vial_base is not None:
            for i in range(T_world_vial_base_raw.shape[1]):
                f_interp = interp1d(
                    T_world_vial_base_times_raw[dex_vial_base],
                    T_world_vial_base_raw[dex_vial_base, i],
                    kind="nearest",
                    bounds_error=False,
                    fill_value=(
                        T_world_vial_base_raw[dex_vial_base[0], i],
                        T_world_vial_base_raw[dex_vial_base[-1], i],
                    ),
                )
                T_world_vial_base[:, i] = f_interp(target_times)

        if T_world_vial_lip is not None and dex_vial_lip is not None:
            for i in range(T_world_vial_lip_raw.shape[1]):
                f_interp = interp1d(
                    T_world_vial_lip_times_raw[dex_vial_lip],
                    T_world_vial_lip_raw[dex_vial_lip, i],
                    kind="nearest",
                    bounds_error=False,
                    fill_value=(
                        T_world_vial_lip_raw[dex_vial_lip[0], i],
                        T_world_vial_lip_raw[dex_vial_lip[-1], i],
                    ),
                )
                T_world_vial_lip[:, i] = f_interp(target_times)
        if T_world_mocap is not None and dex_mocap is not None:
            for i in range(T_world_mocap_raw.shape[1]):
                f_interp = interp1d(
                    T_world_mocap_times_raw[dex_mocap],
                    T_world_mocap_raw[dex_mocap, i],
                    kind="nearest",
                    bounds_error=False,
                    fill_value=(
                        T_world_mocap_raw[dex_mocap[0], i],
                        T_world_mocap_raw[dex_mocap[-1], i],
                    ),
                )
                T_world_mocap[:, i] = f_interp(target_times)

        if lowpass_cutoff_pos is not None:
            b, a = butter(butter_order, lowpass_cutoff_pos / (target_hz / 2), btype="low", analog=False)
            states[:, 0] = filtfilt(b, a, states[:, 0], axis=0)
            states[:, 1] = filtfilt(b, a, states[:, 1], axis=0)
            states[:, 2] = filtfilt(b, a, states[:, 2], axis=0)

        if lowpass_cutoff_wrench is not None:
            b, a = butter(butter_order, lowpass_cutoff_wrench / (target_hz / 2), btype="low", analog=False)
            wrenches[:, 0] = filtfilt(b, a, wrenches[:, 0], axis=0)
            wrenches[:, 1] = filtfilt(b, a, wrenches[:, 1], axis=0)
            wrenches[:, 2] = filtfilt(b, a, wrenches[:, 2], axis=0)
            wrenches[:, 3] = filtfilt(b, a, wrenches[:, 3], axis=0)
            wrenches[:, 4] = filtfilt(b, a, wrenches[:, 4], axis=0)
            wrenches[:, 5] = filtfilt(b, a, wrenches[:, 5], axis=0)

        # Add to interpolated state list
        target_times_list.append(target_times)
        states_list.append(states)
        wrenches_list.append(wrenches)
        if T_world_vial_base is not None:
            T_world_vial_base_list.append(T_world_vial_base)
        if T_world_vial_lip is not None:
            T_world_vial_lip_list.append(T_world_vial_lip)
        if T_world_mocap is not None:
            T_world_mocap_list.append(T_world_mocap)
        if "data/dynamixel" in zarr_data:
            dynamixel_orientationes_list.append(dynamixel_orientationes)

        # Find new episode ends list
        episode_ends_list.append(n_list + len(target_times))
        n_list += len(target_times)

    target_times_all = np.concatenate(target_times_list, axis=0)
    states_all = np.concatenate(states_list, axis=0)
    wrenches_all = np.concatenate(wrenches_list, axis=0)
    T_world_vial_base_all = np.concatenate(T_world_vial_base_list, axis=0) if T_world_vial_base_list else None
    T_world_vial_lip_all = np.concatenate(T_world_vial_lip_list, axis=0) if T_world_vial_lip_list else None
    T_world_mocap_all = np.concatenate(T_world_mocap_list, axis=0) if T_world_mocap_list else None
    if "data/dynamixel" in zarr_data:
        dynamixel_orientationes_all = np.concatenate(dynamixel_orientationes_list, axis=0)
    else:
        dynamixel_orientationes_all = None

    episode_ends_all = np.array(episode_ends_list)

    # Transform all mocap data to world frame
    if T_world_mocap_all is None:
        raise ValueError("T_world_mocap data is required but not found in zarr.")

    #(FIX) Assume the T_world_mocap is constant and use only first value
    # Transform mocap positions to world frame
    states_all_world = np.zeros(states_all.shape)
    for i in range(states_all.shape[0]):
        states_all_world[i,:] =  T_to_pose7(
            pose7_to_T(T_world_mocap_all[0,:]) @ pose7_to_T(states_all[i,:])
            )

    return EpisodeData(
                states=states_all_world, # spatula measurements in world frame
                wrenches=wrenches_all,
                T_world_vialBase=T_world_vial_base_all, # rename with out space
                T_world_vialLip=T_world_vial_lip_all,   # rename with out space
                T_world_mocap=T_world_mocap_all,
                episode_ends=episode_ends_all,
                times=target_times_all, # Rename to times
                images=images_raw,
                images_times=images_times_raw,
            )

def split_episodes(data: EpisodeData) -> list[EpisodeData]:
    """Split a concatenated EpisodeData into one EpisodeData per episode."""
    episodes = []
    prev = 0
    for end in data.episode_ends:
        s = slice(prev, end)
        episodes.append(EpisodeData(
            states=data.states[s],
            wrenches=data.wrenches[s],
            T_world_vialBase=data.T_world_vialBase[s] if data.T_world_vialBase is not None else None,
            T_world_vialLip=data.T_world_vialLip[s] if data.T_world_vialLip is not None else None,
            T_world_mocap=data.T_world_mocap[s] if data.T_world_mocap is not None else None,
            episode_ends=np.array([end - prev]),
            times=data.times[s],
            images=None,       # images aren't indexed the same way
            images_times=None,
        ))
        prev = end
    return episodes

def inverse_compute(episode: EpisodeData, K):
    force_x = episode.wrenches[:, 0]
    force_z = episode.wrenches[:, 2]
    force = - np.sqrt(force_x**2 + force_z**2)
    
    X = episode.states[:,2]

    zft_hat = 1./K * force + X

    return zft_hat

def extract_episode(dataset_name, episode_n: int | list):
    if dataset_name == "calibration":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_calibration_2026-03-04-10-08\data.zarr")
    elif dataset_name == "metal":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_metal_2026-03-04-10-13\data.zarr")
    elif dataset_name == "vial":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_vial_2026-03-04-10-44\data.zarr")
    elif dataset_name == "sponge":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\vitor_sponge_2026-03-04-10-32\data.zarr")
    else:
        print("Undefined Path")
        return
    
    target_hz = 100.0
    lowpass_cutoff_pos = 10.0
    lowpass_cutoff_wrench = 5.0
    butter_order = 2

    episodes = import_data(
        zarr_root=zarr_root,
        target_hz=target_hz,
        lowpass_cutoff_pos=lowpass_cutoff_pos,
        lowpass_cutoff_wrench=lowpass_cutoff_wrench,
        butter_order=butter_order,
    )

    episodes_split = split_episodes(episodes)

    if isinstance(episode_n, list):
        return [episodes_split[i] for i in episode_n]
    else:
        return episodes_split[episode_n]

def main():

    data = "metal"

    if data == "calibration":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_calibration_2026-03-04-10-08\data.zarr")
    elif data == "metal":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_metal_2026-03-04-10-13\data.zarr")
    elif data == "vial":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\victor_vial_2026-03-04-10-44\data.zarr")
    elif data == "sponge":
        zarr_root = Path(r"C:\Users\Victo\OneDrive\Documents\EPFL\Cours\MA4\semester_proj_idiap\code\demonstration\victor_data\vitor_sponge_2026-03-04-10-32\data.zarr")
    else:
        print("Undefined Path")
        return

    target_hz = 100.0
    lowpass_cutoff = None #10.0
    butter_order = 2

    episode = import_data(
        zarr_root=zarr_root,
        target_hz=target_hz,
        lowpass_cutoff=lowpass_cutoff,
        butter_order=butter_order,
    )

    episodes = split_episodes(episode)

    episode = episodes[6]

    K_list = np.linspace(100, 500, 9).tolist()
    zft_list = []

    for K in K_list:
        zft_list.append(inverse_compute(episode, K))

    times = episode.times

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))
    for i, K in enumerate(K_list):
        r, c = divmod(i, 3)
        ax[r, c].plot(times, episode.states[:, 2], label="z")
        ax[r, c].plot(times, zft_list[i], label=f"zft K={K:.1f}")
        ax[r, c].set_ylabel("z")
        ax[r, c].set_xlabel("time [s]")
        ax[r, c].legend()
    fig.suptitle("Position vs Time")

    plt.show()


if __name__ == "__main__":
    main()
