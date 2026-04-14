import numpy as np
from unpack_data.victor_io_zarr import extract_episode, EpisodeData
import matplotlib.pyplot as plt



def main():
    
    name = "calibration_2"
    n = [0,1,2,3,4,5]
    episodes = extract_episode(name, n)

    # Plot drift
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # for i, episode in enumerate(episodes):
    #     dt = episode.times[1] - episode.times[0]
    #     force = episode.wrenches[:, :3]
    #     torque = episode.wrenches[:, 3:6]
        
    #     dforce = np.diff(force[::10], axis=0) / 10*dt
    #     dtorque = np.diff(torque[::10], axis=0) / 10*dt
        
    #     time = episode.times[:-1:10]
        
    #     ax1.plot(range(len(dforce)), dforce, label=f"Trial {i}")
    #     ax2.plot(range(len(dtorque)), dtorque, label=f"Trial {i}")

    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Force drift (N/s)")
    # ax1.set_title("Force Drift")
    # ax1.legend()
    # ax1.grid()

    # ax2.set_xlabel("Time (s)")
    # ax2.set_ylabel("Torque drift (Nm/s)")
    # ax2.set_title("Torque Drift")
    # ax2.legend()
    # ax2.grid()

    # plt.tight_layout()
    # plt.show()

    # Compute drift

    force_drift = []
    torque_drift = []

    for episode in episodes:
        dt = episode.times[-1] - episode.times[0]
        force = episode.wrenches[:, :3]
        torque = episode.wrenches[:, 3:6]

        dforce = (force[-1] - force[0]) /dt
        dtorque = (torque[-1] - torque[0]) /dt

        force_drift.append(dforce)
        torque_drift.append(dtorque)

    force_labels = ["Fx", "Fy", "Fz"]
    torque_labels = ["Tx", "Ty", "Tz"]

    for i, (f, t) in enumerate(zip(force_drift, torque_drift)):
        print(f"Trial {i}")
        f = abs(f)
        t = abs(t)
        print(f"\tFx: {f[0]:.6f} \tFy: {f[1]:.6f} \tFz: {f[2]:.6f} N/s")
        print(f"\tTx: {t[0]:.6f} \tTy: {t[1]:.6f} \tTz: {t[2]:.6f} Nm/s")

    print(f"Average")
    force_drift = np.array(np.abs(force_drift))
    torque_drift = np.array(np.abs(torque_drift))
    avg_drift_force = np.mean(force_drift, axis=0)*1000*3600
    avg_drift_torque = np.mean(torque_drift, axis=0)*1000*3600
    print(f"\tFx: {avg_drift_force[0]:.1f} \tFy: {avg_drift_force[1]:.1f} \tFz: {avg_drift_force[2]:.1f} mN/h")
    print(f"\tTx: {avg_drift_torque[0]:.1f} \tTy: {avg_drift_torque[1]:.1f} \tTz: {avg_drift_torque[2]:.1f} mNm/h")
    



if __name__ == "__main__":
    main()