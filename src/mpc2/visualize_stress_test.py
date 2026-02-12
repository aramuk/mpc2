from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def viz(data_path, title: str = "RRT Performance"):
    df = pd.read_csv(data_path)
    df["upload_time"] = df["upload_end"] - df["upload_start"]
    df["compute_time"] = df["compute_end"] - df["compute_start"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for structure, group in df.groupby("graph_structure"):
        group = group.sort_values("global_comms_all_mb").reset_index(drop=True)
        xs = group["global_comms_all_mb"]
        ax1.plot(xs, group["upload_time"], marker="o", label=structure)
        ax2.plot(xs, group["compute_time"], marker="o", label=structure)
        for i, name in enumerate(group["name"]):
            ax1.annotate(name, (i, group["upload_time"].iloc[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
            ax2.annotate(name, (i, group["compute_time"].iloc[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)

    for ax, title in [(ax1, "Upload Time"), (ax2, "Compute Time")]:
        ax.set_title(title)
        ax.set_xlabel("Global Communication (MB)")
        ax.set_ylabel("Time (s)")
        ax.legend()

    fig.tight_layout()
    fig.suptitle(title)
    plt.savefig("rrt_stress_test_times.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    DATA_PATH = Path("data/rrt_stress_test.csv")
    viz(DATA_PATH)