import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

def load_avg_reward_series_global_time(path):
    """
    Load (global_time, avg_reward) pairs from a log file.
    Time is made monotonic by offsetting at each round.
    """
    with open(path, "r") as f:
        data = json.load(f)

    global_times = []
    rewards = []
    round_start_time = []
    time_offset = 0

    for round_data in data["rounds"]:
        round_log = round_data["log"]

        if not round_log:
            continue
        print(f"{time_offset} begin")
        # Extract local times and rewards
        local_times = [entry["time"] for entry in round_log]
        local_rewards = [entry["avg_reward"] for entry in round_log]

        """for t in local_times:
            with open("output.txt", "a") as f:
                f.write(f"time offset {time_offset} + t {t} = global time {t+time_offset}\n")"""
        # Apply offset
        global_times.extend([t + time_offset for t in local_times])
        rewards.extend(local_rewards)
        round_start_time.append(time_offset)

        # Update offset for next round
        time_offset += max(local_times) + 1
        #print(f"{time_offset} updated")

    return global_times, rewards,round_start_time

round_files = sorted(Path("system/client/data/branch_3").glob("customer_log_round_*.parquet"))

dfs = []
global_offset = 0

for round_id, file in enumerate(round_files, start=1):
    df = pd.read_parquet(file)

    df["Timestamp"] = df["Timestamp"].astype(int)
    df["Q_star"] = df["Q_star"].astype(float)

    # Compute global timestamp
    df["GlobalTimestamp"] = df["Timestamp"] + global_offset
    df["Round"] = round_id

    dfs.append(df)

    # Update offset (robust even if round lengths differ)
    global_offset += df["Timestamp"].max() + 1

all_data = pd.concat(dfs, ignore_index=True)
print(all_data)
max_possible = (
    all_data.groupby("GlobalTimestamp", as_index=False)["TotalSpent"]
      .max()
      .rename(columns={"TotalSpent": "MaxPossible"})
      .sort_values("GlobalTimestamp")
)
max_possible["AvgUpToT"] = (
    max_possible["MaxPossible"]
    .expanding()
    .mean()
)

# -------- Load two logs --------
log1_path = "D:\\HK241\BTL_HPB_Group12\system\client\log_reward copy\\agent_3_rewards.json"
log2_path = "D:\\HK241\BTL_HPB_Group12\system\client\log_reward\\agent_3_rewards.json"

t1, r1 , rst1 = load_avg_reward_series_global_time(log1_path)
t2, r2 , rst2 = load_avg_reward_series_global_time(log2_path)
print(len(t1),len(r1))

# -------- Plot --------
plt.figure()

#plt.plot(max_possible["GlobalTimestamp"], max_possible["AvgUpToT"],label = "Expected results")

# Draw round boundary markers
round_starts = (
    all_data.groupby("Round")["GlobalTimestamp"].min().tolist()
)

for t in round_starts[1:]:
    plt.axvline(t, linestyle="--", linewidth=1, alpha=0.5)

plt.plot(t1, r1, label="Log with federated learning")
plt.plot(t2, r2, label="Log without federated learning")

plt.xlabel("Global Time")
plt.ylabel("Average Reward")
plt.title("Average Reward Comparison (Global Monotonic Time)")
plt.legend()
plt.grid(True)

for t in rst1[1:]:
    plt.axvline(
        x=t,
        linestyle="--",
        linewidth=1,
    )


plt.savefig("my_plot.png", dpi=300, bbox_inches="tight")

plt.show()
