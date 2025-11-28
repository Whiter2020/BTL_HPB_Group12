import sys
import os
import json
import pandas as pd
import numpy as np

"""
run: python bandit.py <branch_id> <round_id>
python bandit.py 1 1
"""

# Parse input arguments
# ------------------------------------------------------------
if len(sys.argv) != 3:
    print("Usage: python bandit.py <branch_id> <round_id>")
    sys.exit(1)

branch_id = int(sys.argv[1])
round_id = int(sys.argv[2])

print(f"Local RL Training | Branch {branch_id} | Round {round_id} ===")

# Load local dataset
# ------------------------------------------------------------
data_path = f"./data/branch_{branch_id}/customer_log.parquet"
if not os.path.exists(data_path):
    print(f"Error: Dataset not found: {data_path}")
    sys.exit(1)

df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} records")

# Debug information about dataset structure
print("\n--- Dataset Columns ---")
print(df.columns.tolist())

# Validate reward column
# ------------------------------------------------------------
if "total_spent" not in df.columns:
    print("The RL model cannot learn without a reward")
    sys.exit(1)

# Load global policy if available (Warm start)
# ------------------------------------------------------------
global_policy_path = f"./client/global/branch_{branch_id}/global_policy.json"
if os.path.exists(global_policy_path):
    try:
        with open(global_policy_path) as f:
            global_policy = json.load(f)
        if isinstance(global_policy, dict) and "Q_values" in global_policy:
            Q_values = np.array(global_policy["Q_values"], dtype=float)
            print("Warm start: initialized from global policy")
        else:
            Q_values = np.zeros(5)
            print("Global policy found but missing Q_values -> Cold start")
    except Exception as e:
        Q_values = np.zeros(5)
        print(f"Error reading global policy ({e}) -> Cold start")
else:
    Q_values = np.zeros(5)
    print("No global policy found -> Cold start")

# Define action space and hyperparameters
# ------------------------------------------------------------
actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]
alpha = 0.01  # learning rate


# Local RL training (Multi-Armed Bandit)
# Each row represents one customer. The reward signal corresponds
# to the total spending value of that customer.
# ------------------------------------------------------------
print("\nStarting training...")
for idx, row in df.iterrows():
    # Placeholder action selection (round-robin)
    a = idx % len(actions)

    reward = float(row.get("total_spent", 0))
    Q_values[a] = Q_values[a] + alpha * (reward - Q_values[a])

print("Training completed.")

# Save local policy for federated aggregation
# ------------------------------------------------------------
save_dir = f"./client/policy/branch_{branch_id}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/policy_branch_{branch_id}_round_{round_id}.json"

policy = {
    "branch_id": branch_id,
    "round": round_id,
    "actions": actions,
    "Q_values": Q_values.tolist(),
    "hyperparameters": {
        "alpha": alpha,
        "dataset_size": len(df)
    }
}

with open(save_path, "w") as f:
    json.dump(policy, f, indent=4)

print(f"\nLocal policy saved to: {save_path}")
