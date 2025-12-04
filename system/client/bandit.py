import sys
import os
import json
import pandas as pd
import numpy as np
import random

"""
run: python bandit.py <branch_id> <round_id>
python bandit.py 1 1
"""
ACTION_SELECTION = "e-greedy"
EPSILON = 0.05

def bandit(branch_id, round_id):
    print(f"Local RL Training | Branch {branch_id} | Round {round_id} ===")

    # Load local dataset
    # ------------------------------------------------------------
    data_path = f"./data/branch_{branch_id}/customer_log_round_{round_id}.parquet"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found: {data_path}")
        return {"status": "Error"}

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records")
    

    # Debug information about dataset structure
    print("\n--- Dataset Columns ---")
    print(df.columns.tolist())

    # Validate reward column
    # ------------------------------------------------------------
    if "TotalSpent" not in df.columns:
        print("The RL model cannot learn without a reward")
        return {"status": "Error"}

    # Load global policy if available (Warm start)
    # ------------------------------------------------------------
    global_policy_path = f"./client/global/branch_{branch_id}/global_policy.json"
    if os.path.exists(global_policy_path):
        try:
            with open(global_policy_path) as f:
                global_policy = json.load(f)
            if isinstance(global_policy, dict) and "global_Q" in global_policy:
                Q_values = np.array(global_policy["global_Q"], dtype=float)
                actions = np.array(global_policy["actions"])
                alpha = np.array(global_policy["hyperparameters"]["alpha"])
                method = ACTION_SELECTION
                print("Warm start: initialized from global policy")
            else:
                Q_values = np.zeros(5)
                actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]
                alpha = 0.01  # learning rate
                method = "random"
                print("Global policy found but missing Q_values -> Cold start")
        except Exception as e:
            Q_values = np.zeros(5)
            actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]
            alpha = 0.01  # learning rate
            method = "random"
            print(f"Error reading global policy ({e}) -> Cold start")
    else:
        Q_values = np.zeros(5)
        actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]
        alpha = 0.01  # learning rate
        method = "random"
        print("No global policy found -> Cold start")

    time_stamp = len(df)/len(actions)
    print(time_stamp)

    # Local RL training (Multi-Armed Bandit)
    # Each row represents one customer. The reward signal corresponds
    # to the total spending value of that customer.
    # ------------------------------------------------------------
    print("\nStarting training...")

    
    
    os.makedirs(f"log_reward", exist_ok=True)
    log_path = os.path.join(f"log_reward", f"agent_{branch_id}_rewards.json")
    
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_data = json.load(f)
        reward = log_data["reward"]
        time_taken = log_data["time_taken"]
    else:
        log_data = {
            "reward" : 0,
            "time_taken": 0,
            "branch_id": branch_id,
            "rounds": []
        }
        reward = 0
        time_taken = 0
    round_entry = None
    
    for r in log_data["rounds"]:
        if r["round"] == round_id:
            round_entry = r
            break

    if round_entry is None:
        round_entry = {
            "round": round_id,
            "log": []
        }
        log_data["rounds"].append(round_entry)


    for time in range (int(time_stamp)):
        if method == "e-greedy":
            if random.random() < EPSILON:
                a = random.choice(range(len(actions)))
            else:
                a = Q_values.index(max(Q_values))
        else:
            a = random.choice(range(len(actions)))
        for idx in range(time*5,time*5+5):
            
            row_dict = df.iloc[idx].to_dict()
        
            if row_dict["ActionApplied"] == actions[a]:
                if (row_dict["TotalSpent"]<0):
                    print("Row"+str(idx))
                Q_values[a]=Q_values[a]+alpha*(row_dict["TotalSpent"]-Q_values[a])
                reward += row_dict["TotalSpent"]
                time_taken += 1
                avg_reward = reward / time_taken
                round_entry["log"].append({
                    "time": row_dict["Timestamp"],
                    "avg_reward": float(avg_reward)
                })

        print("Training completed.")
    print("[CLIENT]")

   
  
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
            "dataset_size": time_stamp
        }
    }

    with open(save_path, "w") as f:
        json.dump(policy, f, indent=4)

    log_data["reward"] = reward
    log_data["time_taken"] = time_taken    
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nLocal policy saved to: {save_path}")
    return {"status": "Success"}

