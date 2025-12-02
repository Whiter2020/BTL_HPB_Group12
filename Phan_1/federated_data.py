import numpy as np
import pandas as pd
import os

def generate_federated_bandits(
    num_clients=5,
    k=10,
    time_steps=500,
    stationary=True,
    drift_std=2,
    heterogeneous_noise=True,
    output_dir="federated_dataset"
):

    os.makedirs(output_dir, exist_ok=True)

    for client_id in range(1, num_clients + 1):

        # --- Client heterogeneity ---
        client_noise = np.random.uniform(80, 150) if heterogeneous_noise else 100
        client_drift = np.random.uniform(1, drift_std) if not stationary else 0

       
        # --- Initialize q* values ---

        q_star = np.random.normal(325.6, 227, size=k)
        

        data = []
        actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]
        for t in range(time_steps):

            # Non-stationary drift per client
            if not stationary:
                q_star += np.random.normal(0, client_drift, size=k)

            # For each arm — generate one reward sample per timestep (full environment record)
            for a in range(k):
                reward = np.random.normal(q_star[a], client_noise)

                data.append((
                    t,
                    actions[a],
                    q_star[a],
                    reward,
                ))
        df = pd.DataFrame(data, columns=[
            "T",
            "ActionApplied",
            "Q_star",
            "TotalSpent",
        ])

        df.to_csv(f"{output_dir}/client_{client_id}.csv", index=False)

    print(f"Federated dataset generated at: {output_dir}/")

if __name__ == "__main__":
    generate_federated_bandits( num_clients=4, k=5, time_steps=1500, stationary=False, drift_std=2, heterogeneous_noise=True,output_dir="nonstat_hetero")
    generate_federated_bandits( num_clients=4, k=5, time_steps=1500, stationary=True, drift_std=2, heterogeneous_noise=False,output_dir="stat")