import numpy as np
import pandas as pd
import os

def generate_federated_bandits(
    num_clients=5,
    k=10,
    time_steps=500,
    num_round = 10,
    stationary=True,
    drift_std=2,
    heterogeneous_noise=True,
    output_dir="federated_dataset"
):

    os.makedirs(output_dir, exist_ok=True)
    q_star = np.random.normal(318, 225, size=k)
    for a in range(len(q_star)):
            while True:
                if q_star[a]> 0:
                    break
                q_star[a]=  np.random.normal(318, 225)



    # Random matrix: shape (k, rounds)
    drift_matrix = np.random.normal(
        loc=0.0,
        scale=drift_std,
        size=(k, num_round)
    )
    print(q_star)
    print(drift_matrix)

    for client_id in range(1, num_clients + 1):

        # --- Client heterogeneity ---
        client_noise = np.random.uniform(80, 150) if heterogeneous_noise else 100

       
        
        actions = ["voucher", "freeship", "combo", "flashsale", "loyalty"]

        for round in range(num_round):
            data = []
            while True:
                actual_size = np.random.normal((time_steps/num_round),100)
                if actual_size > 0:
                    break
                if not stationary:
                    q_star += drift_matrix[:,round]
                    print(q_star)
           
            for t in range(int(actual_size)):

                # Non-stationary drift per client
                
                # For each arm — generate one reward sample per timestep (full environment record)
                for a in range(k):
                    while True:
                        reward = np.random.normal(q_star[a], client_noise)
                        if reward > 0: break

                    data.append((
                        t,
                        actions[a],
                        q_star[a],
                        reward,
                    ))

            df = pd.DataFrame(data, columns=[
                "Timestamp",
                "ActionApplied",
                "Q_star",
                "TotalSpent",
            ])
            
            folder_path = os.path.join(output_dir, f'branch_{client_id}')
            os.makedirs(folder_path, exist_ok=True)
            
            file_path = os.path.join(folder_path, f'customer_log_round_{round+1}.parquet')
            print(f"Dataset generated for: {file_path} with {int(actual_size)}/")

            df.to_parquet(file_path, index=False)

    print(f"Federated dataset generated at: {output_dir}/")

if __name__ == "__main__":
    generate_federated_bandits( num_clients=4, k=5, time_steps=1000, num_round = 10, stationary=False, drift_std=5, heterogeneous_noise=True,output_dir="nonstat_hetero_data")
    generate_federated_bandits( num_clients=4, k=5, time_steps=1000, num_round = 10,stationary=True, drift_std=10, heterogeneous_noise=False,output_dir="stat_data")


