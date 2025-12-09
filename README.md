# Federated Multi-Armed Bandits System

## Overview

This project implements a federated multi-armed bandits system where multiple clients (branches) locally train Q-values and send updates to a central server. The server stores local policies, waits for enough agents, triggers a Spark aggregator, and broadcasts the updated global policy back to all clients in real time.

Core features include:

- WebSocket communication between server and agents

- FastAPI backend with async broadcasting

- Spark-based global Q-value aggregator

- Driver orchestration for multi-round training

- Real-time global policy updates to all connected agents

## Requirements
- Python 
- FastAPI
- Uvicorn
- websockets
- requests
- PySpark (spark 4.0.1)
- Hadoop client (hadoop 3.4.0)
- Download winutils.exe, hadoop.dll and hdfs.dll binaries from [a link](https://github.com/kontext-tech/winutils/tree/master/hadoop-3.4.0-win10-x64/bin) for hadoop on windows
- Java 17 jdk

## Running the System
0. Config executable path
   Change the path in driver.py to your executable path
```
SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
HDFS_EXE = r"C:\hadoop\bin\hdfs.cmd"
```
1. Config parameter and generate dataset if necessary
- In server/driver.py
```
ROUNDS = 10           # Number of aggregation round
MIN_AGENTS = 4        # Number of clients
POLL_INTERVAL = 5     # seconds between checks 
ROUND_TIMEOUT = 300   # seconds to wait max for agents in a round
```
- If change number of rounds and agents, in Phan_1/federated_data.py, change the config to generate new dataset. Copy the choosen dataset to client/data before the run
```
#num_clients = MIN_AGENTS
#num_round = ROUNDS
#time_steps is recommended to be 100*ROUNDS
generate_federated_bandits( num_clients=4, k=5, time_steps=1000, num_round = 10, stationary=False, drift_std=2, heterogeneous_noise=True,output_dir=<OUTPUT_DIR>)
```
- In client/bandit.py
```
EPSILON = 0.05   #epsilon parameter for e-greedy action selection
```
- ***IMPORTANCE*** Before the run, delete server/global_policies/global_policy_round_<R>.json,  server/global_policies/global_policy_latest.json/*, server/policies/*, client/log_reward/*
- ***IMPORTANCE*** Also, remember to navigate to /system/
  
2. Start the Server + Driver
   The driver automatically launches the FastAPI WebSocket server:  
   `python3 server/driver.py`  (MacOS)
   `python server/driver.py`   (Windows)
  This does:
- Starts Uvicorn in the background
- Waits for WebSocket clients
- Monitors policies folder for updates
- Runs the Spark aggregator
- Pushes global_policy to clients
- ***NOTICE_1***: Must wait before "Server is ready!" to start clients
- ***NOTICE_2***: Because of unknown reason, the client may experience timeout when connecting to server. In case of that, kill the server driver process and the terminal of client if needed. Restart the server driver and connect again
3. Start Clients
  Each agent is launched with an ID:  
  `python3 client/client.py 1`    
  `python3 client/client.py 2`    
  `python3 client/client.py 3`    
  `python3 client/client.py 4`    

   *** For Windows: navigate to /client/, then run:
   `python client/client.py {clientId}` 
The client will:
- Connect via WebSocket to /ws
- Register its client ID
- Receive the latest global Q-values
- Train locally
- Send local_update messages
- Receive new global policies from server
- When done rounds of aggregation, the process will be ended
## Synthetic Federated Multi-Armed Bandits Dataset

### Dataset Overview

The dataset models multiple clients, each having their own reward distributions, allowing researchers to study:

- Client heterogeneity
- Non-stationary environments
- Performance of bandit algorithms in a federated context


### Structure

For each client (branch), data is generated for multiple rounds, representing repeated interactions with a set of actions (arms). Each round produces a parquet file with the following columns:

| Column          | Description |
|-----------------|-------------|
| `Timestamp`     | The time step within the round |
| `ActionApplied` | The action chosen (arm) by the client |
| `Q_star`        | The true expected reward of the action at this time step |
| `TotalSpent`    | Observed reward (sampled from a normal distribution around `Q_star`) |


### Actions (Arms)

The dataset uses the following actions, simulating different marketing incentives:

- `voucher`
- `freeship`
- `combo`
- `flashsale`
- `loyalty`


### Variants

Two main types of datasets are generated:

1. Non-stationary with heterogeneous noise 

   - Each client may have different reward noise levels
   - Rewards drift over time
   - Suitable for testing algorithms under non-stationary and heterogeneous environments such as e-commerce

2. Stationary with homogeneous noise
   
   - All clients share the same noise level
   - Rewards are stationary (no drift)
   - Useful for baseline testing
  







