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
- PySpark
- Hadoop client 
- requests

## Running the System
1. Start the Server + Driver
   The driver automatically launches the FastAPI WebSocket server:  
   `python3 server/driver.py`  
  This does:
- Starts Uvicorn in the background
- Waits for WebSocket clients
- Monitors policies folder for updates
- Runs the Spark aggregator
- Pushes global_policy to clients
2. Start Clients
  Each agent is launched with an ID:  
  `python3 client/client.py 1`    
  `python3 client/client.py 2`    
  `python3 client/client.py 3`    
  `python3 client/client.py 4`    
The client will:
- Connect via WebSocket to /ws
- Register its client ID
- Receive the latest global Q-values
- Train locally
- Send local_update messages
- Receive new global policies from server
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






