# server.py
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict
import subprocess
import uuid
import os
from pydantic import BaseModel
from pyspark.sql import SparkSession
from contextlib import asynccontextmanager
import asyncio
from fastapi import Body


spark = SparkSession.builder.appName("ReadGlobalQ").getOrCreate()
dashboard_clients = set()      # all websocket connections from dashboards

HDFS_BASE = "policies"        # nơi agents upload policies
GLOBAL_OUT = "global_policies"
latest_dst = f"{GLOBAL_OUT.rstrip('/')}/global_policy_latest.json"
loop = None

@asynccontextmanager
async def lifespan(app):
    global loop
    loop = asyncio.get_running_loop()
    print("Server started!")
    yield
    print("Server shutting down!")



app = FastAPI(lifespan=lifespan)


active_ws: Dict[str, WebSocket] = {}
# Store local rewards for each round
LOCAL_REWARDS = {}  # { round_number: {branch_id: reward} }

async def save_local_update(client_id: str, content: dict):
    """
    content structure example:
  {
    "type": "local_update"
    "payload": {
        "branch_id": 2,
        "round": 1,
        "actions": [
            "voucher",
            "freeship",
            "combo",
            "flashsale",
            "loyalty"
        ],
        "Q_values": [
            1415.9094210804378,
            1264.7199899611187,
            2009.4087715203937,
            1695.4470336328434,
            1912.512614231632
        ],
        "hyperparameters": {
            "alpha": 0.01,
            "dataset_size": 1085
        }
    }
    
}
    """

    # 1. Read round parameter
    round_k = content.get("round")
    if round_k is None:
        print("[SERVER] ERROR: Missing 'round' in client update")
        return
    client_dir = f"{HDFS_BASE}/round_{round_k}"
    os.makedirs(client_dir, exist_ok=True)  # create the directory if it doesn't exist


    # 2. Save local JSON temporarily on server
    temp_filename = f"{HDFS_BASE}/round_{round_k}/brand_{client_id}.json"
    with open(temp_filename, "w") as f:
        json.dump(content, f)

 


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        # Expect first message: register through WS
        msg = await ws.receive_text()
        data = json.loads(msg)

        if data.get("type") != "register_ws":
            await ws.send_text(json.dumps({"error": "Must register first"}))
            await ws.close()
            return

        client_id = data["client_id"]

        # Save WebSocket connection
        active_ws[client_id] = ws
        print(f"[SERVER] WS connected: {client_id}")
        
        if os.path.exists(latest_dst):
            try:
                df = spark.read.option("multiLine", True).json(latest_dst+"/*")
                df.show()
                if df.count() > 0:
                    row = df.collect()[0]
                    data_dict = row.asDict()
                else:
                    data_dict=None
            except Exception as e:
                data_dict=None

                print(f"[ERROR] Failed to read JSON from {latest_dst}: {e}")
        else:
            data_dict=None
            print(f"[WARN] File does not exist: {latest_dst}")

        await ws.send_text(json.dumps({"status": "ws_connected", "payload": data_dict}))

        # Listen for incoming messages (local updates)
        while True:
            msg = await ws.receive_text()
            content = json.loads(msg)
            if content["type"] == "local_update":
                await save_local_update(client_id, content["payload"])
            print(f"[SERVER] Received from {client_id}: {content}")

    except WebSocketDisconnect:
        print(f"[SERVER] WS disconnected: {client_id}")
        active_ws.pop(client_id, None)

    except Exception as e:
        print(f"[ERROR] WS error from {client_id}: {e}")
        active_ws.pop(client_id, None)



async def broadcast_global_policy(policy: dict):
    message = json.dumps({"type": "global_policy", "payload": policy})
    print(policy)

    lost_clients = []

    for client_id, ws in active_ws.items():
        try:
            await ws.send_text(message)
        except Exception:
            lost_clients.append(client_id)

    # Remove offline clients
    for cid in lost_clients:
        print(f"[SERVER] Removing offline client: {cid}")
        active_ws.pop(cid, None)

class Policy(BaseModel):
    policy: dict

@app.post("/broadcast")
async def broadcast(policy: Policy):
    # print("[SERVER] /broadcast endpoint was called")
    # print(policy)
    asyncio.create_task(broadcast_global_policy(policy.policy))
    # await broadcast_global_policy(policy.policy)
    return {"status": "ok"}

class RewardInput(BaseModel):
    branch_id: int
    round: int
    reward: float

@app.post("/reward")
async def receive_reward(data: RewardInput):
    r = data.round
    bid = data.branch_id
    rew = data.reward

    if r not in LOCAL_REWARDS:
        LOCAL_REWARDS[r] = {}

    LOCAL_REWARDS[r][bid] = rew
    print(f"[SERVER] Received reward from branch {bid} at round {r}: {rew}")

    return {"status": "ok"}

@app.get("/reward/{round}")
async def get_reward_table(round: int):
    rewards = LOCAL_REWARDS.get(round, {})
    return {
        "status": "ok",
        "round": round,
        "payload": rewards
    }

class RoundLog(BaseModel):
    round: int
    local_rewards: Dict[str, float]
    global_reward: float
    runtime: float
    spark_latency: float
    throughput: float
    

@app.post("/log_event")
async def log_event(data: RoundLog):
    global LATEST_LOG
    LATEST_LOG = data
    print(f"[SERVER] New log event for round {data.round}")

    asyncio.create_task(broadcast_log_event(data))

    return {"status": "ok"}

async def broadcast_log_event(log_k: RoundLog):
    message = json.dumps({
        "type": "log_update",
        "log": log_k.model_dump()
    })
    
    print("[BROADCAST] New log event for round", log_k.round)

    dead = []
    print("[SERVER] Broadcasting to", len(dashboard_clients), "dashboard clients")
    for ws in dashboard_clients:
        try:
            await ws.send_text(message)
        except:
            dead.append(ws)

    for ws in dead:
        dashboard_clients.remove(ws)


@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    dashboard_clients.add(ws)
    print("[SERVER] Dashboard connected")

    try:
        while True:
            await ws.receive_text()   # dashboard không gửi gì, chỉ giữ connection
    except:
        dashboard_clients.remove(ws)
        print("[SERVER] Dashboard disconnected")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "server:app",
#         host="127.0.0.1",
#         port=8000,
#         reload=False
#     )
