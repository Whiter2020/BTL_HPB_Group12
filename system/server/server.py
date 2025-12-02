# server.py
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict
import subprocess
import uuid
import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ReadGlobalQ").getOrCreate()


HDFS_BASE = "hdfs:///policies"        # nơi agents upload policies
GLOBAL_OUT = "hdfs:///global_policies"
latest_dst = f"{GLOBAL_OUT.rstrip('/')}/global_policy_latest.json"

app = FastAPI()


active_ws: Dict[str, WebSocket] = {}

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

    # 2. Save local JSON temporarily on server
    temp_filename = f"/tmp/{client_id}_{uuid.uuid4().hex}.json"
    with open(temp_filename, "w") as f:
        json.dump(content["payload"], f)

    # 3. HDFS target directory
    hdfs_dir = f"{HDFS_BASE}/round_{round_k}"
    hdfs_file = f"{hdfs_dir}/{client_id}.json"

    # 4. Upload to HDFS
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_dir])
    subprocess.run(["hdfs", "dfs", "-put", "-f", temp_filename, hdfs_file])

    # 5. Remove temp file
    os.remove(temp_filename)

    print(f"[SERVER] Saved update from {client_id} → {hdfs_file}")


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
        
        df = spark.read.option("multiLine", True).json(latest_dst+"/*")
        df.show()
        row = df.collect()[0]  # collect returns a list of Row objects
        data_dict = row.asDict()

        await ws.send_text(json.dumps({"status": "ws_connected", "payload": data_dict}))

        # Listen for incoming messages (local updates)
        while True:
            msg = await ws.receive_text()
            content = json.loads(msg)
            if content["type"] == "local_update":
                await save_local_update(client_id, content)
            print(f"[SERVER] Received from {client_id}: {content}")

    except WebSocketDisconnect:
        print(f"[SERVER] WS disconnected: {client_id}")
        active_ws.pop(client_id, None)

    except Exception as e:
        print(f"[ERROR] WS error from {client_id}: {e}")
        active_ws.pop(client_id, None)



async def broadcast_global_policy(policy: dict):
    message = json.dumps({"type": "global_policy", "payload": policy})

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
