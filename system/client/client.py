# client_ws.py
import asyncio
import websockets
import json
import sys
import bandit
import os
import requests
import shutil

if len(sys.argv) != 2:
    print("Usage: python client.py <branch_id> ")
    sys.exit(1)

CLIENT_ID = int(sys.argv[1])
ROUND = 1

def cleanup_client_directories(client_id):
    """
    Clean up:
        - log_reward/agent_{id}_reward.json
        - client/policy/branch_{id}/
        - client/global/branch_{id}/
    """

    # === Clean log_reward file ===
    reward_log = f"log_reward/agent_{client_id}_rewards.json"
    if os.path.exists(reward_log):
        # print(f"[CLEANUP] Removing existing reward log: {reward_log}")
        os.remove(reward_log)

    # === Clean policy folder ===
    policy_folder = f"client/policy/branch_{client_id}"
    if os.path.exists(policy_folder):
        print(f"[CLEANUP] Removing old policy folder: {policy_folder}")
        shutil.rmtree(policy_folder)

    os.makedirs(policy_folder, exist_ok=True)
    # print(f"[CLEANUP] Created empty folder: {policy_folder}")

    # === Clean global folder ===
    global_folder = f"client/global/branch_{client_id}"
    if os.path.exists(global_folder):
        # print(f"[CLEANUP] Removing old global folder: {global_folder}")
        shutil.rmtree(global_folder)

    os.makedirs(global_folder, exist_ok=True)
    # print(f"[CLEANUP] Created empty folder: {global_folder}")



async def run_client():
    global ROUND
    global CLIENT_ID
    
    cleanup_client_directories(CLIENT_ID)
    
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        while True:

            # REGISTER FIRST
            await ws.send(json.dumps({
                "type": "register_ws",
                "client_id": CLIENT_ID
            }))

            # RESPONSE: global policy
            resp = await ws.recv()
            # print("[CLIENT] Connected:", resp)

            try:
                data = json.loads(resp)
            except Exception:
                print("[CLIENT ERROR] Invalid JSON from server")
                

            # Check server error
            if "error" in data:
                print(f"[CLIENT ERROR] Server rejected registration: {data['error']}")
            else: 
                client_dir = f"./client/global/branch_{CLIENT_ID}"
                os.makedirs(client_dir, exist_ok=True)  # create the directory if it doesn't exist

                with open(f"./client/global/branch_{CLIENT_ID}/global_policy.json", "w") as f:
                    json.dump(data["payload"], f, indent=4) 
                break
            
        print("[CLIENT] Successfully registered with server!")



        # Listen for updates
        while True:
            
            res = bandit.bandit(CLIENT_ID,ROUND)
            # print("res", res)
            if res["status"]!="Success":
                exit
                
            try:
                reward_value = res.get("reward")
                if reward_value is not None:
                    requests.post(
                        "http://127.0.0.1:8000/reward",
                        json={
                            "branch_id": CLIENT_ID,
                            "round": ROUND,
                            "reward": reward_value
                        },
                        timeout=10
                    )
                    print(f"[CLIENT] Sent reward to server: round={ROUND}, reward={reward_value}")
                else:
                    print("[CLIENT WARNING] No reward found in result!")
            except Exception as e:
                print(f"[CLIENT ERROR] Failed to send reward to server: {e}")
            ### END NEW

          
            with open(f"./client/policy/branch_{CLIENT_ID}/policy_branch_{CLIENT_ID}_round_{ROUND}.json", "r") as f:
                data = json.load(f)
            await ws.send(json.dumps({
                "type": "local_update",
                "payload": data
            }))

            resp = await ws.recv()
            data = json.loads(resp)
            # print(data)

            if data["type"] == "global_policy":
                # print("[CLIENT] Received GLOBAL POLICY:", data["payload"])
                with open(f"./client/global/branch_{CLIENT_ID}/global_policy.json", "w")  as f:
                    json.dump(data["payload"], f, indent=4)
                print("[CLIENT] Updated global policy saved.")
            ROUND += 1

asyncio.run(run_client())
