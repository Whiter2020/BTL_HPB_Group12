# client_ws.py
import asyncio
import websockets
import json
import sys
import bandit
import os

if len(sys.argv) != 2:
    print("Usage: python client.py <branch_id> ")
    sys.exit(1)

CLIENT_ID = int(sys.argv[1])
ROUND = 1


async def run_client():
    global ROUND
    global CLIENT_ID
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        while True:

            # REGISTER FIRST
            await ws.send(json.dumps({
                "type": "register_ws",
                "client_id": CLIENT_ID
            }))

            # RESPONSE: global policy
            resp = await ws.recv()
            print("[CLIENT] Connected:", resp)

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
            print(res)
            if res["status"]!="Success":
                exit
          
            with open(f"./client/policy/branch_{CLIENT_ID}/policy_branch_{CLIENT_ID}_round_{ROUND}.json", "r") as f:
                data = json.load(f)
            await ws.send(json.dumps({
                "type": "local_update",
                "payload": data
            }))

            resp = await ws.recv()
            data = json.loads(resp)
            print(data)

            if data["type"] == "global_policy":
                print("[CLIENT] Received GLOBAL POLICY:", data["payload"])
                with open(f"./client/global/branch_{CLIENT_ID}/global_policy.json", "w")  as f:
                    json.dump(data["payload"], f, indent=4)
                print("[CLIENT] Updated global policy saved.")
            ROUND += 1

asyncio.run(run_client())
