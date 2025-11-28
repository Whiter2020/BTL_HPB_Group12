import sys
import requests
import json
import os

"""
run: python fetch_global_policy.py <branch_id>
python fetch_global_policy.py 1
"""

if len(sys.argv) != 2:
    print("Usage: python fetch_global_policy.py <branch_id>")
    sys.exit(1)

branch_id = int(sys.argv[1])

url = "http://127.0.0.1:8000/global_policy"   # server endpoint

try:
    response = requests.get(url)
    response.raise_for_status()
    policy = response.json()
except Exception as e:
    print("Lỗi khi lấy global policy:", e)
    sys.exit(1)

save_dir = f"./client/global/branch_{branch_id}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/global_policy.json"

with open(save_path, "w") as f:
    json.dump(policy, f, indent=4)

print(f"Global policy saved: {save_path}")
print("Nội dung:")
print(json.dumps(policy, indent=4))
