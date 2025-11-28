from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from typing import List

app = FastAPI()

PENDING_DIR = "./system/server/pending_policies"
GLOBAL_POLICY_PATH = "./system/server/global_policy.json"

# ========== Model request ==========
class Policy(BaseModel):
    branch_id: int
    round: int
    actions: List[str]
    Q_values: List[float]

# ========== Upload policy from client ==========
@app.post("/upload_policy")
def upload_policy(policy: Policy):
    os.makedirs(PENDING_DIR, exist_ok=True)
    save_path = f"{PENDING_DIR}/policy_branch_{policy.branch_id}_round_{policy.round}.json"
    with open(save_path, "w") as f:
        json.dump(policy.dict(), f, indent=4)
    return {"message": f"Policy saved to {save_path}"}

# ========== Return global policy =============
@app.get("/global_policy")
def get_global_policy():
    if not os.path.exists(GLOBAL_POLICY_PATH):
        raise HTTPException(status_code=404, detail="Global policy not ready yet")
    with open(GLOBAL_POLICY_PATH) as f:
        return json.load(f)
