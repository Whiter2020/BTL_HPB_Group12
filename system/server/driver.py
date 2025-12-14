# driver.py
import os
import subprocess
import asyncio
import time
import sys
from datetime import datetime
from pathlib import Path
import uvicorn
import socket
import server
from server import app
from pyspark.sql import SparkSession
import requests
import json
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process
import webbrowser


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# store rewards received from agents
LOCAL_REWARDS = {}        # LOCAL_REWARDS[round][branch_id] = reward
GLOBAL_REWARDS = {}       # GLOBAL_REWARDS[round] = global_reward
SPARK_LATENCY = {}        # SPARK_LATENCY[round]
ROUND_RUNTIME = {}        # ROUND_RUNTIME[round]
THROUGHPUT = {}           # THROUGHPUT[round]

spark = SparkSession.builder.appName("ReadGlobalQ").getOrCreate()


# Config
HDFS_BASE = "policies"        # nơi agents upload policies
GLOBAL_OUT = "global_policies"
AGG_SCRIPT = "aggregator_qvalues.py"  # đường dẫn trên FS của cluster
SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
HDFS_EXE = r"C:\hadoop\bin\hdfs.cmd"

ROUNDS = 10
MIN_ROUNDS = 1
MIN_AGENTS = 2        # require at least this many agent files per round before aggregating
MAX_AGENTS = 4
POLL_INTERVAL = 5     # seconds between checks (adjust)
ROUND_TIMEOUT = 300   # seconds to wait max for agents in a round

def cleanup_directories():
    """
    Cleanup local folders:
        - policies/
        - global_policies/
        - logs/
    And cleanup HDFS directories for:
        - policies/*
        - global_policies/*
    """

    # ====== Local folder cleanup ======
    folders = ["policies", "global_policies", "logs", "output"]

    for folder in folders:
        if os.path.exists(folder):
            print(f"[CLEANUP] Removing local folder: {folder}")
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print(f"[CLEANUP] Recreated empty folder: {folder}")

    # ====== HDFS cleanup ======
    try:
        print("[CLEANUP] Removing HDFS: policies/*")
        subprocess.run([HDFS_EXE, "dfs", "-rm", "-r", "-f", "policies/*"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("[CLEANUP] Removing HDFS: global_policies/*")
        subprocess.run([HDFS_EXE, "dfs", "-rm", "-r", "-f", "global_policies/*"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print("[WARN] HDFS cleanup failed:", e)


def wait_for_server(host, port, timeout=10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print("Server is ready!")
                return True
        except (ConnectionRefusedError, OSError):
            if time.time() - start_time > timeout:
                print("Timeout waiting for server.")
                return False
            time.sleep(0.5)


def hdfs_ls(pattern):
    # uses 'hdfs dfs -ls' to list files; adjust if using 'hadoop fs -ls'
    cmd = [HDFS_EXE, "dfs", "-ls", pattern]
    try:
        print(cmd)
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
        lines = out.strip().splitlines()
        # lines header or permission strings. parse filenames at end of each line
        files = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 8:
                files.append(parts[-1])
        return files
    except subprocess.CalledProcessError as e:
        return []

def run_aggregator(policies_dir, out_dir, round_number):
    spark_start = time.time()
    cmd = [SPARK_SUBMIT, AGG_SCRIPT, policies_dir, out_dir, str(round_number)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(proc.stdout.decode())
    if proc.returncode != 0:
        print("Aggregator failed:", proc.stderr.decode())
    spark_latency = time.time() - spark_start

    SPARK_LATENCY[round_number] = spark_latency
    return proc.returncode == 0

def save_output_charts(local_reward_log, global_reward_log, latency_log, throughput_log):
    os.makedirs("output", exist_ok=True)

    # Convert dict -> sorted list of tuples
    latency_items = sorted(latency_log.items())
    throughput_items = sorted(throughput_log.items())

    # === Build reward table with global + local rewards ===
    rows = []

    for round_id in sorted(global_reward_log.keys()):
        row = {
            "round": round_id,
            "global_reward": global_reward_log[round_id]
        }

        # Add branch rewards from local_reward_log
        local_rewards = local_reward_log.get(round_id, {})
        for branch_id, reward_val in local_rewards.items():
            row[f"branch_{branch_id}"] = reward_val

        rows.append(row)
        
    # print("ROWS:", rows)

    df_reward = pd.DataFrame(rows)
    df_latency = pd.DataFrame(latency_items, columns=["round", "latency_ms"])
    df_throughput = pd.DataFrame(throughput_items, columns=["round", "throughput"])

    # === Save CSV ===
    df_reward.to_csv("output/reward.csv", index=False)
    df_latency.to_csv("output/latency.csv", index=False)
    df_throughput.to_csv("output/throughput.csv", index=False)

    # === Plot reward chart (global + all branches) ===
    plt.figure()

    # Plot global reward
    plt.plot(
        df_reward["round"],
        df_reward["global_reward"],
        marker="o",
        label="global_reward"
    )

    # Plot branch rewards dynamically
    for col in df_reward.columns:
        if col.startswith("branch_"):
            plt.plot(
                df_reward["round"],
                df_reward[col],
                marker="o",
                label=col
            )

    plt.title("Reward per Round (Global vs Branches)")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig("output/reward_all_lines.png")
    plt.close()

    # === Plot latency chart ===
    plt.figure()
    plt.plot(df_latency["round"], df_latency["latency_ms"], marker="o")
    plt.title("Spark Latency per Round")
    plt.xlabel("Round")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.savefig("output/latency.png")
    plt.close()

    # === Plot throughput chart ===
    plt.figure()
    plt.plot(df_throughput["round"], df_throughput["throughput"], marker="o")
    plt.title("System Throughput per Round")
    plt.xlabel("Round")
    plt.ylabel("Actions / Second")
    plt.grid(True)
    plt.savefig("output/throughput.png")
    plt.close()

    print("[DRIVER] Exported charts and logs to ./output/")
    
def launch_clients_subprocess(num_clients):
    processes = []
    client_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "client"))
    client_script = os.path.join(client_dir, "client.py")

    for cid in range(1, num_clients + 1):
        print(f"[DRIVER] Launching client {cid}...")
        proc = subprocess.Popen(
            [sys.executable, client_script, str(cid)],
            cwd=client_dir,
            shell=False
        )
        processes.append(proc)
    
    return processes

def launch_dashboard_subprocess():
    dashboard_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dashboard.py")
    )
    print(f"[DRIVER] Launching dashboard: {dashboard_path}")
    
    proc = subprocess.Popen(
        [sys.executable, dashboard_path],
        shell=False
    )
    return proc

def main():
    """uvicorn.run(
    app,
    host="127.0.0.1",
    port=8000,
    log_level="info"
    )"""
    if len(sys.argv) != 3:
        print("Usage: python driver.py <num_clients> <rounds>")
        sys.exit(1)

    NUM_CLIENTS = int(sys.argv[1])
    ROUNDS = int(sys.argv[2])
    
    if ROUNDS < MIN_ROUNDS:
        print(f"Error: ROUNDS must be >= {MIN_ROUNDS}")
        sys.exit(1)
        
    if NUM_CLIENTS < MIN_AGENTS:
        print(f"Error: NUM_CLIENTS must be >= {MIN_AGENTS}")
        sys.exit(1)
    
    if NUM_CLIENTS > MAX_AGENTS:
        print(f"Error: NUM_CLIENTS must be <= {MAX_AGENTS}")
        sys.exit(1)

    cleanup_directories()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting Server...")
    server_proc = subprocess.Popen(
    [sys.executable,"-m","uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=current_dir, shell=True)
    print("Server started in background.", server_proc.pid)

    if not wait_for_server("127.0.0.1", 8000, timeout=30):
        server_proc.kill()
        raise RuntimeError("Server did not start in time!")
    time.sleep(2)
    
    # client_procs = launch_clients_subprocess(NUM_CLIENTS)
    
    # === Launch Dashboard ===
    # dashboard_proc = launch_dashboard_subprocess()
    # time.sleep(5)
    # webbrowser.open("http://127.0.0.1:7860")

    try:
        for r in range(1, ROUNDS+1):
            policies_dir = f"{HDFS_BASE}/round_{r}"
            print(f"\n=== Starting round {r} at {datetime.now().isoformat()} ===")
            # Wait until at least MIN_AGENTS files exist (or timeout)
            start = time.time()
            while True:
                print(policies_dir)
                files = hdfs_ls(policies_dir)
                THROUGHPUT[r] = len(files) / max(1, (time.time() - start))
                if len(files) >= NUM_CLIENTS:
                    print(f"Found {len(files)} agent files.")
                    break
                if time.time() - start > ROUND_TIMEOUT:
                    print(f"Timeout waiting for agents for round {r}. Found {len(files)}. Proceeding anyway.")
                    break
                print(f"Waiting for agents... found {len(files)} (need {NUM_CLIENTS}). Checking again in {POLL_INTERVAL}s.")
                time.sleep(POLL_INTERVAL)

            # Call aggregator
            success = run_aggregator(f"{HDFS_BASE}/round_{r}/", GLOBAL_OUT, r)
            if not success:
                print(f"Aggregation for round {r} failed. You can retry manually.")
            else:
                print(f"Aggregation for round {r} succeeded. global saved as global_policy_round_{r}.json")

            latest_src = f"{GLOBAL_OUT.rstrip('/')}/global_policy_round_{r}.json"
            latest_dst = f"{GLOBAL_OUT.rstrip('/')}/global_policy_latest.json"
            try:
                subprocess.run([HDFS_EXE, "dfs", "-rm", "-r", latest_dst], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run([HDFS_EXE, "dfs", "-cp", latest_src, latest_dst], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print([HDFS_EXE, "dfs", "-cp", latest_src, latest_dst])
                print("Updated latest pointer to", latest_dst)
            except Exception as e:
                print("Warning: couldn't update latest pointer:", e)
            # small wait or proceed immediately to next round
            df = spark.read.option("multiLine", True).json(latest_dst+"/*")
            df.show()
            row = df.collect()[0]  # collect returns a list of Row objects
            data_dict = row.asDict()
            print("data_dict", data_dict)

            requests.post(
                "http://127.0.0.1:8000/broadcast",
                json={"policy": data_dict},
                timeout=30
            )
            
            res = requests.get(
                "http://127.0.0.1:8000/reward/" + str(r),
                timeout=30
            )
            
            print("reward", res.json())
            
            local_rewards = res.json().get("payload", {})

            
            LOCAL_REWARDS[r] = local_rewards
            
            if local_rewards:
                global_reward = sum(local_rewards.values()) / len(local_rewards)
            else:
                global_reward = 0
            
            GLOBAL_REWARDS[r] = global_reward
            
            ROUND_RUNTIME[r] = time.time() - start
            
            log = {
                "round": r,
                "local_rewards": LOCAL_REWARDS.get(r, {}),
                "global_reward": GLOBAL_REWARDS.get(r),
                "runtime": ROUND_RUNTIME[r],
                "spark_latency": SPARK_LATENCY[r],
                "throughput": THROUGHPUT[r]
            }

            with open(f"{LOG_DIR}/round_{r}.json", "w") as f:
                json.dump(log, f, indent=4)
                
            requests.post(
                "http://127.0.0.1:8000/log_event",
                json=log,
                timeout=10
            )

            
            time.sleep(1)

        server_proc.wait()  # blocks until server is manually stopped
        print("All rounds done.")
    
    except KeyboardInterrupt:
        print("\n[DRIVER] Interrupted by user.")
    except Exception as e:
        print(f"\n[DRIVER] Error: {e}")
    finally:
        print("\n[DRIVER] Stopping all processes...")

        # for p in client_procs:
        #     p.terminate()
        
        # if dashboard_proc:
        #     dashboard_proc.terminate()

        if server_proc:
            server_proc.terminate()
            
        print("[DRIVER] Saving charts...")
        save_output_charts(LOCAL_REWARDS, GLOBAL_REWARDS, SPARK_LATENCY, THROUGHPUT)
        print("[DRIVER] Done.")
    

if __name__ == "__main__":
    main()
