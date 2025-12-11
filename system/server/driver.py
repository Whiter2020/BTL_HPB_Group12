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

spark = SparkSession.builder.appName("ReadGlobalQ").getOrCreate()


# Config
HDFS_BASE = "policies"        # nơi agents upload policies
GLOBAL_OUT = "global_policies"
AGG_SCRIPT = "aggregator_qvalues.py"  # đường dẫn trên FS của cluster
SPARK_SUBMIT = r"C:\spark\bin\spark-submit.cmd"
HDFS_EXE = r"C:\hadoop\bin\hdfs.cmd"

ROUNDS = 10
MIN_AGENTS = 4        # require at least this many agent files per round before aggregating
POLL_INTERVAL = 5     # seconds between checks (adjust)
ROUND_TIMEOUT = 300   # seconds to wait max for agents in a round
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
    
    cmd = [SPARK_SUBMIT, AGG_SCRIPT, policies_dir, out_dir, str(round_number)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(proc.stdout.decode())
    if proc.returncode != 0:
        print("Aggregator failed:", proc.stderr.decode())
    return proc.returncode == 0

def main():
    """uvicorn.run(
    app,
    host="127.0.0.1",
    port=8000,
    log_level="info"
    )"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    server_proc = subprocess.Popen(
    [sys.executable,"-m","uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=current_dir, shell=True)
    print("Server started in background.", server_proc.pid)

    if not wait_for_server("127.0.0.1", 8000, timeout=60):
        raise RuntimeError("Server did not start in time!")
    time.sleep(2)
    for r in range(1, ROUNDS+1):
        policies_dir = f"{HDFS_BASE}/round_{r}"
        print(f"\n=== Starting round {r} at {datetime.now().isoformat()} ===")
        # Wait until at least MIN_AGENTS files exist (or timeout)
        start = time.time()
        while True:
            print(policies_dir)
            files = hdfs_ls(policies_dir)
            if len(files) >= MIN_AGENTS:
                print(f"Found {len(files)} agent files.")
                break
            if time.time() - start > ROUND_TIMEOUT:
                print(f"Timeout waiting for agents for round {r}. Found {len(files)}. Proceeding anyway.")
                break
            print(f"Waiting for agents... found {len(files)} (need {MIN_AGENTS}). Checking again in {POLL_INTERVAL}s.")
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
        print(data_dict)

        requests.post(
            "http://127.0.0.1:8000/broadcast",
            json={"policy": data_dict},
            timeout=30
        )
        time.sleep(1)

    server_proc.wait()  # blocks until server is manually stopped
    print("All rounds done.")

if __name__ == "__main__":
    main()
