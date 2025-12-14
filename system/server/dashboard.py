import gradio as gr
import websockets
import asyncio
import json
import pandas as pd
from datetime import datetime

WS_URL = "ws://172.31.26.200:8000/ws/dashboard"

async def connect_and_stream():
    history_df = pd.DataFrame(columns=["round", "global_reward", "runtime", "spark_latency", "throughput"])
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print(f"Đã kết nối tới {WS_URL}")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if(data["type"] != "log_update"): continue
                
                data = data["log"]
                
                # print("[WS] Received:", data)
                
                r = data.get("round", 0)
                local_rewards = data.get("local_rewards", {})
                global_reward = data.get("global_reward", 0)
                runtime = data.get("runtime", 0)
                spark_latency = data.get("spark_latency", 0)
                throughput = data.get("throughput", 0)

                new_row = pd.DataFrame([{
                    "round": r,
                    "global_reward": global_reward,
                    "runtime": runtime,
                    "spark_latency": spark_latency,
                    "throughput": throughput
                }])
                history_df = pd.concat([history_df, new_row], ignore_index=True)

                # print("Sorted:", sorted(list(local_rewards.items()), key=lambda x: x[0]))

                local_rewards_df = pd.DataFrame(sorted(list(local_rewards.items()), key=lambda x: x[0]), columns=["Client", "Reward"])

                yield (
                    r,                  
                    global_reward,       
                    runtime,             
                    throughput,          
                    spark_latency,       
                    history_df,       
                    local_rewards_df, 
                    history_df        
                )
                
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

with gr.Blocks(title="Realtime System Dashboard") as demo:
    gr.Markdown("## Training Dashboard")
    
    with gr.Row():
        d_round = gr.Number(label="Current Round", value=0, precision=0)
        d_global = gr.Number(label="Global Reward", value=0.0)
        d_runtime = gr.Number(label="Runtime (s)", value=0.0)
        d_throughput = gr.Number(label="Throughput (ops/s)", value=0.0)
        d_latency = gr.Number(label="Spark Latency (ms)", value=0.0)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Global Reward")
            plot_global = gr.LinePlot(
                x="round", 
                y="global_reward", 
                title="Global Reward over Rounds",
                width=600, 
                height=300,
                tooltip=["round", "global_reward"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Local Rewards (Current Round)")
            plot_local = gr.BarPlot(
                x="Client", 
                y="Reward", 
                title="Local Rewards Distribution",
                width=300, 
                height=300,
                vertical=False,
                tooltip=["Client", "Reward"]
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### System Metrics History")
            plot_metrics = gr.LinePlot(
                x="round",
                y="spark_latency",
                title="Spark Latency over Rounds",
                width=1000,
                height=250,
                tooltip=["round", "spark_latency", "throughput"]
            )

    demo.load(
        fn=connect_and_stream, 
        inputs=None, 
        outputs=[
            d_round, d_global, d_runtime, d_throughput, d_latency, 
            plot_global, plot_local, plot_metrics                 
        ],
        stream_every=0.5 
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
