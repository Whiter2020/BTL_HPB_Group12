# plot_rewards.py
import json
import os
import argparse
import matplotlib.pyplot as plt

def load_agent_rewards(agent_rewards_dir):
    # expects files like agent_{id}_rewards.json with structure:
    # { "agent_id": "...", "rounds": [ {"round":1, "avg_reward": 0.4}, ... ] }
    agg = {}
    for fname in os.listdir(agent_rewards_dir):
        if not fname.endswith(".json"): continue
        path = os.path.join(agent_rewards_dir, fname)
        try:
            obj = json.load(open(path))
            aid = obj.get("agent_id", fname)
            rounds = obj.get("rounds", [])
            agg[aid] = {int(r['round']): r.get('avg_reward', None) for r in rounds}
        except Exception as e:
            continue
    return agg

def load_global_rewards(global_file):
    # expects { "round": r, "avg_reward": x } per line OR a JSON containing list
    try:
        data = json.load(open(global_file))
        if isinstance(data, dict) and "global_rewards" in data:
            return {int(r['round']): r.get('avg_reward', None) for r in data['global_rewards']}
        elif isinstance(data, list):
            return {int(item['round']): item.get('avg_reward', None) for item in data}
        elif isinstance(data, dict) and "round" in data:
            return {int(data["round"]): data.get("avg_reward", None)}
        else:
            # fallback empty
            return {}
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_dir", required=True, help="local dir with agent_{id}_rewards.json")
    parser.add_argument("--global_file", required=True, help="global rewards JSON")
    args = parser.parse_args()

    agents = load_agent_rewards(args.agent_dir)
    global_rewards = load_global_rewards(args.global_file)

    rounds = sorted(set().union(*[set(v.keys()) for v in agents.values()] + [set(global_rewards.keys())]))

    # Plot global
    global_y = [global_rewards.get(r, None) for r in rounds]
    plt.figure()
    plt.plot(rounds, global_y, marker='o', label='global avg reward')
    # plot each agent (light lines)
    for aid, data in agents.items():
        y = [data.get(r, None) for r in rounds]
        plt.plot(rounds, y, linestyle='--', marker='.', label=f'local {aid}')
    plt.xlabel("round")
    plt.ylabel("avg_reward")
    plt.title("Global vs Local average reward per round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
