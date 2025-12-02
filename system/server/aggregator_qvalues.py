# aggregator_qvalues.py
import json
import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

def parse_policy(content):
    text = content
    try:
        obj = json.loads(text)
        return obj
    except:
        return None


def map_qvalues(obj):
    """
    obj:
      {
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
    """
    if obj is None:
        return []

    weight = obj["hyperparameters"]["dataset_size"]
    actions = obj["actions"]
    Q = obj["Q_values"]

    # → emit: (action, (Q_i * weight, weight))
    out = []
    for a, q in zip(actions, Q):
        out.append((a, (q * weight, weight)))
    return out


def reduce_pair(a, b):
    # (sum_Qw, sum_w)
    return (a[0] + b[0], a[1] + b[1])


def main():
    if len(sys.argv) < 4:
        print("Usage: spark-submit aggregator_qvalues.py <hdfs_input_dir> <output_dir> <round>")
        sys.exit(1)

    IN = sys.argv[1]
    OUT = sys.argv[2]
    ROUND = int(sys.argv[3])

    conf = SparkConf().setAppName(f"FedAvg_Qvalues_Round_{ROUND}")
    sc = SparkContext(conf=conf)

    rdd = sc.wholeTextFiles(IN)
    

    contents = rdd.map(lambda x: x[1])  # x[0] = filename, x[1] = file content

    parsed = contents.map(parse_policy)\
                    .filter(lambda x: x is not None and int(x["round"]) == ROUND)
    print(parsed.collect())
    alphas = parsed.map(lambda p: p["hyperparameters"]["alpha"]).collect()
    avg_alpha = sum(alphas) / len(alphas) if len(alphas) !=0 else 0

    # flatMap arms
    mapped = parsed.flatMap(map_qvalues)

    #alphas = parsed.map(lambda p: p["alpha"]).collect()
    #print(alphas)
    # reduce by action
    reduced = mapped.reduceByKey(reduce_pair)

    arm_sums = reduced.collect()

    # compute global Q per arm
    global_Q = {}
    actions = []

    for action, (sum_Qw, sum_w) in arm_sums:
        global_Q[action] = sum_Qw / sum_w
        actions.append(action)

    print(global_Q)
    print("global_Q")

    actions = sorted(actions)
    sorted_actions = sorted(global_Q.keys())   # [1, 2, 3]
    Q_list = [global_Q[a] for a in sorted_actions]
    
    out_obj = {
        "actions": actions,
        "round": ROUND,
        "global_Q": Q_list,
        "hyperparameters": {
            "alpha": avg_alpha,
        }
    }
  
    out_json = json.dumps(out_obj, indent=2)
    print(out_json)
    print("DONE")

    print(f"{OUT}\\global_Q_round_{ROUND}.json")
    sc.parallelize([out_json], 1).saveAsTextFile(f"{OUT}\\global_policy_round_{ROUND}.json")

    print("Done. Global Q-values saved.")

    sc.stop()


if __name__ == "__main__":
    main()
