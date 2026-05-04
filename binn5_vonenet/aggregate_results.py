import argparse
import json
import os
import numpy as np

def load_jsons(paths):
    return [json.load(open(p, "r")) for p in paths]

def aggregate(json_list):
    all_metrics = [list(result.keys()) for result in json_list]
    unordered_shared_metrics = set.intersection(*map(set, all_metrics))
    shared_metrics = [m for m in json_list[0].keys() if m in unordered_shared_metrics]

    stats = {}
    for metric in shared_metrics:
        stats[metric] = {
            "mean": None,
            "std": None
        }
        metric_vals = []
        for result in json_list:
            metric_vals.append(result[metric])
        stats[metric]["mean"] = np.mean(metric_vals)
        stats[metric]["std"] = np.std(metric_vals)

    print(stats)
    json_save_path = f"{args.out}/num_summary.json"
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(stats, f)
    print(f"Aggregated results saved to {json_save_path}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root where models are saved")    
    parser.add_argument("--train_seeds", nargs="+", type=int, help="List of train seeds to aggregate over")
    parser.add_argument("--model_arch", type=str, help="model architecture (e.g., resnet50)")
    parser.add_argument("--out", type=str, help="save path for aggregated results")
    args = parser.parse_args()

    found_paths = []
    for t_seed in args.train_seeds:
        path = os.path.join(
            args.root,
            f"{args.model_arch}_vonenet_seed_{t_seed}",
            "results.json"
        )

        if os.path.exists(path):
            found_paths.append(path)
        else:
            print(f"Warning: File not found for t_seed {t_seed} at {path}")

    if not found_paths:
        print("Error: No valid result files found for the provided seeds.")
    else:
        print(f"Aggregating {len(found_paths)} files...")
        json_list = load_jsons(found_paths)
        aggregated_data = aggregate(json_list)
