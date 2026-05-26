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
        metric_vals = [result[metric] for result in json_list]
        stats[metric] = {
            "mean": np.mean(metric_vals),
            "std": np.std(metric_vals),
        }
    return stats


def collect_stats(root, train_method, model_arch, train_seeds):
    train_method_root = os.path.join(root, train_method)
    found_paths = []
    for t_seed in train_seeds:
        path = os.path.join(
            train_method_root,
            f"{model_arch}_vonenet_seed_{t_seed}",
            "results.json",
        )
        if os.path.exists(path):
            found_paths.append(path)
        else:
            print(
                f"Warning: File not found for {train_method} t_seed {t_seed} at {path}"
            )

    if not found_paths:
        print(f"Error: No valid result files found for train_method '{train_method}'.")
        return None

    print(f"Aggregating {len(found_paths)} files for train_method '{train_method}'...")
    return aggregate(load_jsons(found_paths))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing 'clean/' and 'fgsm/' subdirectories",
    )
    parser.add_argument(
        "--train_seeds",
        nargs="+",
        type=int,
        help="List of train seeds to aggregate over",
    )
    parser.add_argument(
        "--model_arch", type=str, help="Model architecture (e.g., resnet50)"
    )
    parser.add_argument("--out", type=str, help="Save path for aggregated results JSON")
    args = parser.parse_args()

    combined = {}
    for train_method in ("clean", "fgsm"):
        stats = collect_stats(
            args.root, train_method, args.model_arch, args.train_seeds
        )
        if stats is not None:
            combined[train_method] = stats

    print(combined)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(combined, f)
    print(f"Aggregated results saved to {args.out}")
