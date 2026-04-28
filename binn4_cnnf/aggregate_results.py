import json
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt

def load_jsons(paths):
    return [json.load(open(p, "r")) for p in paths]

def aggregate(json_list):
    all_metrics = [list(result.keys()) for result in json_list]
    shared_metrics = set.intersection(*map(set, all_metrics))

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

    return stats

def plot_results(aggregated_data, baselines_path, output_path='results_bar_plot.png'):

    orig_results = json.load(open(baselines_path, "r"))
    shared_keys = [k for k in aggregated_data.keys() if k in orig_results]
    if not shared_keys:
        raise Warning("shared keys is empty")

    x = np.arange(len(shared_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    agg_means = [aggregated_data[k]['mean'] for k in shared_keys]
    agg_stds = [aggregated_data[k]['std'] for k in shared_keys]

    orig_means = [orig_results[k]['mean'] for k in shared_keys]
    orig_stds = [orig_results[k]['std'] for k in shared_keys]

    rects1 = ax.bar(x-width/2, agg_means, width, yerr=agg_stds,
                    label = 'Aggregated Data')

    rects2 = ax.bar(x+width/2, orig_means, width, yerr=orig_stds,
                    label='Baseline Results')

    ax.set_ylabel('Scores')
    
    clean_labels = [k.replace('_',' ').title() for k in shared_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels)

    ax.legend()

    fig.tight_layout()
    plot_path = f"{output_path}/bar_plot.png"
    plt.savefig(plot_path)
    print(f"Saved figure to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Base results directory (e.g., Rescifar10)")
    parser.add_argument("--model_name_base", type=str, help="base name of model e.g., 'CNNF'")
    parser.add_argument("--bool_debug", type=bool, default=False, help="Flag for debugging")
    parser.add_argument("--train_seeds", nargs="+", type=int, help="List of train seeds to aggregate over")
    parser.add_argument("--attack_seeds", nargs="+", type=int, help="List of attacks seeds to aggregate over. Must be the same set for all train seeds.")
    parser.add_argument("--out", type=str, help="output folder for aggregated results")
    parser.add_argument("--baselines_path", type=str, default="orig_results.json", help="path for the results from the paper")
    args = parser.parse_args()

    found_paths = []
    for t_seed in args.train_seeds:
        for a_seed in args.attack_seeds:
            path = os.path.join(
                args.results_dir,
                f"{args.model_name_base}_seed_{t_seed}",
                f"attack_seed_{a_seed}",
                "results.json"
            )

            if os.path.exists(path):
                found_paths.append(path)
            else:
                print(f"Warning: File not found for t_seed {t_seed} and a_seed {a_seed} at {path}")

    if not found_paths:
        print("Error: No valid result files found for the provided seeds.")
    else:
        print(f"Aggregating {len(found_paths)} files...")
        json_list = load_jsons(found_paths)
        aggregated_data = aggregate(json_list)
        plot_results(aggregated_data=aggregated_data, baselines_path=args.baselines_path, output_path=args.out)

    
