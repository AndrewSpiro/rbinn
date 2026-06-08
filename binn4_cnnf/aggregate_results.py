import json
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt


def load_jsons(paths):
    return [json.load(open(p, "r")) for p in paths]


def aggregate_folder(results_dir, model_name_base, train_seeds, attack_seeds):
    found_paths = []
    for t_seed in train_seeds:
        for a_seed in attack_seeds:
            path = os.path.join(
                results_dir,
                f"{model_name_base}_seed_{t_seed}",
                f"attack_seed_{a_seed}",
                "results.json",
            )
            if os.path.exists(path):
                found_paths.append(path)
            else:
                print(
                    f"Warning: File not found for t_seed {t_seed} and a_seed {a_seed} at {path}"
                )

    if not found_paths:
        print(f"Error: No valid result files found in {results_dir}.")
        return None

    print(f"Aggregating {len(found_paths)} files from {results_dir}...")
    json_list = load_jsons(found_paths)

    all_metrics = [list(r.keys()) for r in json_list]
    shared_metrics_unordered = set.intersection(*map(set, all_metrics))
    shared_metrics = [m for m in json_list[0].keys() if m in shared_metrics_unordered]

    stats = {}
    for metric in shared_metrics:
        vals = [r[metric] for r in json_list]
        stats[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return stats


def save_aggregated(stats, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{label}_summary.json")
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Aggregated results saved to {save_path}")


def plot_results(sources, output_path):
    shared_keys = None
    first_keys = None
    for src in sources:
        if src["data"] is None:
            continue
        if first_keys is None:
            first_keys = list(src["data"].keys())
        keys = set(src["data"].keys())
        shared_keys = keys if shared_keys is None else shared_keys & keys

    if not shared_keys:
        raise ValueError("No shared metric keys found across all sources.")

    shared_keys = [k for k in first_keys if k in shared_keys]

    n_groups = len(shared_keys)
    n_bars = sum(1 for src in sources if src["data"] is not None)
    x = np.arange(n_groups)
    total_width = 0.75
    width = total_width / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2.5), 7))
    colors = plt.cm.tab10.colors

    bar_i = 0
    for src in sources:
        if src["data"] is None:
            continue
        scale = src.get("scale", 1.0)
        means = [src["data"][k]["mean"] * scale for k in shared_keys]
        stds = [src["data"][k]["std"] * scale for k in shared_keys]
        offset = (bar_i - (n_bars - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=src["label"],
            color=colors[bar_i % len(colors)],
        )
        bar_i += 1

    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    acronyms = {"Pgd", "Spsa", "Ete"}

    def format_label(k):
        words = k.replace("_", " ").title().split()
        return " ".join(w.upper() if w in acronyms else w for w in words)

    ax.set_xticklabels([format_label(k) for k in shared_keys], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, "bar_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved figure to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of train seeds to aggregate over",
    )
    parser.add_argument(
        "--attack_seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of attack seeds to aggregate over",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder for aggregated results and plot",
    )

    # --- fixed sources ---
    parser.add_argument(
        "--orig_results_path",
        type=str,
        default="orig_results.json",
        help="Path to the original baseline results JSON",
    )
    parser.add_argument(
        "--supclean_results_path",
        type=str,
        default="supclean_results.json",
        help="Path to the repro supclean results JSON",
    )
    parser.add_argument(
        "--clean_results_dir",
        type=str,
        default="clean_results",
        help="Folder to aggregate clean results from",
    )
    parser.add_argument(
        "--clean_model_name_base",
        type=str,
        required=True,
        help="Full model folder prefix inside clean_results_dir, e.g. 'clean_CNNF'",
    )
    parser.add_argument(
        "--adv_fgsm_results_dir",
        type=str,
        default="adv_fgsm_results",
        help="Folder to aggregate adversarial FGSM results from",
    )
    parser.add_argument(
        "--adv_fgsm_model_name_base",
        type=str,
        required=True,
        help="Full model folder prefix inside adv_fgsm_results_dir, e.g. 'adv_fgsm_CNNF'",
    )

    args = parser.parse_args()

    # orig_results.json (already has mean/std per metric)
    orig_data = json.load(open(args.orig_results_path))

    # supclean_results.json (same format)
    supclean_data = json.load(open(args.supclean_results_path))


    clean_data = aggregate_folder(
        args.clean_results_dir,
        args.clean_model_name_base,
        args.train_seeds,
        args.attack_seeds,
    )
    if clean_data:
        save_aggregated(clean_data, args.out, "clean")

    adv_data = aggregate_folder(
        args.adv_fgsm_results_dir,
        args.adv_fgsm_model_name_base,
        args.train_seeds,
        args.attack_seeds,
    )
    if adv_data:
        save_aggregated(adv_data, args.out, "adv_fgsm")

    # scale to account for discrepancies in saving as % or decimal in the JSON
    sources = [
        {"label": "Original Baseline", "data": orig_data, "scale": 0.01},
        {"label": "Repro (SupClean) Results", "data": supclean_data, "scale": 0.01},
        {"label": "Clean Results", "data": clean_data, "scale": 1.0},
        {"label": "Adv FGSM Results", "data": adv_data, "scale": 1.0},
    ]

    plot_results(sources=sources, output_path=args.out)
