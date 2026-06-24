import json
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt

def mean_std(values):
    if len(values) == 0:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def load_jsons(paths):
    return [json.load(open(p, "r")) for p in paths]


def aggregate(json_list):
    agg = {
        "clean_model_baseline": [],
        "experiments": {}
    }

    for data in json_list:
        if "clean_model_baseline" in data:
            agg["clean_model_baseline"].append(data["clean_model_baseline"])

        for eps_key, exp in data.get("experiments", {}).items():
            if eps_key not in agg["experiments"]:
                agg["experiments"][eps_key] = {
                    "clean_model": {"attacks": [], "clean_accs": []},
                    "robust_model": {"attacks": [], "clean_accs": []},
                    "robust_redetect_model": {"attacks": [], "clean_accs": []}
                }

            for model_type in ["clean_model", "robust_model", "robust_redetect_model"]:
                if model_type not in exp:
                    continue
                
                for key, val in exp[model_type].items():
                    if key == "clean_accuracy":
                        agg["experiments"][eps_key][model_type]["clean_accs"].append(val)
                    else:
                        agg["experiments"][eps_key][model_type]["attacks"].append(val)

    out = {"clean_model_baseline": {}, "experiments": {}}
    m, s = mean_std(agg["clean_model_baseline"])
    out["clean_model_baseline"] = {"mean": m, "std": s}

    for eps_key, exp in agg["experiments"].items():
        out["experiments"][eps_key] = {}

        for model_type, results in exp.items():
            vals_false = [r["redetect_edge_false"] for r in results["attacks"] if "redetect_edge_false" in r]
            vals_true = [r["redetect_edge_true"] for r in results["attacks"] if "redetect_edge_true" in r]

            m_f, s_f = mean_std(vals_false)
            m_t, s_t = mean_std(vals_true)
            
            m_c, s_c = mean_std(results["clean_accs"])

            out["experiments"][eps_key][model_type] = {
                "clean_accuracy": {"mean": m_c, "std": s_c} if m_c is not None else None,
                "redetect_edge_false": {"mean": m_f, "std": s_f},
                "redetect_edge_true": {"mean": m_t, "std": s_t} if vals_true else None
            }

    return out


def plot_results(agg_alpha05, agg_alpha0, baselines_path, output_path="deep_clustered_plot.png"):
    orig_results = json.load(open(baselines_path, "r"))

    experiments = agg_alpha05.get("experiments", {})
    if not experiments:
        return

    eps_key = list(experiments.keys())[0]
    results_05 = agg_alpha05["experiments"][eps_key]
    results_0  = agg_alpha0["experiments"][eps_key]
    model_types = list(results_05.keys())  # [clean_model, robust_model, robust_redetect_model]

    clean_base_m_05 = agg_alpha05.get("clean_model_baseline", {}).get("mean", 0)
    clean_base_s_05 = agg_alpha05.get("clean_model_baseline", {}).get("std", 0)

    def get_metric(results, model, metric, fallback_m=0, fallback_s=0):
        entry = results.get(model, {}).get(metric)
        if entry is None:
            return fallback_m, fallback_s
        return entry.get("mean") or fallback_m, entry.get("std") or fallback_s

    COL_ORIG = 'tab:blue'
    COL_05   = 'tab:orange'
    COL_0    = 'tab:green'

    metrics = [
        ("clean_accuracy",      "Clean Acc"),
        ("redetect_edge_true",  "RT"),
        ("redetect_edge_false", "RF"),
    ]

    width = 0.15
    gap   = width * 1.0

    group_spacing = 2
    x_positions = np.arange(len(model_types)) * group_spacing

    fig, ax = plt.subplots(figsize=(15, 5))

    plt.subplots_adjust(bottom=0.25)

    for idx, model in enumerate(model_types):
        xc = x_positions[idx]
        is_clean_model = (model == "clean_model")
        
        num_runs = 2 if is_clean_model else 3
        block_width = num_runs * width
        total_group_width = 3 * block_width + 2 * gap
        
        start_offset = -total_group_width / 2 + block_width / 2
        
        metric_centers = []

        for bi, (metric_key, metric_label) in enumerate(metrics):
            bc = xc + start_offset + bi * (block_width + gap)
            metric_centers.append(bc)

            if metric_key == "clean_accuracy":
                orig_m = orig_results[model]["clean_accuracy"]["mean"]
                orig_s = orig_results[model]["clean_accuracy"]["std"]
            else:
                orig_m = orig_results[model].get(metric_key, {}).get("mean", 0)
                orig_s = orig_results[model].get(metric_key, {}).get("std", 0)

            fallback_m = clean_base_m_05 if (is_clean_model and metric_key == "clean_accuracy") else 0
            fallback_s = clean_base_s_05 if (is_clean_model and metric_key == "clean_accuracy") else 0
            r05_m, r05_s = get_metric(results_05, model, metric_key, fallback_m, fallback_s)

            label_orig = 'Orig' if idx == 0 and bi == 0 else '_nolegend_'
            label_05   = r'Repro $\alpha=0.5$' if idx == 0 and bi == 0 else '_nolegend_'
            label_0    = r'Repro $\alpha=0$' if idx == 1 and bi == 0 else '_nolegend_'

            if is_clean_model:
                ax.bar(bc - width/2, orig_m, width, yerr=orig_s, color=COL_ORIG, capsize=3, label=label_orig)
                ax.bar(bc + width/2, r05_m,  width, yerr=r05_s, color=COL_05,   capsize=3, label=label_05)
            else:
                r0_m, r0_s = get_metric(results_0, model, metric_key)
                ax.bar(bc - width, orig_m, width, yerr=orig_s, color=COL_ORIG, capsize=3, label=label_orig)
                ax.bar(bc,         r05_m,  width, yerr=r05_s, color=COL_05,   capsize=3, label=label_05)
                ax.bar(bc + width, r0_m,   width, yerr=r0_s,  color=COL_0,    capsize=3, label=label_0)

            ax.text(bc, -0.04, metric_label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=10)

        model_display_name = model.replace('_', ' ').title()
        group_midpoint = np.mean(metric_centers)
        
        ax.text(group_midpoint, -0.12, model_display_name, ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=12, fontweight='bold')
        
        ax.plot([metric_centers[0] - block_width/2, metric_centers[-1] + block_width/2], 
                [-0.09, -0.09], color='black', transform=ax.get_xaxis_transform(), clip_on=False, lw=1)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks([])
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title='Run configuration', framealpha=0.9)

    for i in range(len(model_types) - 1):
        mid = (x_positions[i] + x_positions[i + 1]) / 2
        ax.axvline(mid, color='gray', linestyle='--', alpha=0.3)

    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",  type=str, required=True, help="Root directory containing Res<data_dir>_alpha_* subdirs")
    parser.add_argument("--data_dir",  type=str, required=True, help="Base dataset name (e.g., cifar10)")
    parser.add_argument("--seeds",     nargs="+", type=int, required=True, help="List of seeds to aggregate")
    parser.add_argument("--attack",    type=str, default="FGSM",     help="Attack folder name")
    parser.add_argument("--net_type",  type=str, default="rgbedge",  help="Network type for filename")
    parser.add_argument("--out",       type=str, required=True,      help="Output directory for aggregated JSON files")
    parser.add_argument("--baselines_path", type=str, default="orig_results.json", help="Path to original paper results JSON")
    args = parser.parse_args()

    def collect_paths(alpha_tag):
        found = []
        for seed in args.seeds:
            path = os.path.join(
                args.root_dir,
                f"Res{args.data_dir}_{alpha_tag}",
                f"seed_{seed}",
                args.attack,
                f"results_{args.net_type}.json"
            )
            if os.path.exists(path):
                found.append(path)
            else:
                print(f"Warning: File not found at {path}")
        return found

    paths_05 = collect_paths("alpha_0.5")
    paths_0  = collect_paths("alpha_0")

    if not paths_05:
        print("Error: No result files found for alpha=0.5.")
        exit(1)
    if not paths_0:
        print("Error: No result files found for alpha=0.")
        exit(1)

    print(f"Aggregating {len(paths_05)} file(s) for alpha=0.5 ...")
    agg_05 = aggregate(load_jsons(paths_05))

    print(f"Aggregating {len(paths_0)} file(s) for alpha=0 ...")
    agg_0  = aggregate(load_jsons(paths_0))

    os.makedirs(args.out, exist_ok=True)

    out_05 = os.path.join(args.out, "aggregated_alpha_0.5.json")
    out_0  = os.path.join(args.out, "aggregated_alpha_0.json")

    with open(out_05, "w") as f:
        json.dump(agg_05, f, indent=4)
    print(f"Saved alpha=0.5 results to {out_05}")

    with open(out_0, "w") as f:
        json.dump(agg_0, f, indent=4)
    print(f"Saved alpha=0   results to {out_0}")

    plot_path = os.path.join(args.out, "deep_clustered_plot.png")
    plot_results(agg_05, agg_0, args.baselines_path, plot_path)
    print(f"Plot saved to {plot_path}")