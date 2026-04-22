import json
import numpy as np
import argparse
import os


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
            # Process Attack Results
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Dir relative to binn3_eat for results root")
    parser.add_argument("--data_dir", type=str, required=True, help="Base results directory (e.g., Rescifar10)")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of seeds to aggregate (e.g., 0 1 2)")
    parser.add_argument("--attack", type=str, default="FGSM", help="Attack folder name")
    parser.add_argument("--net_type", type=str, default="rgbedge", help="Network type for filename")
    parser.add_argument("--out", type=str, required=True, help="Output json path")
    args = parser.parse_args()

    found_paths = []
    for seed in args.seeds:
        path = os.path.join(
            args.root_dir,
            f"Res{args.data_dir}", 
            f"seed_{seed}", 
            args.attack, 
            f"results_{args.net_type}.json"
        )
        
        if os.path.exists(path):
            found_paths.append(path)
        else:
            print(f"Warning: File not found for seed {seed} at {path}")

    if not found_paths:
        print("Error: No valid result files found for the provided seeds.")
    else:
        print(f"Aggregating {len(found_paths)} files...")
        json_list = load_jsons(found_paths)
        result = aggregate(json_list)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved aggregated results to {args.out}")