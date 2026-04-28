import json
import os
import argparse

def load_jsons(paths):
    return [json.load(open(p, "r")) for p in paths]

def aggregate(json_list):
    print(json_list)

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
    if args.bool_debug:
        debug_string = "_debug"
    else:
        debug_string = ""
    for t_seed in args.train_seeds:
        for a_seed in args.attack_seeds:
            path = os.path.join(
                args.results_dir,
                f"{args.model_name_base}{debug_string}_seed_{t_seed}",
                f"attack_seed_{a_seed}.json"
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
        aggregate(json_list)
