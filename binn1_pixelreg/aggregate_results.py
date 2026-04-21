import os
import json
import statistics
# from matplotlib import pyplot as plt

def aggregate_hierarchical(root_dir="save"):
    # structure: { train_seed: { attack_type: [ [run1..10], [run1..10] ] } }
    hierarchy = {}

    # dollect all data into the hierarchy
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if not (os.path.isdir(model_path) and model_dir.startswith("model_seed_")):
            continue
        
        t_seed = model_dir.split("_")[-1]
        hierarchy[t_seed] = {}

        for trial_dir in os.listdir(model_path):
            trial_path = os.path.join(model_path, trial_dir)
            if os.path.isdir(trial_path) and trial_dir.startswith("attack_"):
                attack_type = trial_dir.split("_")[1]
                json_path = os.path.join(trial_path, "results.json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        res = json.load(f)
                        # data is like [[run1, run2...]]
                        runs = res[attack_type][0] if isinstance(res[attack_type][0], list) else res[attack_type]
                        
                        if attack_type not in hierarchy[t_seed]:
                            hierarchy[t_seed][attack_type] = []
                        hierarchy[t_seed][attack_type].append(runs)

    # Process and Print Stats
    all_train_seed_means = [] # store averages for the final global mean

    for t_seed, attacks in hierarchy.items():
        print(f"\n=== RESULTS FOR TRAIN SEED: {t_seed} ===")
        
        for attack, trials in attacks.items():
            # trials is a list of lists: [[run1..10], [run1..10]]
            # transpose to get stats for each epsilon
            runs_transposed = list(zip(*trials)) 
            
            means = [statistics.mean(r) for r in runs_transposed]
            stds = [statistics.stdev(r) if len(r) > 1 else 0.0 for r in runs_transposed]
            
            print(f"Attack: {attack}")
            print(f"  Means (10 runs): {[round(m, 4) for m in means]}")
            print(f"  Stds  (10 runs): {[round(s, 4) for s in stds]}")
            
            all_train_seed_means.append(means)

    # global average (average of the trainings)
    if all_train_seed_means:
        print("\n=== GLOBAL AVERAGE (OVER ALL SEEDS) ===")
        global_transposed = list(zip(*all_train_seed_means))
        global_means = [statistics.mean(g) for g in global_transposed]
        global_stds = [statistics.stdev(g) if len(g) > 1 else 0.0 for g in global_transposed]
        
        print(f"  Global Means: {[round(m, 4) for m in global_means]}")
        print(f"  Global Stds:  {[round(s, 4) for s in global_stds]}")

        return global_means, global_stds

if __name__ == "__main__":
    aggregate_hierarchical()