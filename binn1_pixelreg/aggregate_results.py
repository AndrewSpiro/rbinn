import os
import json
import statistics
from matplotlib import pyplot as plt
import sys
import numpy as np

def aggregate_hierarchical(root_dir="save"):
    hierarchy = {}
    # collect all data into the hierarchy
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
    # dictionary to store results per attack type: {attack_type: (means, stds)}
    attack_results = {}

    # Identify all unique attack types across all seeds
    all_attacks = set()
    for t_seed in hierarchy:
        all_attacks.update(hierarchy[t_seed].keys())

    for attack in all_attacks:
        attack_means_across_seeds = []
        
        for t_seed in hierarchy:
            if attack in hierarchy[t_seed]:
                # Transpose the trials for this specific seed/attack
                trials = hierarchy[t_seed][attack]
                runs_transposed = list(zip(*trials))
                seed_means = [statistics.mean(r) for r in runs_transposed]
                attack_means_across_seeds.append(seed_means)
        
        if attack_means_across_seeds:
            print(f"\n=== GLOBAL AVERAGE FOR ATTACK: {attack} ===")
            global_transposed = list(zip(*attack_means_across_seeds))
            g_means = [statistics.mean(g) for g in global_transposed]
            g_stds = [statistics.stdev(g) if len(g) > 1 else 0.0 for g in global_transposed]
            
            print(f"  Means: {[round(m, 4) for m in g_means]}")
            attack_results[attack] = (g_means, g_stds)

    return attack_results

def plot_results(baselines, means, stds, attack_name, eps_min, eps_max, save_dir, seed_string):
    plt.figure(figsize=(8, 5))
    x = np.linspace(eps_min, eps_max, len(means))

    means = np.array(means)
    stds = np.array(stds)

    plt.plot(x, means, label=f'Repro ({attack_name})', marker='o', markersize=4)
    plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    # Plot relevant baselines
    for b_name, b_data in baselines.items():
        # Check if attack type and range matche
        b_type = b_data['attack_config']['type']
        b_range_min = b_data['attack_config']['range'][0] 
        b_range_max = b_data['attack_config']['range'][1] 

        if b_type.lower() == attack_name.lower() and np.isclose(b_range_min, eps_min) and np.isclose(b_range_max, eps_max):
            plt.plot(x, b_data['means'], '--', label=b_data['label'])

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title(f'BINN1 Pixelreg Comparison: {attack_name.capitalize()} Attack')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(save_dir, f"binn1_{attack_name}_train_seeds_{seed_string}.png")
    plt.savefig(out_path)
    plt.close()
    print(f'Plot saved for {attack_name} to {out_path}')

if __name__ == "__main__":
    baselines = {
        'cifar100_reg_baseline': {
            'means' : [0.8465, 0.8384, 0.7973, 0.7288, 0.6438, 0.5562, 0.4685, 0.3945, 0.3342, 0.2822],
            'label' : "Baseline: Regularized - CIFAR10",
            'attack_config': {
                'type': 'Gaussian',
                'range': (0.0, 0.3) 
            }
        }
    }
    
    eps_min = float(sys.argv[1])
    eps_max = float(sys.argv[2])
    save_dir = sys.argv[3]
    seed_string = sys.argv[4]

    # Get results for all attacks found in the save folder
    attack_dict = aggregate_hierarchical(save_dir)

    # Create a separate plot for each attack found
    for attack_name, (means, stds) in attack_dict.items():
        plot_results(baselines, means, stds, attack_name, eps_min, eps_max, save_dir, seed_string)