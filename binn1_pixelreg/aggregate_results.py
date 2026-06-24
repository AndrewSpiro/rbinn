import os
import json
import statistics
from matplotlib import pyplot as plt
import sys
import numpy as np

def aggregate_hierarchical(root_dir, acc_type):
    hierarchy = {}
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if not (os.path.isdir(model_path) and model_dir.startswith("model_seed_")):
            continue

        t_seed = model_dir.split("_")[-1]
        hierarchy[t_seed] = {}

        for trial_dir in os.listdir(os.path.join(model_path, acc_type)):  # <-- added acc_type
            trial_path = os.path.join(model_path, acc_type, trial_dir)
            if os.path.isdir(trial_path) and trial_dir.startswith("attack_"):
                attack_type = trial_dir.split("_")[1]
                json_path = os.path.join(trial_path, "results.json")

                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        res = json.load(f)
                        runs = res[attack_type][0] if isinstance(res[attack_type][0], list) else res[attack_type]

                        if attack_type not in hierarchy[t_seed]:
                            hierarchy[t_seed][attack_type] = []
                        hierarchy[t_seed][attack_type].append(runs)

    attack_results = {}
    all_attacks = set()
    for t_seed in hierarchy:
        all_attacks.update(hierarchy[t_seed].keys())

    for attack in all_attacks:
        attack_means_across_seeds = []

        for t_seed in hierarchy:
            if attack in hierarchy[t_seed]:
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


def aggregate_both(save_dir, acc_type):
    results = {}
    for split in ("clean", "fgsm"):
        subdir = os.path.join(save_dir, split)
        if os.path.isdir(subdir):
            print(f"\nAggregating {split} models")
            results[split] = aggregate_hierarchical(subdir, acc_type)  # <-- passed through
        else:
            print(f"Warning: {subdir} not found, skipping.")
            results[split] = {}
    return results["clean"], results["fgsm"]


def plot_results(baselines, clean_results, adv_results, attack_name, acc_type,
                 eps_min, eps_max, save_dir, seed_string):
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    plt.figure(figsize=(8, 5))

    # clean
    if attack_name in clean_results:
        means, stds = clean_results[attack_name]
        x = np.linspace(eps_min, eps_max, len(means))
        means, stds = np.array(means), np.array(stds)
        plt.plot(x, means, label="Repro (clean)", marker='o', markersize=4, color = 'C1')
        plt.fill_between(x, means - stds, means + stds, alpha=0.2, color='C1')

    # fgsm
    if attack_name in adv_results:
        means, stds = adv_results[attack_name]
        x = np.linspace(eps_min, eps_max, len(means))
        means, stds = np.array(means), np.array(stds)
        plt.plot(x, means, label="Repro (fgsm)", marker='s', markersize=4, color='C2')
        plt.fill_between(x, means - stds, means + stds, alpha=0.2, color='C2')

    # baselines
    if acc_type=='absolute':
        for b_name, b_data in baselines.items():
            b_type = b_data['attack_config']['type']
            b_range_min = b_data['attack_config']['range'][0]
            b_range_max = b_data['attack_config']['range'][1]

            if (b_type.lower() == attack_name.lower()
                    and np.isclose(b_range_min, eps_min)
                    and np.isclose(b_range_max, eps_max)):
                bx = np.linspace(eps_min, eps_max, len(b_data['means']))
                plt.plot(bx, b_data['means'], '--', label=b_data['label'], color='C0')

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"binn1_{acc_type}_{attack_name}_train_seeds_{seed_string}.png")
    plt.savefig(out_path)
    plt.close()
    print(f'Plot saved for {attack_name} to {out_path}')

if __name__ == "__main__":
    baselines = {
        'cifar100_reg_baseline': {
            'means': [0.8465, 0.8384, 0.7973, 0.7288, 0.6438, 0.5562, 0.4685, 0.3945, 0.3342, 0.2822],
            # 'means': [x + (1-0.8465) for x in [0.8465, 0.8384, 0.7973, 0.7288, 0.6438, 0.5562, 0.4685, 0.3945, 0.3342, 0.2822]], # for obtaining relative pbc
            'label': "Baseline: Regularized - CIFAR10",
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
    
    for acc_type in ['absolute', 'relative']:
        clean_dict, adv_dict = aggregate_both(save_dir, acc_type)
        all_attacks = set(clean_dict.keys()) | set(adv_dict.keys())
        for attack_name in all_attacks:
            plot_results(baselines=baselines, clean_results=clean_dict, adv_results=adv_dict,
                         attack_name=attack_name, acc_type=acc_type,
                         eps_min=eps_min, eps_max=eps_max, save_dir=save_dir, seed_string=seed_string)
    
    all_attacks = set(clean_dict.keys()) | set(adv_dict.keys())
    print(set(clean_dict.keys()))
    print(set(adv_dict.keys()))
    print(all_attacks)
    for attack_name in all_attacks:
        plot_results(baselines=baselines, clean_results=clean_dict, adv_results=adv_dict, attack_name=attack_name, acc_type=acc_type,
                     eps_min=eps_min, eps_max=eps_max, save_dir=save_dir, seed_string=seed_string)