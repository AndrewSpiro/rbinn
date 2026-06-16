import torch
import os
import pickle as pkl
import numpy as np
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt

from LocalLearning.Experiments import PerturbationExperiment, RandomPerturbationExperiment, FGSMExperiment, PGDExperiment


def make_stats_dict(exp_dict):
    grouped_dict = defaultdict(list)
    for path in exp_dict['pkl_paths']:
        exp = exp_dict['exp']
        exp.load(path)
        for key, result in exp:
            for metric, values in result.items():
                grouped_dict[metric].append(values)
    stats = {}
    stats['eps'] = grouped_dict['eps'][0]
    acc_data = np.array(grouped_dict['acc'])
    stats['acc'] = {
        "mean": acc_data.mean(axis=0).tolist(),
        "std":  acc_data.std(axis=0).tolist(),
    }
    stats['crit_eps'] = grouped_dict['crit_eps']
    stats['crit_norm'] = grouped_dict['crit_norm']
    return stats


def get_orig_results(result_path):
    pbc_baseline_dict = dict()
    pbc_baselines = [('rp', 'orig_rp_pbc.csv'), ('fgsm', 'orig_fgsm_pbc.csv'), ('pgd', 'orig_pgd_pbc.csv')]
    for (name, fn) in pbc_baselines:
        x = []
        y = []
        with open(result_path / fn, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x.append(row[0])
                y.append(row[1])
        pbc_baseline_dict[name] = {'x': x, 'y': y}

    rd_baseline_dict = dict()
    rd_baselines = [('rp', 'orig_rp_rd.csv'), ('fgsm', 'orig_fgsm_rd.csv'), ('pgd', 'orig_pgd_rd.csv')]
    for (name, fn) in rd_baselines:
        x = []
        with open(result_path / fn, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x.append(row[1])
        rd_baseline_dict[name] = {'x': x}

    return pbc_baseline_dict, rd_baseline_dict


def collect_pkl_paths(root, acc_type=None):
    """Walk directory and return lists of pkl paths for each attack type."""
    rp_paths, fgsm_paths, pgd_paths = [], [], []
    for subdir, dirs, files in os.walk(root):
        if acc_type and f"experiments/{acc_type}" not in subdir:
            continue
        if "random_perturbation_results.pkl" in files:
            rp_paths.append(Path(subdir) / "random_perturbation_results.pkl")
        if "fgsm_results.pkl" in files:
            fgsm_paths.append(Path(subdir) / "fgsm_results.pkl")
        if "pgd_results.pkl" in files:
            pgd_paths.append(Path(subdir) / "pgd_results.pkl")
    return rp_paths, fgsm_paths, pgd_paths


def build_experiments(rp_paths, fgsm_paths, pgd_paths, ce_loss):
    return {
        'rp':   {'pkl_paths': rp_paths,   'exp': RandomPerturbationExperiment(ce_loss)},
        'fgsm': {'pkl_paths': fgsm_paths, 'exp': FGSMExperiment(ce_loss)},
        'pgd':  {'pkl_paths': pgd_paths,  'exp': PGDExperiment(ce_loss)},
    }


def extract_crit_norms(stats_dict):
    all_crit_norm = []
    for crit_eps, crit_norm in zip(stats_dict['crit_eps'], stats_dict['crit_norm']):
        crit_eps = np.array(crit_eps)
        crit_norm = np.array(crit_norm)
        nan_mask = np.isnan(crit_eps)
        all_crit_norm.append(crit_norm[~nan_mask])
    return np.concatenate(all_crit_norm)

def create_pbc_plot(configs_stats, attack_name, acc_type, pbc_baseline_dict, result_path, color_map):
    """
    Accuracy vs epsilon (perturbation budget curve) plot for one attack type and norm type,
    with one line per train config plus the original baseline.
    """
    fig, ax = plt.subplots()

    for cfg, stats in configs_stats.items():
        eps = np.array(stats["eps"], dtype=float)
        mean = stats["acc"]["mean"]
        std  = stats["acc"]["std"]
        color = color_map[cfg]
        ax.semilogx(eps, mean, label=cfg, color=color)
        ax.fill_between(
            eps,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.3, color=color
        )

    ax.semilogx(
        np.array(pbc_baseline_dict[attack_name]['x'], dtype=float),
        np.array(pbc_baseline_dict[attack_name]['y'], dtype=float),
        '--', label="Original", color='C0'
    )

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower left')
    fig.savefig(result_path / f'{attack_name}_{acc_type}_pbc.png')
    plt.close(fig)


def create_rd_plot(configs_stats, attack_name, acc_type, rd_baseline_dict, result_path, color_map):
    """
    Robustness distribution (critical norm) box plot for one attack type and norm type,
    with one box per train config plus the original baseline.
    """
    baseline_vals = np.array(rd_baseline_dict[attack_name]['x'], dtype=float)
    whislo, q1, median, q3, whishi = (
        baseline_vals.min(), np.percentile(baseline_vals, 25),
        np.median(baseline_vals),
        np.percentile(baseline_vals, 75), baseline_vals.max()
    )
    baseline_stats = {
        'whislo': whislo, 'q1': q1, 'med': median,
        'q3': q3, 'whishi': whishi, 'fliers': []
    }

    n_configs = len(configs_stats)
    positions = list(range(2, 2 + n_configs))
    labels = ['Original'] + list(configs_stats.keys())

    fig, ax = plt.subplots()
    ax.bxp([baseline_stats], positions=[1], showfliers=False)

    for pos, (cfg, stats) in zip(positions, configs_stats.items()):
        crit_norms = extract_crit_norms(stats)
        ax.boxplot(crit_norms, positions=[pos], showfliers=False)

    ax.set_xticks([1] + positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, 1.5 + n_configs)
    ax.set_ylabel("Critical norm")
    fig.savefig(result_path / f'{attack_name}_{acc_type}_rd.png')
    plt.close(fig)


def create_plots(configs_stats, attack_name, acc_type, result_path, color_map):
    pbc_baseline_dict, rd_baseline_dict = get_orig_results(result_path)
    create_pbc_plot(configs_stats, attack_name, acc_type, pbc_baseline_dict, result_path, color_map)
    create_rd_plot(configs_stats, attack_name, acc_type, rd_baseline_dict, result_path, color_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str,
        help="Root path containing train-config subdirectories each with 'experiments/absolute' and 'experiments/relative' sub-subdirectories."
    )
    parser.add_argument(
        '--train_configs', nargs='+',
        help="List of train config folder names, e.g. clean fgsm_4 fgsm_8 fgsm_16"
    )
    args = parser.parse_args()
    print(f"Processing train configs: {args.train_configs}")

    result_path  = Path(args.result_path)
    ce_loss      = torch.nn.CrossEntropyLoss()
    color_map = {cfg: f'C{i+1}' for i, cfg in enumerate(args.train_configs)}
    acc_types   = ('absolute', 'relative')
    attack_names = ('rp', 'fgsm', 'pgd')

    all_stats = {acc_type: {} for acc_type in acc_types}

    for cfg in args.train_configs:
        for acc_type in acc_types:
            rp_paths, fgsm_paths, pgd_paths = collect_pkl_paths(result_path / cfg, acc_type)
            experiments = build_experiments(rp_paths, fgsm_paths, pgd_paths, ce_loss)

            for attack_name in attack_names:
                stats = make_stats_dict(experiments[attack_name])
                all_stats[acc_type].setdefault(attack_name, {})[cfg] = stats

    # One plot per (attack_name, acc_type) pair, all train configs on the same axes.
    for acc_type in acc_types:
        for attack_name in attack_names:
            if attack_name not in all_stats[acc_type]:
                continue
            print(f"Plotting {attack_name} / {acc_type}")
            configs_stats = all_stats[acc_type][attack_name]
            create_plots(configs_stats, attack_name, acc_type, result_path, color_map)