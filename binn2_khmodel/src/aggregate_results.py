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
                x.append(row[1])  # row with id 1 because of csv saving from webplot digi
        rd_baseline_dict[name] = {'x': x}

    return pbc_baseline_dict, rd_baseline_dict


def collect_pkl_paths(root):
    """Walk directory and return lists of pkl paths for each attack type"""
    rp_paths, fgsm_paths, pgd_paths = [], [], []
    for subdir, dirs, files in os.walk(root):
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


def make_boxplots(repro_clean, repro_adv, baseline_rd, name, result_path):
    """Box plot with three groups: original baseline, clean repro, fgsm trained model."""
    whislo, q1, median, q3, whishi = baseline_rd
    baseline_stats = {
        'whislo': whislo, 'q1': q1, 'med': median,
        'q3': q3, 'whishi': whishi, 'fliers': []
    }

    fig, ax = plt.subplots()
    ax.bxp([baseline_stats], positions=[1], showfliers=False)
    ax.boxplot(repro_clean, positions=[2], showfliers=False)
    ax.boxplot(repro_adv,   positions=[3], showfliers=False)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['original', 'clean', 'fgsm'])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylabel("critical norm")

    fig.savefig(result_path / f'{name}_rd.png')
    plt.close(fig)


def create_plots(clean_stats, adv_stats, name, result_path):
    pbc_baseline_dict, rd_baseline_dict = get_orig_results(result_path)

    # pbc plot
    eps_clean = np.array(clean_stats["eps"], dtype=float)
    eps_adv   = np.array(adv_stats["eps"],   dtype=float)

    fig, ax = plt.subplots()
    ax.semilogx(
        np.array(pbc_baseline_dict[name]['x'], dtype=float),
        np.array(pbc_baseline_dict[name]['y'], dtype=float),
        label="original"
    )
    ax.semilogx(eps_clean, clean_stats["acc"]["mean"], label="clean")
    ax.fill_between(
        eps_clean,
        [m - s for m, s in zip(clean_stats["acc"]["mean"], clean_stats["acc"]["std"])],
        [m + s for m, s in zip(clean_stats["acc"]["mean"], clean_stats["acc"]["std"])],
        alpha=0.3
    )
    ax.semilogx(eps_adv, adv_stats["acc"]["mean"], label="fgsm")
    ax.fill_between(
        eps_adv,
        [m - s for m, s in zip(adv_stats["acc"]["mean"], adv_stats["acc"]["std"])],
        [m + s for m, s in zip(adv_stats["acc"]["mean"], adv_stats["acc"]["std"])],
        alpha=0.3
    )
    ax.set_xlabel("eps")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.savefig(result_path / f'{name}_pbc.png')
    plt.close(fig)

    # rd box plot
    repro_clean = extract_crit_norms(clean_stats)
    repro_adv   = extract_crit_norms(adv_stats)
    baseline_rd = np.array(rd_baseline_dict[name]['x'], dtype=float)

    make_boxplots(repro_clean, repro_adv, baseline_rd, name, result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str,
        help="Root path containing 'clean/' and 'fgsm/' subdirectories, "
             "e.g. 'root/binn2_khmodel/data/repro/experiments'"
    )
    args = parser.parse_args()

    result_path = Path(args.result_path)
    clean_root  = result_path / "clean"
    adv_root    = result_path / "fgsm"

    ce_loss = torch.nn.CrossEntropyLoss()

    clean_rp, clean_fgsm, clean_pgd = collect_pkl_paths(clean_root)
    adv_rp,   adv_fgsm,   adv_pgd   = collect_pkl_paths(adv_root)

    clean_experiments = build_experiments(clean_rp, clean_fgsm, clean_pgd, ce_loss)
    adv_experiments   = build_experiments(adv_rp,   adv_fgsm,   adv_pgd,   ce_loss)

    for name in ('rp', 'fgsm', 'pgd'):
        print(f"Processing: {name}")
        clean_stats = make_stats_dict(clean_experiments[name])
        adv_stats   = make_stats_dict(adv_experiments[name])
        create_plots(clean_stats, adv_stats, name, result_path)