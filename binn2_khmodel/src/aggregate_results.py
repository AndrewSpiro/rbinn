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
    print(f"max eps: {max(stats['eps'])}, min eps: {min(stats['eps'])}")
    print(f"max crit_eps: {max([max(x) for x in stats['crit_eps']])}, min eps: {min([min(x) for x in stats['crit_eps']])}")
    print(f"max crit_norm: {max([max(x) for x in stats['crit_norm']])}, min eps: {min([min(x) for x in stats['crit_norm']])}")
    return stats

def get_orig_results():

    pbc_baseline_dict = dict()
    pbc_baselines = [('rp','orig_rp_pbc.csv'), ('fgsm', 'orig_fgsm_pbc.csv'), ('pgd', 'orig_pgd_pbc.csv')]
    for (name, fn) in pbc_baselines:
        x = []
        y = []
        with open(result_path/fn, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x.append(row[0])
                y.append(row[1])
            pbc_baseline_dict[name] = {
                'x': x,
                'y': y
            }

    rd_baseline_dict = dict()
    rd_baselines = [('rp', 'orig_rp_rd.csv'), ('fgsm', 'orig_fgsm_rd.csv'), ('pgd', 'orig_pgd_rd.csv')]
    for (name, fn) in rd_baselines:
        x = []
        with open(result_path/fn, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                x.append(row[1]) # row with id 1 because of csv saving from webplot digi
            rd_baseline_dict[name] = {
                'x': x
            }
    
    return pbc_baseline_dict, rd_baseline_dict
                
def create_plots(stats_dict, name, result_path):

    pbc_baseline_dict, rd_baseline_dict = get_orig_results()
    
    print(stats_dict.keys())
    eps = np.array(stats_dict["eps"], dtype=float)
    acc_mean = stats_dict["acc"]["mean"]
    acc_std = stats_dict["acc"]["std"]

    fig, ax = plt.subplots()
    ax.semilogx(np.array(pbc_baseline_dict[name]['x'], dtype=float), np.array(pbc_baseline_dict[name]['y'], dtype=float), label="original")
    ax.semilogx(eps, acc_mean, label="repro")
    ax.fill_between(eps, [m - s for m, s in zip(acc_mean, acc_std)], [m + s for m, s in zip(acc_mean, acc_std)], alpha=0.3)
    ax.set_xlabel("eps")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.savefig(result_path/f'{name}_pbc.png')
    plt.close(fig)

    all_crit_norm = []
    for crit_eps, crit_norm in zip(stats_dict['crit_eps'], stats_dict['crit_norm']):
        crit_eps = np.array(crit_eps)
        crit_norm = np.array(crit_norm)
        cns_nan = np.isnan(crit_eps)
        all_crit_norm.append(crit_norm[~cns_nan])

    repro = np.concatenate(all_crit_norm)
    baseline_rd = np.array(rd_baseline_dict[name]['x'], dtype=float)
    whislo, q1, median, q3, whishi = baseline_rd
    baseline_stats = {
        'whislo': whislo,
        'q1': q1,
        'med': median,
        'q3': q3,
        'whishi': whishi,
        'fliers': []
    }

    repro_stats = {
        'whislo': repro.min(),
        'q1': np.percentile(repro, 25),
        'med': np.median(repro),
        'q3': np.percentile(repro, 75),
        'whishi': repro.max(),
        'fliers': []
    }

    fig, ax = plt.subplots()
    ax.bxp([baseline_stats, repro_stats], positions=[1,2])
    ax.set_xticks([1,2])
    ax.set_xticklabels(['original', 'repro'])
    ax.set_ylabel("critical norm")
    fig.savefig(result_path/f'{name}_rd.png')
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help="path for the epxeriments e.g., 'root/binn2_khmodel/data/repro/experiments")
    args = parser.parse_args()

    result_path = Path(args.result_path)

    rp_pkl_paths = []
    fgsm_pkl_paths = []
    pgd_pkl_paths = []

    for subdir, dir, file in os.walk(result_path):
        if "random_perturbation_results.pkl" in file:
            rp_pkl_path = Path(subdir)/Path("random_perturbation_results.pkl")
            # print(f"found: {str(rp_pkl_path)}")
            rp_pkl_paths.append(rp_pkl_path)
        if "fgsm_results.pkl" in file:
            fgsm_pkl_path = Path(subdir)/Path("fgsm_results.pkl")
            # print(f"found: {str(fgsm_pkl_path)}")
            fgsm_pkl_paths.append(fgsm_pkl_path)
        if "pgd_results.pkl" in file:
            pgd_pkl_path = Path(subdir)/Path("pgd_results.pkl")
            # print(f"found: {str(pgd_pkl_path)}")
            pgd_pkl_paths.append(pgd_pkl_path)

    ce_loss = torch.nn.CrossEntropyLoss()

    experiments = {
        'rp': {
            'pkl_paths': rp_pkl_paths,
            'exp': RandomPerturbationExperiment(ce_loss)
        },
        'fgsm': {
            'pkl_paths': fgsm_pkl_paths,
            'exp': FGSMExperiment(ce_loss)
        },
        'pgd': {
            'pkl_paths': pgd_pkl_paths,
            'exp': PGDExperiment(ce_loss)
        }
    }
    
    for name, exp_dict in experiments.items():
        print(name)
        stats = make_stats_dict(exp_dict)
        create_plots(stats, name, result_path)    