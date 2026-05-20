import os
from pathlib import Path
import argparse

import torch
from torchvision.transforms import ToTensor

from binn2_khmodel.tests.context import *
from LocalLearning import ModelFactory, HiddenLayerModel, KHModel, SHLP, FKHL3
from LocalLearning.Data import BaselineAccurateTestData, LpUnitCIFAR10, DeviceDataLoader
from LocalLearning.Attacks import AdversarialAttack, WhiteGaussianPerturbation, FGSM, PGD, AttackTest
from LocalLearning.Experiments import PerturbationExperiment, RandomPerturbationExperiment, FGSMExperiment, PGDExperiment
from LocalLearning import Trainers

from pathlib import Path
from collections import OrderedDict
import pickle as pkl

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def saveKHModelPerturbationResults(exp: PerturbationExperiment, exp_path: str, attack: str):
    '''making this function to save epsilon values and accuracies -aspiro'''
    crit_norm = []
    for i, (key, results) in enumerate(exp):
        if key != 'khmodel_cifar10_pruned.pty': # KH is the target model to reproduce
            continue
        else:
            cns_nan = np.isnan(results['crit_eps'])
            crit_norm.append(results['crit_norm'][~cns_nan])
            file_path = f"{exp_path}/{attack}_attack.txt"
            with open(file_path, "w") as f:
                f.write("Epsilons\n")
                for eps in results['eps']:
                    f.write(f"{eps}\n")
                f.write("\nCritical Epsilons\n")
                for crit_eps in crit_norm[0]:
                    f.write(f"{crit_eps}\n")
                f.write("\nAccuracies\n")
                for acc in results['acc']:
                    f.write(f"{acc}\n")
            print(f"Saved adv attack results to {file_path}")

def plotPerturbationResults(axs, exp: PerturbationExperiment, names: list, clrs: list) -> tuple:
    crit_norm = []
    left_ax, right_ax = axs
    #fig, axs = plt.subplots(1, 2)
    for i, (key, results) in enumerate(exp):      
        left_ax.semilogx(results['eps'], results['acc'], color=clrs[i], label=names[i])
        #crit_eps.append(results['crit_eps'])
        # clean nans from data
        cns_nan = np.isnan(results['crit_eps'])
        crit_norm.append(results['crit_norm'][~cns_nan])
        saveKHModelPerturbationResults(key, results['eps'], results['acc'], crit_norm[0])
    #axs[1].boxplot(crit_eps, labels=names, showfliers=False)
    right_ax.boxplot(crit_norm, labels=names, showfliers=False)
    left_ax.legend(
        handlelength=0.5,
    )
    left_ax.set_xlabel(r"$\epsilon$")
    left_ax.set_ylabel("Rel. accuracy")
    right_ax.set_ylabel(r"$\| \Delta x \|_{\mathrm{crit}}$")
    
    return left_ax, right_ax

def plotAccVSRobustness(ax, exp: PerturbationExperiment, names: list, clrs: list, acc: dict):
    
    for i, (key, results) in enumerate(exp):
        cns_nan = np.isnan(results['crit_eps'])
        median = np.median(results['crit_norm'][~cns_nan])
        acc_array = np.array([acc[names[i]]])
        ax.scatter(acc_array, median, label=names[i], c=clrs[i])
    ax.set_xlabel(r"Test accuracy")
    # ax.set_ylabel(r"$\| \Delta x \|_{\textrm{crit}}$")
    ax.set_ylabel(r"$\| \Delta x \|_{\mathrm{crit}}$")
    return ax

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, help = 'whether running in debug mode')
    parser.add_argument('--attack_seed', type=int, help='seed for attacking')
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--figure_path', type=str, help="path for figures e.g., 'root/data/repro/figures'")
    parser.add_argument('--exp_path', type=str, help="path for the epxeriments e.g., 'root/binn2_khmodel/data/repro/experiments")
    parser.add_argument('--attack_models', choices=['khmodel', 'shlp', 'L2', 'jreg', 'specreg'], help="models to train")

    args = parser.parse_args()

    set_seed(args.attack_seed)

    # create directory structure
    model_path = Path(args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    figure_path = Path(args.figure_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    exp_path = Path(args.exp_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # model filenames
    khmodel_name = Path("khmodel_cifar10_pruned.pty")
    shlp_init_name = Path("shlp_init.pty")
    shlp_name = Path("shlp_cifar10.pty")
    shlp_l2_name = Path("shlp_l2.pty")
    shlp_jreg_name = Path("shlp_jreg.pty")
    shlp_specreg_name = Path("shlp_specreg.pty")

    fn_dict = {
        'khmodel': (khmodel_name,),
        'shlp': (shlp_init_name, shlp_name),
        'L2': (shlp_l2_name,),
        'jreg': (shlp_jreg_name,),
        'specreg': (shlp_specreg_name,),
    }           

    fn_list = [fn for model in args.attack_models for fn in fn_dict[model]]

    rp_fname = Path("random_perturbation_results.pkl")
    fgsm_fname = Path("fgsm_results.pkl")
    pgd_fname = Path("pgd_results.pkl")

    # hyper parameters
    BATCH_SIZE = 1000

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"args data path: {args.data_path}")
    cifar10Test = LpUnitCIFAR10(
    root=args.data_path,
    train=False,
    transform=ToTensor(),
    p=FKHL3.pSet["p"],
    )
    
    ce_loss = torch.nn.CrossEntropyLoss()
    
    eps = np.logspace(-6, np.log(2.5), num=400)
    rpE = RandomPerturbationExperiment(ce_loss)
    rpE.run(model_path, fn_list, cifar10Test, eps, device, norm_p=2.0)
    rpE.save(exp_path / rp_fname)

    eps_fgsm = np.logspace(-6, -1.5, num=400)
    fgsmE = FGSMExperiment(ce_loss)
    fgsmE.run(model_path, fn_list, cifar10Test, eps_fgsm, device, norm_p=2.0)
    fgsmE.save(exp_path / fgsm_fname)

    eps_pgd = np.logspace(-6, -1.5, num=400)
    pgdE = PGDExperiment(ce_loss)
    pgdE.run(model_path, fn_list, cifar10Test, eps, device, norm_p=2.0)
    pgdE.save(exp_path / pgd_fname)