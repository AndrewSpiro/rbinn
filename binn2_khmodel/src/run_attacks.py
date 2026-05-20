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

    breakpoint()

    fgsmE = FGSMExperiment(ce_loss)
    fgsmE.run(model_path, fn_list, cifar10Test, eps, device, norm_p=2.0)
    fgsmE.save(exp_path / fgsm_fname)
    fgsmE.load(exp_path / fgsm_fname)

    pgdE = PGDExperiment(ce_loss)
    pgdE.run(model_path, fn_list, cifar10Test, eps, device, norm_p=2.0)
    pgdE.save(exp_path / pgd_fname)
    pgdE.load(exp_path / pgd_fname)
    

    clrs = ['c0','c1','c2','c3','c4','c5']
    names = ['KH', 'BP', 'L2', 'JReg', 'SpecReg']

    experiments = [(pgdE, 'pgdE'), (fgsmE, 'fgsmE'), (rpE, 'rpE')]
    for exp,name in experiments:
        saveKHModelPerturbationResults(exp, name)

    print('potting rd')
    fig = plt.figure()
    # fig.set_figheight(6.0)
    # width_ratios = [5e-3, 0.5, 0.5]
    # height_ratios = [0.05, 1.0, 0.05, 1.0, 0.05, 1.0]
    # gs = mpl.gridspec.GridSpec(6, 3, width_ratios=width_ratios, height_ratios=height_ratios)

    # RP_left_ax = fig.add_subplot(gs[1, 1])
    # RP_right_ax = fig.add_subplot(gs[1, 2])
    # RP_left_ax, RP_right_ax = plotPerturbationResults(
    #     (RP_left_ax, RP_right_ax),
    #     rpE, names, clrs,
    # )

    # fgsm_left_ax = fig.add_subplot(gs[3, 1])
    # fgsm_right_ax = fig.add_subplot(gs[3, 2])
    # fgsm_left_ax, fgsm_right_ax = plotPerturbationResults(
    #     (fgsm_left_ax, fgsm_right_ax),
    #     fgsmE, names, clrs,
    # )

    pdg_left_ax = fig.add_subplot() #aspiro
    pdg_right_ax = fig.add_subplot() #aspiro
    pdg_left_ax, pdg_right_ax = plotPerturbationResults( #aspiro
        (pdg_left_ax, pdg_right_ax), #aspiro
        pgdE, names, clrs, #aspiro
    ) #aspiro
    plt.show() #aspiro
    # fig.text(0.1, 0.665, r"Random Perturbation", rotation='vertical', ha='center')
    # fig.text(0.1, 0.45, r"FGSM", rotation='vertical', ha='center')
    # fig.text(0.1, 0.2, r"PGD", rotation='vertical', ha='center')
    # fig.savefig(figure_directory / Path("Figure1-AdversarialResults.eps"))
    # fig.savefig(Path("AdversarialResults.pdf"))

    print('plotting pbc and rd')
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(6.0)
    width_ratios = [5e-3, 0.5, 0.5]
    height_ratios = [0.05, 1.0, 0.05, 1.0, 0.05, 1.0]
    gs = mpl.gridspec.GridSpec(6, 3, width_ratios=width_ratios, height_ratios=height_ratios)

    # RP_left_ax = fig.add_subplot(gs[1, 1])
    # RP_right_ax = fig.add_subplot(gs[1, 2])
    # RP_left_ax, RP_right_ax = plotPerturbationResults(
    #     (RP_left_ax, RP_right_ax),
    #     rpE, names, clrs,
    # )

    # fgsm_left_ax = fig.add_subplot(gs[3, 1])
    # fgsm_right_ax = fig.add_subplot(gs[3, 2])
    # fgsm_left_ax, fgsm_right_ax = plotPerturbationResults(
    #     (fgsm_left_ax, fgsm_right_ax),
    #     fgsmE, names, clrs,
    # )

    pdg_left_ax = fig.add_subplot(gs[5, 1])
    pdg_right_ax = fig.add_subplot(gs[5, 2])
    pdg_left_ax, pdg_right_ax = plotPerturbationResults(
        (pdg_left_ax, pdg_right_ax),
        pgdE, names, clrs,
    )

    # fig.text(0.1, 0.665, r"Random Perturbation", rotation='vertical', ha='center')
    # fig.text(0.1, 0.45, r"FGSM", rotation='vertical', ha='center')
    fig.text(0.1, 0.2, r"PGD", rotation='vertical', ha='center')
    fig.savefig(figure_directory / Path("Figure1-AdversarialResults.eps"))
    #fig.savefig(Path("AdversarialResults.pdf"))

    print('saving stats')
    fig, axs = PlotResults(rpE, names, clrs)
    plt.savefig(figure_directory / Path("RandomPerturbationStats.pdf"))

    print('cell')
    eps_fgsm = np.logspace(-6, -1.5, num=400)

    print('saving fgsm exp')
    fgsmE = FGSMExperiment(ce_loss)
    fgsmE.run(directory, fn_list, cifar10Test, eps_fgsm, device, norm_p=2.0)
    fgsmE.save(exp_directory / fgsm_fname)

    print('plot results')
    fig, axs = PlotResults(fgsmE, names, clrs)
    fig.savefig(figure_directory / Path("FGSMStats.pdf"))

    print('plot pgd exp')
    eps_pgd = np.logspace(-6, -1.5, num=400)

    pgdE = PGDExperiment(ce_loss)
    pgdE.run(directory, fn_list, cifar10Test, eps_pgd, device, norm_p=2.0)
    pgdE.save(exp_directory / pgd_fname)

    print('save pgd stats')
    fig, axs = PlotResults(pgdE, names, clrs)
    fig.savefig(figure_directory / Path("PGDStats.pdf"))

    print('log files')
    # file names of the training log files
    khmodel_log_name = Path("khmodel_cifar10_log.json")
    shlp_log_name = Path("shlp_cifar10_log.json")
    shlp_l2_log_name = Path("shlp_l2_log.json")
    shlp_jreg_log_name = Path("shlp_jreg_log.json")
    shlp_specreg_log_name = Path("shlp_specreg_log.json")

    fn_log_list = [khmodel_log_name, shlp_log_name, shlp_l2_log_name, shlp_jreg_log_name, shlp_specreg_log_name]

    model_accuracy = {}

    for name, log_file in zip(names, fn_log_list):
        log = Trainers.Trainer.Logger()
        log.load(directory / log_file)
        model_accuracy[name] = log['eval_acc'][-1]

    print('acc robustness plots')
    # accuracy - robustness plots

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(1.5)
    width_ratios = [0.05, 0.5, 0.05, 0.5, 0.75]
    height_ratios = [0.05, 1.0, 0.05]
    gs = mpl.gridspec.GridSpec(3, 5, width_ratios=width_ratios, height_ratios=height_ratios)

    RP_ax = fig.add_subplot(gs[1, 1])
    RP_ax = plotAccVSRobustness(
        RP_ax,
        rpE, names, clrs, model_accuracy
    )
    RP_ax.set_xlim(0.42, 0.57)
    RP_ax.set_ylim(0.5, 3.5)
    # RP_ax.set_ylabel(r"$\| \Delta x \|_{\textrm{crit}}$")
    RP_ax.set_ylabel(r"$\| \Delta x \|_{\mathrm{crit}}$")

    pgd_ax = fig.add_subplot(gs[1, 3])
    pgd_ax = plotAccVSRobustness(
        pgd_ax,
        pgdE, names, clrs, model_accuracy
    )
    pgd_ax.set_xlim(0.42, 0.57)
    handles, labels = pgd_ax.get_legend_handles_labels()
    fig.text(0.4, 0.0, r"Test accuracy", rotation='horizontal', ha='center')

    legend_ax = fig.add_subplot(gs[1, 4])
    legend_ax.axis("off")
    legend_ax.legend(handles, labels)

    #fig.text(0.1, 0.665, r"Random Perturbation", rotation='vertical', ha='center')
    #fig.text(0.1, 0.45, r"FGSM", rotation='vertical', ha='center')
    #fig.text(0.1, 0.2, r"PGD", rotation='vertical', ha='center')
    fig.savefig(figure_directory / Path("FigureA2-AccVsRobustness.pdf"))
    #fig.savefig(Path("AccVsRobustness.pdf"))