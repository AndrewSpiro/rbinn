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
    
def saveKHModelPerturbationResults(exp: PerturbationExperiment, attack: str):
    '''making this function to save epsilon values and accuracies -aspiro'''
    crit_norm = []
    for i, (key, results) in enumerate(exp):
        if key != 'khmodel_cifar10_pruned.pty': # KH is the target model to reproduce
            continue
        else:
            cns_nan = np.isnan(results['crit_eps'])
            crit_norm.append(results['crit_norm'][~cns_nan])
            with open(f"{attack}_attack.txt", "w") as f:
                f.write("Epsilons\n")
                for eps in results['eps']:
                    f.write(f"{eps}\n")
                f.write("\nCritical Epsilons\n")
                for crit_eps in crit_norm[0]:
                    f.write(f"{crit_eps}\n")
                f.write("\nAccuracies\n")
                for acc in results['acc']:
                    f.write(f"{acc}\n")
            print("Saved adv attack results")

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, help = 'whether running in debug mode')
    parser.add_argument('--attack_seed', type=int, help='seed for attacking')
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--figure_path', type=str, help="path for figures e.g., 'root/data/repro/figures'")
    parser.add_argument('--exp_path', type=str, help="path for the epxeriments e.g., 'root/binn2_khmodel/data/repro/experiments")
    parser.add_argument('--khlayer_name', type=str, default='fkhl3_cifar10_pruned.pty', help='name of the layer')
    parser.add_argument('--attack_models', choices=['khmodel', 'shlp', 'L2', 'jreg', 'specreg'], help="models to train")
    parser.add_argument('--train_seed', type=int, help='seed for training')


    args = parser.parse_args()

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