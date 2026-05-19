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

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, help = 'whether running in debug mode')
    parser.add_argument('--attack_seed', type=int, help='seed for attacking')
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--model_name', type=str, default='fkhl3_cifar10_pruned.pty', help='name of the model')

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    if args.debug:
        ROOT = ROOT/"debug"

    # create directory structure
    model_path = Path(args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    figure_path = ROOT / "data/repro/figures"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    exp_path = ROOT / "data/repro/experiments"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # model filenames
    model_name = Path(args.model_name)
    print(f"model name type: {type(model_name)}", flush=True)
               
    rp_fname = Path("random_perturbation_results.pkl")
    fgsm_fname = Path("fgsm_results.pkl")
    pgd_fname = Path("pgd_results.pkl")

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
    rpE.run(model_path, [model_name], cifar10Test, eps, device, norm_p=2.0)
    rpE.save(exp_path / rp_fname)

    breakpoint()

    fgsmE = FGSMExperiment(ce_loss)
    # fgsmE.run(directory, fn_list, cifar10Test, eps, device, norm_p=2.0) #acs
    # fgsmE.save(exp_directory / fgsm_fname) # acs
    fgsmE.load(exp_directory / fgsm_fname)

    pgdE = PGDExperiment(ce_loss)
    # pgdE.run(directory, fn_list, cifar10Test, eps, device, norm_p=2.0) #acs
    # pgdE.save(exp_directory / pgd_fname) #acs
    pgdE.load(exp_directory / pgd_fname)