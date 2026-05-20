import os
from pathlib import Path
import argparse

import torch
from torchvision.transforms import ToTensor
from torch import Tensor


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

class IdentityModel(HiddenLayerModel):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(start_dim=1)
        
    def hidden(self, x: Tensor) -> Tensor:
        return self.flat(x)
    
    def _forward(self, x: Tensor) -> Tensor:
        h = self.hidden(x)
        return (h, h)

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--debug', type=str2bool, help = 'whether running in debug mode')
    parser.add_argument('--attack_seed', type=int, help='seed for attacking')
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--khlayer_name', type=str, default='fkhl3_cifar10_pruned.pty', help='name of the layer')
    parser.add_argument('--attack_models', choice=['khmodel', 'shlp', 'L2', 'jreg', 'specreg'], help="models to train")


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

    BATCH_SIZE = 1000

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cifar10Test = LpUnitCIFAR10(
    root=args.data_path,
    train=False,
    transform=ToTensor(),
    p=FKHL3.pSet["p"],
    )

    idmodel = IdentityModel()

    TestLoader = DeviceDataLoader(
                cifar10Test,
                device=device,
                batch_size=BATCH_SIZE,
                num_workers=4,
                shuffle=False,
            )

    rp_fname = Path("random_perturbation_results.pkl")
    fgsm_fname = Path("fgsm_results.pkl")
    pgd_fname = Path("pgd_results.pkl")


    print(f"args data path: {args.data_path}")
    
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