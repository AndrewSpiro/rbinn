import os, sys
import subprocess
from pathlib import Path
import argparse

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
    '''
    Creates the basic data structure in /data to start reproducing results from scratch, 
    thereby creating another sample
    '''

    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='number of epochs for unsupervised training')
    parser.add_argument('--debug', type=str2bool, help = 'whether running in debug mode')
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    if args.debug:
        ROOT = ROOT/"debug"

    # create directory structure
    model_path = ROOT / "data/repro/models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    figure_path = ROOT / "data/repro/figures"
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    exp_path = ROOT / "data/repro/experiments"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # learn bio-layer on CIFAR10 data according to Krotov and Hopfield
    fkhl3_name = Path("fkhl3_cifar10.pty")
    subprocess.call(
        f"python src/llearn_CIFAR.py --model_path {str(model_path / fkhl3_name)} --epochs {args.epochs} --debug {args.debug}",
        shell=True,
        )

    # prune the previously learned model and create Figure A1
    subprocess.call(
        f"python src/prune_and_plot_FKHL3_CIFAR.py --model_path {str(model_path / fkhl3_name)} --figure_path {str(figure_path)} --epochs {args.epochs}",
        shell=True,
        )
