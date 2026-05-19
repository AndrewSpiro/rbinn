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
    parser.add_argument('--train_seed', type=int, help='seed for training')
    parser.add_argument('--model_path', type=str, help="path for models e.g., 'root/data/repro/models'")
    parser.add_argument('--figure_path', type=str, help="path for figures e.g., 'root/data/repro/figures'")
    parser.add_argument('--data_path', type=str, help="path for the data e.g., 'root/data'")
    args = parser.parse_args()
    print("args parsed", flush=True)
    ROOT = Path(__file__).resolve().parent.parent
    if args.debug:
        ROOT = ROOT/"debug"

    # create directory structure
    model_path = Path(args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    figure_path = Path(args.figure_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    exp_path = ROOT / "data/repro/experiments"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # learn bio-layer on CIFAR10 data according to Krotov and Hopfield
    fkhl3_name = Path("fkhl3_cifar10.pty")
    print(f"data path: {args.data_path}", flush=True)
    subprocess.call(
        f"python src/llearn_CIFAR.py --model_path {str(model_path / fkhl3_name)} \
            --epochs {args.epochs} \
            --debug {args.debug} \
            --train_seed {args.train_seed} \
            --data_path {args.data_path}",
        shell=True,
        )

    # prune the previously learned model and create Figure A1
    subprocess.call(
        f"python src/prune_and_plot_FKHL3_CIFAR.py --model_path {str(model_path / fkhl3_name)} --figure_path {str(figure_path)} --epochs {args.epochs} --debug {args.debug} --train_seed {args.train_seed}",
        shell=True,
        )
