import os
import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor

from LocalLearning import ModelFactory, HiddenLayerModel, KHModel, SHLP, FKHL3
from LocalLearning import Data
from LocalLearning import Trainers
from LocalLearning.Regularizers import LpReg, JFReg, SpecReg
from LocalLearning.Data import BaselineAccurateTestData, LpUnitCIFAR10, DeviceDataLoader

def create_datasets(batch_size=1000, num_workers=4):
    '''define training and test datasets for training'''
    cifar10Train= Data.LpUnitCIFAR10(
        root="../data/CIFAR10",
        train=True,
        transform=ToTensor(),
        p=khlayer.pSet["p"],
    )

    TrainLoader = Data.DeviceDataLoader(
        cifar10Train,# define batch window in the global index coordinates
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    cifar10Test = Data.LpUnitCIFAR10(
        root="../data/CIFAR10",
        train=False,
        transform=ToTensor(),
        p=khlayer.pSet["p"],
    )

    TestLoader = Data.DeviceDataLoader(
        cifar10Test,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return cifar10Train, TrainLoader, cifar10Test, TestLoader

def get_clean_acc(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in TrainLoader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def minmaxnorm(a):
    a_ = a - a.min()
    return a_ / a_.max()

def draw_weights(synapses, n_hiddenx, n_hiddeny, n_pixelx=32, n_pixely=32, n_chan=3):
    cnt = 0
    synapses = synapses[np.random.choice(synapses.shape[0], n_hiddenx*n_hiddeny, replace=False)]
    HM = np.zeros((n_pixely*n_hiddeny, n_pixelx*n_hiddenx, n_chan))
    for y in range(n_hiddeny):
        for x in range(n_hiddenx):
            HM[y*n_pixely:(y+1)*n_pixely, x*n_pixelx:(x+1)*n_pixelx, :] = minmaxnorm(synapses[cnt].reshape(n_pixely, n_pixelx, n_chan))
            cnt += 1

    nc = np.max(np.absolute(HM))
    im = plt.imshow(HM, vmin=-nc, vmax=nc)
    plt.axis('off')

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--figure_path', type=str, help="path for the figure e.g., 'root/binn2_khmodel/data/repro/figures")
    parser.add_argument('--num_workers', type=int, help="number of workers to use")
    parser.add_argument('--train_models', choice=['khmodel'], help="models to train")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for training")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs for training")

    args = parser.parse_args()

    NUMBER_OF_EPOCHS = args.lr
    LEARNING_RATE = args.epochs

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

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # khmodel path variables
    khlayer_name = Path("fkhl3_cifar10_pruned.pty")
    khmodel_name = Path("khmodel_cifar10_pruned.pty")
    khmodel_log_name = Path("khmodel_cifar10_log.json")

    # shlp initialisation
    shlp_init_name = Path("shlp_init.pty")

    # shlp without regularization path variables
    shlp_name = Path("shlp_cifar10.pty")
    shlp_log_name = Path("shlp_cifar10_log.json")

    # shlp with L2 reg path variables
    shlp_l2_name = Path("shlp_l2.pty")
    shlp_l2_log_name = Path("shlp_l2_log.json")

    # shlp with Jacobian regularization path variables
    shlp_jreg_name = Path("shlp_jreg.pty")
    shlp_jreg_log_name = Path("shlp_jreg_log.json")

    # shlp with Spectral Regularization path variables
    shlp_specreg_name = Path("shlp_specreg.pty")
    shlp_specreg_log_name = Path("shlp_specreg_log.json")

    # shlp with Spectral Regularization and Parameters similar to Krotov and Hopfield Exponent on CIFAR10
    shlp_specreg_kh_name = Path("shlp_specreg_kh.pty")
    shlp_specreg_kh_log_name = Path("shlp_specreg_kh_log.json")

    # load the local learning model
    model_info = torch.load(model_path / khlayer_name)
    state_dict = model_info['model_state_dict']
    khlayer = FKHL3(state_dict)
    khmodel = KHModel(khlayer, no_classes=10)
    khmodel.to(device)

    cifar10Train, TrainLoader, cifar10Test, TestLoader = create_datasets()

    clean_acc = get_clean_acc(khmodel)
    print(f"Clean accuracy on KHModel: {clean_acc}")

    @JFReg(alpha_JF=0.0, n=3)
    class CETrainerJac(Trainers.CETrainer):
        pass

    if 'khmodel' in args.train_models:
        khmodel.train()
        ce_trainer = CETrainerJac(khmodel, learning_rate=LEARNING_RATE)
        ce_trainer.run(TrainLoader, TestLoader, no_epochs=NUMBER_OF_EPOCHS)
        ce_trainer.save(model_path / khmodel_name, model_path / khmodel_log_name)

        log = Trainers.Trainer.Logger()
        log.load(model_path / khmodel_log_name)

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log["epoch"], log["loss"])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Mean training loss")

        axs[1].plot(log["epoch"], log["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log["epoch"], np.array(log["JFReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\|J\|_{F}$ - Loss")

        axs[3].plot(log["epoch"], np.array(log["eval_JFReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\|J\|_{F}$ - Test")

        fig = plt.figure(figsize = (12.9, 10))
        draw_weights(khmodel.local_learning.W.T.detach().cpu().numpy(), 20, 20)

    elif 'shlp' in args.train_models:
        ...