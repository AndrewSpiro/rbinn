import os
import argparse
from pathlib import Path
import random
import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision.transforms import ToTensor

from LocalLearning import ModelFactory, HiddenLayerModel, KHModel, SHLP, FKHL3
from LocalLearning import Data
from LocalLearning import Trainers
from LocalLearning.Regularizers import LpReg, JFReg, SpecReg
from LocalLearning.Data import BaselineAccurateTestData, LpUnitCIFAR10, DeviceDataLoader
from LocalLearning.Statistics import cov_spectrum


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

def shlp_schedule(epoch):
    if epoch <= 30:
        return 1e-3
    if (epoch <= 90) and (epoch > 30):
        return 5e-4
    if (epoch <= 120) and (epoch > 90):
        return 2e-4
    if (epoch <= 160) and (epoch > 120):
        return 1e-4
    return 1e-5

def specreg_schedule(epoch):
    if epoch <= 90:
        return 1e-3
    if epoch >= 90:
        return 1e-3*np.exp(- (epoch - 90) / gamma )
    #if (epoch <= 200) and (epoch > 100):
    #    return 5e-4
    #if (epoch <= 400) and (epoch > 100):
    #    return 1e-5
    #    return 2e-4
    #if (epoch <= 160) and (epoch > 120):
    #    return 1e-4
    #return 5e-6

def specreg_schedule_adj(epoch):
    if epoch <= 200:
        return 1e-3
    if epoch >= 200:
        return 1e-3*np.exp(- (epoch - 200) / gamma )    

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
    parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
    parser.add_argument('--figure_path', type=str, help="path for the figure e.g., 'root/binn2_khmodel/data/repro/figures")
    parser.add_argument('--num_workers', type=int, help="number of workers to use")
    parser.add_argument('--train_models', choice=['khmodel', 'shlp', 'L2', 'jreg', 'specreg', 'adj_exp'], help="models to train")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for training")
    parser.add_argument('--epochs', type=int, default=1000, help="number of epochs for training")

    args = parser.parse_args()

    NUMBER_OF_EPOCHS = args.lr
    LEARNING_RATE = args.epochs

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.train_seed)

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

    shlp_pSet = khmodel.pSet.copy()

    cifar10Train, TrainLoader, cifar10Test, TestLoader = create_datasets()

    clean_acc = get_clean_acc(khmodel)
    print(f"Clean accuracy on KHModel: {clean_acc}")

    if 'khmodel' in args.train_models:
        @JFReg(alpha_JF=0.0, n=3)
        class CETrainerJac(Trainers.CETrainer):
            pass

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

        plt.savefig(str(figure_path)/"khmodel_training.png")

        fig = plt.figure(figsize = (12.9, 10))
        draw_weights(khmodel.local_learning.W.T.detach().cpu().numpy(), 20, 20)

        plt.savefig(str(figure_path)/"khmodel_weigths.png")

    if 'shlp' in args.train_models:
        shlp_pSet["n"] = 1.0

        # train the shlp model for comparison
        shlp = SHLP(shlp_pSet)
        shlp.to(device)
        init_state = shlp.state_dict()
        torch.save(init_state, model_path / shlp_init_name)

        shlp.train()
        ce_trainer = CETrainerJac(shlp, learning_rate=shlp_schedule)
        ce_trainer.run(TrainLoader, TestLoader, no_epochs=1000)
        ce_trainer.save(model_path / shlp_name, model_path / shlp_log_name)

        log_shlp = Trainers.Trainer.Logger()
        log_shlp.load(model_path / shlp_log_name)

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log_shlp["epoch"], log_shlp["loss"])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Mean training loss")

        axs[1].plot(log_shlp["epoch"], log_shlp["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log_shlp["epoch"], np.array(log_shlp["JFReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\|J\|_{F}$ - Loss")

        axs[3].plot(log_shlp["epoch"], np.array(log_shlp["eval_JFReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\|J\|_{F}$ - Test")

        plt.savefig(str(figure_path)/"shlp_training.png")

    if 'L2' in args.train_models:
        shlp_pSet["n"] = 1.0
        shlp_l2 = SHLP(shlp_pSet)
        shlp_l2.to(device)

        @LpReg(alpha_Lp=5e-4, p=2.0)
        class LpTrainer(Trainers.CETrainer):
            pass

        shlp_l2.train()
        ce_trainer = LpTrainer(shlp_l2, learning_rate=shlp_schedule)
        ce_trainer.run(TrainLoader, TestLoader, no_epochs=1000)
        ce_trainer.save(model_path / shlp_l2_name, model_path / shlp_l2_log_name)

        log_shlp_l2 = Trainers.Trainer.Logger()
        log_shlp_l2.load(model_path / shlp_l2_log_name)

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log_shlp_l2["epoch"], log_shlp_l2["loss"])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Mean training loss")

        axs[1].plot(log_shlp_l2["epoch"], log_shlp_l2["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log_shlp_l2["epoch"], np.array(log_shlp_l2["LpReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\| \cdot \|_{L2}$ - Loss")

        axs[3].plot(log_shlp_l2["epoch"], np.array(log_shlp_l2["eval_LpReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\| \cdot \|_{L2}$ - Score")

        plt.savefig(str(figure_path)/"L2_training.png")

    if 'jreg' in args.train_models:
        shlp_jreg = SHLP(shlp_pSet)
        shlp_jreg.to(device)

        @JFReg(alpha_JF=1e-3, n=3)
        class JFRegTrainer(Trainers.CETrainer):
            pass

        shlp_jreg.train()
        trainer = JFRegTrainer(shlp_jreg, learning_rate=shlp_schedule)
        trainer.run(TrainLoader, TestLoader, no_epochs=1000)
        trainer.save(model_path / shlp_jreg_name, model_path / shlp_jreg_log_name)

        log_shlp_jreg = Trainers.Trainer.Logger()
        log_shlp_jreg.load(model_path / shlp_jreg_log_name)

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log_shlp["epoch"], log_shlp_jreg["loss"])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Mean training loss")

        axs[1].plot(log_shlp["epoch"], log_shlp_jreg["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log_shlp["epoch"], np.array(log_shlp_jreg["JFReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\|J\|_{F}$ - Loss")

        axs[3].plot(log_shlp["epoch"], np.array(log_shlp_jreg["eval_JFReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\|J\|_{F}$ - Test")

        plt.savefig(str(figure_path)/"jreg_training.png")

    if 'specreg' in args.train_models:
        shlp_specreg = SHLP(shlp_pSet, batch_norm=True)
        shlp_specreg.to(device)

        @SpecReg(alpha_SR=1.0, alpha=1.0, tau=0)
        @JFReg(alpha_JF=0.0, n=3)
        class SpecRegTrainer(Trainers.CETrainer):
            pass

        gamma = 30.0

        shlp_specreg.train()
        trainer = SpecRegTrainer(shlp_specreg, learning_rate=specreg_schedule)
        trainer.run(TrainLoader, TestLoader, no_epochs=NUMBER_OF_EPOCHS)
        trainer.save(model_path / shlp_specreg_name, model_path / shlp_specreg_log_name)

        log_shlp_specreg = Trainers.Trainer.Logger()
        log_shlp_specreg.load(model_path / shlp_specreg_log_name)

        l_n = cov_spectrum(TestLoader, shlp_specreg)

        fig, axs = plt.subplots(1, 3)
        axs[0].plot(log_shlp_specreg["epoch"][3:], np.array(log_shlp_specreg["SpecReg_loss"])[3:])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Spectral Loss")

        axs[1].plot(log_shlp_specreg["epoch"][3:], np.array(log_shlp_specreg["eval_SpecReg_score"])[3:])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Spectral Score")

        n = np.arange(1, len(l_n)+1)
        l_n_np = l_n.detach().cpu().numpy()
        stringer_n = l_n_np[0] / n

        axs[2].loglog(n, l_n_np)
        axs[2].loglog(n, stringer_n, ":b")
        axs[2].set_xlabel(r"$t \; [\mathrm{n}]$")
        axs[2].set_ylabel(r"$\lambda_{n}$")

        plt.savefig(str(figure_path)/"specreg_spectra.png")

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log_shlp_specreg["epoch"], log_shlp_specreg["ce_loss"])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Cross Entropy Loss")

        axs[1].plot(log_shlp_specreg["epoch"], log_shlp_specreg["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log_shlp_specreg["epoch"], np.array(log_shlp_specreg["JFReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\|J\|_{F}$ - Loss")

        axs[3].plot(log_shlp_specreg["epoch"], np.array(log_shlp_specreg["eval_JFReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\|J\|_{F}$ - Test")

        plt.savefig(str(figure_path)/"specreg_training.png")

    if 'adj_exp' in args.train_models:
        shlp_specreg_kh = SHLP(shlp_pSet, batch_norm=True)
        shlp_specreg_kh.to(device)

        gamma = 80.0
        alpha_lambda = 2.62 

        @SpecReg(alpha_SR=1e-3, alpha=alpha_lambda, tau=10)
        @JFReg(alpha_JF=0.0, n=3)
        class SpecRegKHTrainer(Trainers.CETrainer):
            pass

        shlp_specreg_kh.train()
        trainer = SpecRegKHTrainer(shlp_specreg_kh, learning_rate=1e-3)
        trainer.run(TrainLoader, TestLoader, no_epochs=1000)
        trainer.save(model_path / shlp_specreg_kh_name, model_path / shlp_specreg_kh_log_name)

        log_shlp_specreg_kh = Trainers.Trainer.Logger()
        log_shlp_specreg_kh.load(model_path / shlp_specreg_kh_log_name)

        l_n = cov_spectrum(TestLoader, shlp_specreg_kh)

        fig, axs = plt.subplots(1, 3)
        axs[0].plot(log_shlp_specreg_kh["epoch"][900:], np.array(log_shlp_specreg_kh["SpecReg_loss"])[900:])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Spectral Loss")

        axs[1].plot(log_shlp_specreg_kh["epoch"][900:], np.array(log_shlp_specreg_kh["eval_SpecReg_score"])[900:])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Spectral Score")

        l_n_np = l_n.detach().cpu().numpy()
        n = np.arange(1, len(l_n)+1)
        stringer_n = l_n_np[0] / n
        kh_n = l_n_np[0] / (n**alpha_lambda)#2.62

        axs[2].loglog(n, l_n.detach().cpu().numpy())
        axs[2].loglog(n, stringer_n, ":b")
        axs[2].loglog(n, kh_n, ":k")
        axs[2].set_xlabel(r"$t \; [\mathrm{n}]$")
        axs[2].set_ylabel(r"$\lambda_{n}$")

        plt.savefig(str(figure_path)/"adjexp_spectra.png")

        fig, axs = plt.subplots(1, 4)
        axs[0].plot(log_shlp_specreg_kh["epoch"][1:], log_shlp_specreg_kh["ce_loss"][1:])
        axs[0].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[0].set_ylabel(r"Cross Entropy Loss")

        axs[1].plot(log_shlp_specreg_kh["epoch"], log_shlp_specreg_kh["eval_acc"])
        axs[1].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[1].set_ylabel(r"Test accuracy")

        axs[2].plot(log_shlp_specreg_kh["epoch"], np.array(log_shlp_specreg_kh["JFReg_loss"]) / len(TrainLoader.dataset))
        axs[2].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[2].set_ylabel(r"$\|J\|_{F}$ - Loss")

        axs[3].plot(log_shlp_specreg_kh["epoch"], np.array(log_shlp_specreg_kh["eval_JFReg_score"]) / len(TestLoader.dataset))
        axs[3].set_xlabel(r"$t \; [\mathrm{epochs}]$")
        axs[3].set_ylabel(r"$\|J\|_{F}$ - Test")

        plt.savefig(str(figure_path)/"adjexp_training.png")