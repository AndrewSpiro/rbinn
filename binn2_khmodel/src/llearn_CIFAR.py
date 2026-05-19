import os, sys, argparse
from pathlib import Path
import torch
from torchvision.transforms import ToTensor
from LocalLearning import FKHL3
from LocalLearning.Data import LpUnitCIFAR10
from LocalLearning.Data import DeviceDataLoader
from LocalLearning import train_unsupervised
from LocalLearning import weight_convergence_criterion
from LocalLearning import weight_mean_criterion
import random
import numpy as np

# Model parameters

model_ps = {
    "in_size": 3 * 32 ** 2,
    "hidden_size": 2000,
    "n": 4.5,
    "p": 3.0,
    "tau_l": 1.0 / 0.02,  # 1 / learning rate
    "k": 2,
    "Delta": 0.4,  # inhibition rate
    "R": 1.0,  # asymptotic weight norm radius
}

parser=argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help="path for the model e.g., 'root/binn2_khmodel/data/repro/models")
parser.add_argument('--epochs', type=int, help='number of epochs for unsupervised training')
parser.add_argument('--debug', type=bool, help='whether running in debug mode')
parser.add_argument('--train_seed', type=int, help='seed for training')
parser.add_argument('--data_path', type=str, help="path to the data e.g., 'root/data'")
args = parser.parse_args()

MODEL_PATH = Path(args.model_path)

# Unsupervised Training Hyperparameters
NO_EPOCHS = args.epochs
BATCH_SIZE = 1000

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.train_seed)

if __name__ == "__main__":
    '''
    Learns Krotov and Hopfield's local learning layer on CIFAR10 data in an 
    unsupervised fashion.

    ARGS: 
        <modelpath> (string):   path including file name to save the model to after training
    '''
    print("main called")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(MODEL_PATH.parent):
        os.makedirs(MODEL_PATH.parent)

    model = FKHL3(model_ps, sigma=1.0)
    model.to(device=device)
    print("model intialized", flush=True)
    training_data = LpUnitCIFAR10(
            root=args.data_path, train=True, transform=ToTensor(), p=model_ps["p"]
    )
    print("train data loaded", flush=True)
    dataloader_train = DeviceDataLoader(
            training_data, device=device, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
    )

    lr = 1.0 / model_ps["tau_l"] 
    def learning_rate(epoch: int) -> float:
        return (1.0 - epoch / NO_EPOCHS) * lr
    print("learning rate defined", flush=True)
    train_unsupervised(
        dataloader_train,
        model,
        device,
        MODEL_PATH,
        no_epochs=NO_EPOCHS,
        checkpt_period=NO_EPOCHS,
        learning_rate=learning_rate,
    )

    # check convergence criteria
    # weights converge towards 1.0
    if (not weight_convergence_criterion(model, 1e-2, 1e-1)) and not args.debug:
        print("Less than 10pc of weights converged close enough. Model not saved. Try running again.")
        os._exit(os.EX_OK)

    if (not weight_mean_criterion(model)) and not args.debug:
        print("Weights converged to the wrong attractor. Model not saved. Try running again.")
        os._exit(os.EX_OK)  


    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )
