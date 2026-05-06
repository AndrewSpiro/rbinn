from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from shutil import copyfile
from datetime import datetime
from cnnf.model_cifar import WideResNet
from cnnf.model_mnist import CNNF
from utils import seed_torch
import json

# version issue- resolving manuall -aspiro
import sys
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, (list, tuple)):
        for v in x:
            zero_gradients(v)

import torch.autograd.gradcheck
sys.modules['torch.autograd.gradcheck'].zero_gradients = zero_gradients

from eval import Evaluator
import numpy as np
import os

def main():

    print("Running main")

    parser = argparse.ArgumentParser(description='CNNF testing')
    parser.add_argument('--dataset', choices=['cifar10', 'fashion'],
                        default='cifar10', help='the dataset for training the model')
    parser.add_argument('--data-dir', type=str, default = 'data', help='path to dataset root')
    parser.add_argument('--test', choices=['average', 'last','other'],
                        default='average', help='output averaged logits or logits from the last iteration')
    parser.add_argument('--results-path', default='results_temp.json',
                        help='Directory for Saving the Evaluation results')
    parser.add_argument('--model-dir', default='models',
                        help='Directory for Saved Models')
    parser.add_argument('--seed', type=int, default=0) # for variance in tests -acs
    parser.add_argument('--bool-debug', type=bool, default=False, help='flag for debugging')
    parser.add_argument('--attack-model', type=str, required=False, default=None, help='model for transfer attacks')
    parser.add_argument('--target-model', type=str, required=True, help='model being evaluated')

    args = parser.parse_args()
    print("Args parsed")
    seed_torch(seed=args.seed)

    if args.bool_debug:
        print(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    clean_dir = f"{args.data_dir}/"
    
    # load in data
    if args.dataset=='cifar10':
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
            batch_size=1, shuffle=False,
            num_workers=4, pin_memory=True)
        eps = 0.063
        eps_iter = 0.02
        nb_iter = 7

    elif args.dataset == 'fashion':
        dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
            batch_size=100, shuffle=True)
        eps = 0.2
        eps_iter = 0.071
        nb_iter = 7
    print("dataset loaded")

    evalmethod = args.test
    model_dir = args.model_dir

    results = {}

    # Attacker model
    model1 = None
    if args.dataset=='cifar10':
        if args.attack_model:
            model1_name = args.attack_model
            model1_path = os.path.join(model_dir, model1_name)
            model1 = WideResNet(40, 10, 2, 0.0, ind=0, cycles=0, res_param=0.0).to(device)
            model1.load_state_dict(torch.load(model1_path))

    elif args.dataset == 'fashion':
        model1_name = 'CNN_fmnist.pt'
        model1_path = os.path.join(model_dir, model1_name)
        model1 = CNNF(10, 0, 0, 0.0).to(device)
        model1.load_state_dict(torch.load(model1_path))

    # Model to evaluate
    if args.dataset=='cifar10':
        model_name = args.target_model
        model = WideResNet(40, 10, 2, 0.0, ind=5, cycles=2, res_param=0.1).to(device)
    elif args.dataset == 'fashion':
        model_name = 'CNNF_1_fmnist.pt'
        model = CNNF(10, ind=2, cycles=1, res_param=0.1).to(device)    
    print("model loaded")

    model_path = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_path))
    eval = Evaluator(device, model)
    print("getting clean accuracy...")
    clean_acc = eval.clean_accuracy(dataloader, test=evalmethod)
    results['clean_acc'] = clean_acc

    if args.bool_debug:    
        with open(args.results_path, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {args.results_path}")
        return

    # adv attack
    pgd_acc_first = eval.attack_pgd(dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, ete=False, nb_iter=nb_iter)
    results['pgd_acc_first'] = pgd_acc_first
    pgd_acc_ete = eval.attack_pgd(dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, ete=True, nb_iter=nb_iter)
    results['pgd_acc_ete'] = pgd_acc_ete

    spsa_acc_first = eval.attack_spsa(dataloader, test=evalmethod, epsilon=eps, ete=False, nb_iter=nb_iter)
    results['spsa_acc_first'] = spsa_acc_first
    spsa_acc_ete = eval.attack_spsa(dataloader, test=evalmethod, epsilon=eps, ete=True, nb_iter=nb_iter)
    results['spsa_acc_ete'] = spsa_acc_ete

    if model1:
        transfer_acc = eval.attack_pgd_transfer(model1, dataloader, test=evalmethod, epsilon=eps, eps_iter=eps_iter, nb_iter=nb_iter)
        results['transfer_acc'] = transfer_acc

    with open(args.results_path, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {args.results_path}")

if __name__ == '__main__':
    main()

