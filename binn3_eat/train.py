from __future__ import print_function
import argparse
import os
import torch
import random
import numpy as np

from lib import *
from config import *
from model import model_dispatcher
from utils import * 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###############  IMPORTANT: Specify your edge detector in config.py #######################################################

# Training settings
parser = argparse.ArgumentParser(description='shape defence adv training')
parser.add_argument('--epochs', type=int, default=20, help='num epochs')
parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
parser.add_argument('--attack', type=str, default='FGSM', help='attack type FGSM or PGD')
parser.add_argument('--net_type', type=str, default='rgb', help='edge, rgb, grayedge, rgbedge')
parser.add_argument('--data_dir', type=str, default='MNIST', help='data directory')
parser.add_argument('--model', type=str, default='mnist', help='which model implemented in model.py')
parser.add_argument('--classes', type=int, default=10, help='number of classes')
parser.add_argument('--inp_size', type=int, default=28, help='size of the input image')
parser.add_argument('--sigmas', nargs='+', type=int)
parser.add_argument('--load_model', type=str, default='', help='path to the trained model')
parser.add_argument('--alpha', type=int, default=.5, help='loss balance ratio') 
parser.add_argument('--train_id', type=int, default=0, help='id to direct train outputs')
parser.add_argument('--save_prefix', type=str, default=None, help='save to alt folder if debugging')

parser.add_argument('--root_dir', type=str, default='.', help='Root directory for results (e.g., ./debug)')
parser.add_argument('--seed', type=int, default=0, help='Seed for training and data shuffling')
parser.add_argument('--attack_seeds', nargs='+', type=int, default=[0], help='List of seeds for repeated attacks')

opt = parser.parse_args()
print(opt)


# eg\
# python train.py --net_type rgbedge --model gtsrb  --sigmas 8 32 --data_dir GTSRB --classes 43 --epochs 10 --inp_size 64 --load_model gtsrb_rgbedge.pth


num_epochs = opt.epochs # 20
batch_size = opt.batch_size # 100
attack_type = opt.attack #'FGSM'
net_type = opt.net_type #'grayedge'
data_dir = opt.data_dir #'MNIST'
which_model = opt.model #'mnist'
n_classes = opt.classes # 10
inp_size = opt.inp_size# 28
sigmas = opt.sigmas
train_id = opt.train_id
save_prefix = opt.save_prefix

# Apply global training seed
set_seed(opt.seed)

# Define pathing based on root_dir and seed
base_path = os.path.join(opt.root_dir, f'Res{opt.data_dir}', f'seed_{opt.seed}')
attack_base_path = os.path.join(base_path, opt.attack)

if not os.path.exists(attack_base_path):
    os.makedirs(attack_base_path)

fo = open(f'{attack_base_path}/results_{net_type}.txt', 'a+')

# --------------------------------------------------------------------------------------------------------------------------------------------
# Train a model first


if opt.load_model: # if model exist just load it
    save_path = opt.load_model
    net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    net.to(device)


else:
    save_path = f'{attack_base_path}/{data_dir}_{net_type}.pth'    
    net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    net.to(device)
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs, save_path)


# Test the clean model on clean and attacks
acc, images = test_model_clean(net, dataloader_dict)
print('Accuracy of original model on clean images: %f ' % acc)
fo.write('Accuracy of original model on clean images: %f \n' % acc)



for eps_t in sigmas: #[8,32,64]:

    print(f'eps_t={eps_t}')
    fo.write(f'eps_t={eps_t} \n')

    epsilons = [eps_t/255]

    # --- REPEATED ATTACKS WITH DIFFERENT SEEDS ---
    for a_seed in opt.attack_seeds:
        set_seed(a_seed) # Reset seed for attack variability
        fo.write(f'Attack Seed: {a_seed}\n')

    # Test the clean model on clean and attacks
    net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    net.to(device)

    acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
    print('Accuracy of clean model on adversarial images: %f %%' % acc_attack[0])
    fo.write('Accuracy of clean model on adversarial images: %f \n' % acc_attack[0])


    if (net_type.lower() in ['grayedge', 'rgbedge']):
        acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of clean model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of clean model on adversarial images with redetect_edge: %f \n' % acc_attack[0])




    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Now perform adversarial training
    set_seed(opt.seed) # Return to train seed
    save_path_robust = f'{attack_base_path}/{data_dir}_{net_type}_{eps_t}_robust.pth'

    # if train_phase:    
    net_robust, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    net_robust.to(device)
    train_robust_model(net_robust, dataloader_dict, criterior, optimizer, num_epochs, save_path_robust, attack_type, eps=eps_t/255, net_type=net_type, redetect_edge=False)


    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Test the robust model on clean and attacks
    net_robust, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    load_model(net_robust, save_path_robust) 
    # load_model(net_robust, f'./{attack_type}-imagenette2-160/imagenette2-160_rgbedge_{eps_t}_robust_{eps_t}.pth')     
    # load_model(net_robust, f'./{attack_type}-gtsrb/gtsrb_rgbedge_{eps_t}_robust_{eps_t}.pth')     
    net_robust.to(device)


    acc, images = test_model_clean(net_robust, dataloader_dict)
    print('Accuracy of robust model on clean images: %f %%' % acc)
    fo.write('Accuracy of robust model on clean images: %f \n' % acc)

    # Test robust model with attack seeds
    for a_seed in opt.attack_seeds:
        set_seed(a_seed)
        acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
        print('Accuracy of robust model on adversarial images: %f %%' % acc_attack[0])
        fo.write('Accuracy of robust model on adversarial images: %f \n' % acc_attack[0])


    if (net_type.lower() in ['grayedge', 'rgbedge']):    
        acc_attack, images = test_model_attack(net_robust, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of robust  model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of robust  model on adversarial images with redetect_edge: %f \n' % acc_attack[0])


    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Now perform adversarial training with redetect

    if not (net_type.lower() in ['grayedge', 'rgbedge']): continue

    set_seed(opt.seed)
    save_path_redetect = f'{attack_base_path}/{data_dir}_{net_type}_{eps_t}_robust_redetect.pth'

    net_redetect, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    net_redetect.to(device)
    train_robust_model(net_redetect, dataloader_dict, criterior, optimizer, num_epochs, save_path_redetect, attack_type, eps=eps_t/255, net_type=net_type, redetect_edge=True)

    acc, images = test_model_clean(net_redetect, dataloader_dict)
    print('Accuracy of robust redetect model on clean images: %f %%' % acc)
    fo.write('Accuracy of robust redetect model on clean images: %f \n' % acc)

    for a_seed in opt.attack_seeds:
        set_seed(a_seed)
        acc_attack, images = test_model_attack(net_redetect, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
        print('Accuracy of robust redetect  model on adversarial images: %f %%' % acc_attack[0])
        fo.write('Accuracy of robust redetect  model on adversarial images: %f \n' % acc_attack[0])


        acc_attack, images = test_model_attack(net_redetect, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of robust redtect model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of robust redetect model on adversarial images with redetect_edge: %f \n' % acc_attack[0])




fo.close()

