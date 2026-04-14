from __future__ import print_function
import argparse

from lib import *
from config import *
from model import model_dispatcher
from utils import * 
import os

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
parser.add_argument('--alpha', type=int, default=.5, help='loss balance ratio') # not implemented yet in the utils model!! TODO

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

# breakpoint()

if not os.path.exists(f'./Res{data_dir}'):
    os.mkdir(f'./Res{data_dir}')

if not os.path.exists(f'./Res{data_dir}/{attack_type}'):
    os.mkdir(f'././Res{data_dir}/{attack_type}')


fo = open(f'./Res{data_dir}/{attack_type}/results_{net_type}.txt', 'a+')

# --------------------------------------------------------------------------------------------------------------------------------------------
# Train a model first


if opt.load_model: # if model exist just load it
    save_path = opt.load_model
    net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    # breakpoint()
    net.to(device)


else:
    raise Warning("This file is for test existing models.")


# Test the clean model on clean and attacks
acc, images = test_model_clean(net, dataloader_dict)
print('Accuracy of model on clean images: %f ' % acc)
fo.write('Accuracy of model on clean images: %f \n' % acc)



for eps_t in sigmas: #[8,32,64]:

    print(f'eps_t={eps_t}')
    fo.write(f'eps_t={eps_t} \n')

    epsilons = [eps_t/255]


    # Test the clean model on clean and attacks
    net, dataloader_dict, criterior, optimizer = model_dispatcher(which_model, net_type, data_dir, inp_size, n_classes)
    load_model(net, save_path)
    net.to(device)

    acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=False)
    print('Accuracy of model on adversarial images: %f %%' % acc_attack[0])
    fo.write('Accuracy of model on adversarial images: %f \n' % acc_attack[0])


    if (net_type.lower() in ['grayedge', 'rgbedge']):
        acc_attack, images = test_model_attack(net, dataloader_dict, epsilons, attack_type, net_type, redetect_edge=True)
        print('Accuracy of model on adversarial images with redetect_edge: %f %%' % acc_attack[0])
        fo.write('Accuracy of model on adversarial images with redetect_edge: %f \n' % acc_attack[0])



fo.close()

