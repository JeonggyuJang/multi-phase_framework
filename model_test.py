import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import time
import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

from pruning.methods import weight_prune, layer_prune
from pruning.utils import to_var, train, test, prune_rate, update_params, print_specific_layer, data_load_pickle
from models import *
from pruning.layers import MaskedLinear, MaskedConv2d
import pdb
"""
    This script used for check accuracy & value of parameters of Network.
"""
# Data loaders
test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=500, shuffle=True)

def test(model, loader):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))

    return acc

# Load the pretrained model-----------------------------------------------------------------------------
#folder_name = './models/codes2020/'
folder_name = './models/iccad2020/'

# Load net_info-----------------------------------------------------------------------------
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc_kc_info_94-84_multi_phased_1#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc_kc_info_97%_pruned.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG7_kc_kc_info_94-89_multi_phased_1#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87_multi_phased_1#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74_multi_phased_2#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49_multi_phased_3#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0_multi_phased_4#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0_multi_phased_5#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0-75_multi_phased_6#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0-75-50_multi_phased_7#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0-75-50-0_multi_phased_8#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc_kc_info_98-96-94-87-76-52-0_multi_phased_6#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc_kc_info_97-84_multi_phased_1#.pickle') ##
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_94-70_multi_phased_1#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84_multi_phased_1#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83_multi_phased_2#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83-70-45-0_multi_phased_5#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83-59_multi_phased_3#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-32-100-0-76-32-100_multi_phased_8#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-18_multi_phased_3#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-18-37-100_multi_phased_5#.pickle')
#pruned_net_info_from = data_load_pickle('./masks/VGG16L_CIFAR_BIG_kc_kc_info_82%_pruned.pickle')
pruned_net_info_from = data_load_pickle('./masks/VGG16L_CIFAR_BIG_kc+simd_kc_info_82-58-57-39-100_multi_phased_4#.pickle')

#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_94-70_multi_phased_1#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_94-70-41_multi_phased_2#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_94-70-41-0-40_multi_phased_4#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0-87_multi_phased_6#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_98-87-74-49-0-0-87-75-50-0_multi_phased_9#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_94-70-41-0_multi_phased_3#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc_kc_info_97-84-0_multi_phased_2#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc_kc_info_98-96-94-87-76-52_multi_phased_5#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83_multi_phased_2#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83-59-0_multi_phased_4#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG11_kc+simd_kc_info_97-84-83-28-100-0-84-28_multi_phased_7#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG7_kc_kc_info_94-89-77_multi_phased_2#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-32-100_multi_phased_4#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-32-100-0-76-32_multi_phased_7#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-18-37_multi_phased_4#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG7_kc+simd_kc_info_94-77-76-18-37_multi_phased_4#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG16L_CIFAR_BIG_kc_kc_info_82-72_multi_phased_1#.pickle')
pruned_net_info_to = data_load_pickle('./masks/VGG16L_CIFAR_BIG_kc+simd_kc_info_82-58-57-39_multi_phased_3#.pickle')
#pruned_net_info_to = data_load_pickle('./masks/VGG16L_CIFAR_BIG_kc+simd_kc_info_82-72-33-39-100-0-71-39-100_multi_phased_8#.pickle')

# Load VGG11_ Model class-----------------------------------------------------------------------------
net_from = VGG16L_CIFAR_BIG(pruned_net_info_from['cfg_list'], with_fc = True)
net_to = VGG16L_CIFAR_BIG(pruned_net_info_to['cfg_list'], with_fc = True)
#net_from = VGG7()
#net_from = VGG11()
#net_to = VGG7()
#net_to = VGG11()
#net_from = VGG7(pruned_net_info_from['cfg_list'], with_fc=True)
#net_to = VGG7(pruned_net_info_to['cfg_list'], with_fc=True)
#net_from = VGG11(pruned_net_info_from['cfg_list'], with_fc=True)
#net_to = VGG11(pruned_net_info_to['cfg_list'], with_fc=True)
#net_from = ALEXnet(pruned_net_info_from['cfg_list'], with_fc=True)
#net_to = ALEXnet(pruned_net_info_to['cfg_list'], with_fc=True)

# Update Params-----------------------------------------------------------------------------
#update_params(net_from,folder_name + 'VGG11_kc_pruned_97%.pkl')
#update_params(net_from, folder_name + 'VGG7_kc_multi_phased_94-89%_1#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87%_1#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74%_2#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74-49%_3#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74-49-0%_4#.pkl')
#update_params(net_from, folder_name + 'VGG11_kc+simd_multi_phased_98-87-74-49-0-0%_5#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74-49-0-0-75%_6#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74-49-0-0-75-50%_7#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_98-87-74-49-0-0-75-50-0%_8#.pkl')
#update_params(net_from, folder_name+'VGG11_kc_multi_phased_98-96-94-87-76-52-0%_6#.pkl')
#update_params(net_from, folder_name+'VGG11_simd_multi_phased_98-96%_1#.pkl')
#update_params(net_from, folder_name+'VGG11_simd_pruned_94%.pkl')
#update_params(net_from, folder_name + 'VGG7_kc+simd_multi_phased_94-77-76-32-100-0-76-32-100%_8#.pkl')
#update_params(net_from, folder_name + 'VGG16L_CIFAR_BIG_kc_multi_phased_82-72%_1#.pkl')
#update_params(net_from, folder_name + 'VGG16L_CIFAR_BIG_kc_multi_phased_82-72-58-35%_3#.pkl')
#update_params(net_from, folder_name + 'VGG16L_CIFAR_BIG_kc_pruned_82%.pkl')
update_params(net_from, folder_name + 'VGG16L_CIFAR_BIG_kc+simd_multi_phased_82-58-57-39-100%_4#.pkl')
#update_params(net_from, folder_name + 'VGG7_layer_multi_phased_94-80-10%_2#.pkl')
#update_params(net_from, './models/date2020/VGG11_kc+simd_multi_phased_97-84-82%_2#.pkl')
#update_params(net_from, './models/date2020/VGG11_pretrained.pkl')
#update_params(net_from, folder_name + 'ALEXnet_kc_pruned_93%.pkl')
print("prune_rate for net_from\n"); prune_rate(net_from)

#update_params(net_to, 'models/date2020/VGG11_kc_pruned_94%.pkl')
#update_params(net_to, './models/date2020/VGG11_kc_multi_phased_94-84-70%_2#.pkl')
#update_params(net_to, './models/date2020/VGG11_kc+simd_multi_phased_94-70-41%_2#.pkl')
#update_params(net_to, './models/date2020/VGG11_kc+simd_multi_phased_94-70-41-0%_3#.pkl')
#update_params(net_to, './models/date2020/VGG11_kc+simd_multi_phased_94-70-41-0-40%_4#.pkl')
#update_params(net_to, 'VGG11_kc_multi_phased_98-96%_1#.pkl')
#update_params(net_to, 'VGG11_kc_multi_phased_98-96-94%_2#.pkl')
#update_params(net_to, 'VGG11_kc_multi_phased_98-96-94-87%_3#.pkl')
#update_params(net_to, 'VGG11_kc_multi_phased_98-96-94-87-76%_4#.pkl')
#update_params(net_to, folder_name+'VGG11_kc_multi_phased_98-96-94-87-76-52%_5#.pkl')
#update_params(net_to, folder_name + 'VGG11_kc+simd_multi_phased_98-87-74-49-0-0-87%_6#.pkl')
#update_params(net_to, folder_name + 'VGG11_kc+simd_multi_phased_98-87-74-49-0-0-87-75%_7#.pkl')
#update_params(net_to, folder_name + 'VGG11_kc+simd_multi_phased_98-87-74-49-0-0-87-75-50-0%_9#.pkl')
#update_params(net_to, folder_name + 'VGG11_kc+simd_multi_phased_97-84-83-28-100-0-84-28%_7#.pkl')
#update_params(net_to, folder_name + 'VGG11_kc_multi_phased_97-84-0%_2#.pkl')
#update_params(net_to, 'VGG11_kc_multi_phased_98-96-94-87-76-52-0%_6#.pkl')
#update_params(net_to, folder_name+'VGG11_simd_multi_phased_98-96%_1#.pkl')
#update_params(net_to, folder_name+'VGG11_simd_multi_phased_98-96-93-87-75-51%_5#.pkl')
#update_params(net_to, folder_name + 'VGG7_kc_multi_phased_94-89-77%_2#.pkl')
#update_params(net_to, folder_name + 'VGG7_kc+simd_multi_phased_94-77-76-32-100-0-76-32%_7#.pkl')
#update_params(net_to, folder_name + 'ALEXnet_kc_multi_phased_93-74%_1#.pkl')
#update_params(net_to, folder_name + 'VGG16L_CIFAR_BIG_kc+simd_multi_phased_82-72-33-39%_3#.pkl')
#update_params(net_to, folder_name + 'VGG16L_CIFAR_BIG_kc_multi_phased_82-72%_1#.pkl')
update_params(net_to, folder_name + 'VGG16L_CIFAR_BIG_kc+simd_multi_phased_82-58-57-39%_3#.pkl')
#update_params(net_to, folder_name + 'VGG16L_CIFAR_BIG_kc+simd_multi_phased_82-72-33-39-100-0-71-39-100%_8#.pkl')
#update_params(net_to, folder_name + 'VGG16L_CIFAR_BIG_kc_pruned_82%.pkl')
print("prune_rate for net_to\n"); prune_rate(net_to)

# print Model Parameters
#print_specific_layer(net_from, 'batchnorm', 'mean', 0)
#print_specific_layer(net_to, 'batchnorm', 'mean', 0)
print_specific_layer(net_from, 'Conv', 'param', 0)
print_specific_layer(net_to, 'Conv', 'param', 0)
#print_specific_layer(net_from, 'Conv', 'param', 2)
#print_specific_layer(net_to, 'Conv', 'param', 2)
#print_specific_layer(net_from, 'batchnorm', None, 0)
#print_specific_layer(net_to, 'batchnorm', None, 0)
pdb.set_trace()
#test(net2, loader_test)
