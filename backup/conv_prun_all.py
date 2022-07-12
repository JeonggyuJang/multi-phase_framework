"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pruning.methods import weight_prune, layer_prune
from pruning.utils import to_var, train, test, prune_rate
from models import ConvNet_prun_all

# Name of NN
NN_name = 'conv_prun_all'

# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 500, 
    'test_batch_size': 500,
    'num_epochs': 200,
    'learning_rate': 0.01,
    'weight_decay': 1e-6,
    'alpha' : 0.99,
    'momentum' : 0,
    'betas' : (0.9,0.999),
    'eps' : 1e-8
}
# Image preprocessing modules
transform = transforms.Compose([
	transforms.Pad(4),
	transforms.RandomHorizontalFlip(),
	transforms.RandomCrop(32),
	transforms.ToTensor()])

# Data loaders
train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True, 
    transform=transform)
loader_train = torch.utils.data.DataLoader(dataset=train_dataset, 
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True, 
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=True)

# Training
#'''
net = ConvNet_prun_all()
if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
print("--- training network ---")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],alpha=param['alpha'],eps=param['eps'],weight_decay=param['weight_decay'],momentum=param['momentum']) 
#optimizer = torch.optim.Adam(net.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
#optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], 
 #                               weight_decay=param['weight_decay'],
  #                              momentum=0.9, nesterov=True)

train(net, criterion, optimizer, param, loader_train, loader_test)
torch.save(net.state_dict(), 'models/'+NN_name+'_pretrained.pkl')
#'''

# Load the pretrained model
net2 = ConvNet_prun_all()
net2.load_state_dict(torch.load('models/'+NN_name+'_pretrained.pkl'))
#net.load_state_dict(torch.load('models/'+NN_name+'_pruned.pkl'))
if torch.cuda.is_available():
    print('CUDA ensabled.')
    net2.cuda()
print("--- Pretrained network loaded ---")
test(net2, loader_test)

# prune the weights
masks = layer_prune(net2, param['pruning_perc'])
net2.set_masks(masks)
print("--- {}% parameters pruned ---".format(param['pruning_perc']))
prune_rate(net2)
test(net2, loader_test)

# Retraining
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.RMSprop(net2.parameters(), lr=param['learning_rate'],alpha=param['alpha'],eps=param['eps'],weight_decay=param['weight_decay'],momentum=param['momentum']) 
#optimizer2 = torch.optim.Adam(net2.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
#optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], 
 #                               weight_decay=param['weight_decay'],
  #                              momentum=0.9, nesterov=True)

train(net2, criterion2, optimizer2, param, loader_train, loader_test)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(net2, loader_test)
prune_rate(net2)


# Save and load the entire model
torch.save(net2.state_dict(), 'models/'+NN_name+'_pruned.pkl')
#torch.save(net.state_dict(), 'models/'+NN_name+'_pruned2.pkl')
