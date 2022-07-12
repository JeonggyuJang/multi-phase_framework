"""
Pruning a MLP by weights with one shot
"""
# Dependencies
import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import time

from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from pruning.methods import weight_prune, layer_prune
from pruning.utils import to_var, train, test, prune_rate, flipped_grad_masks_gen, vis_class, gen_inf_masks, data_save_pickle, data_load_pickle, vis_title_gen
from pruning.layers import MaskedLinear, MaskedConv2d


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 500,
    'test_batch_size': 500,
    'num_epochs': 1000,
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'betas' : (0.9,0.999),
    'eps' : 1e-8,
    'de-pruning_perc' : 0
}
def main(train_opt, pruning_opt, retrain_opt, multi_phase_opt, vis_opt, NN_name, last_epoch, \
        pruning_rate, dev, Train_Batch, Test_Batch, lr, dP_rate,test_opt, Maskfile_id, text):
    # Image preprocessing modules
    transform_train = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # Data loaders
    train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,transform=transform_train)
    loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=param['batch_size'], shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,transform=transform_test)
    loader_test = torch.utils.data.DataLoader(test_dataset,
        batch_size=param['test_batch_size'], shuffle=True, num_workers=2)

    # GPU Device setting
    if dev != -1:
        torch.cuda.set_device(dev)
    #Hyper parameters setting
    param['pruning_perc'] = pruning_rate
    param['batch_size'] = Train_Batch
    param['test_batch_size'] = Test_Batch
    param['learning_rate'] = lr
    param['de-pruning_rate'] = float(dP_rate*0.01)
    print(param)
    ldict = locals()

    # Model import
    exec('from models import '+NN_name)


    # Model Testing
    if test_opt != None:
        code = compile('net = '+NN_name+'()','<string>','single')
        exec(code,globals(),ldict)
        net = ldict['net']
        print('selected model is '+str(type(net).__name__))
        model_dict = net.state_dict()
        pretrained_dict = torch.load('models/'+NN_name+'_'+test_opt+'.pkl')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        if torch.cuda.is_available():
            print('CUDA:'+str(dev)+' enabled.')
            net.cuda()
            if dev == -1 : net = nn.DataParallel(net,device_ids=[0,1,2])
        time_sum = 0
        for i in range(20):
            start_time = time.time()
            test(net, loader_test)
            result = time.time() - start_time
            print("--- %s seconds ---"%(result))
            time_sum += result
        print("--- avg time : %s seconds ---"%(time_sum/20))

    # Training
    if train_opt == 1:
        code = compile('net = ' + NN_name + '()', '<string>', 'single')
        exec(code,globals(),ldict)
        net = ldict['net']
        print('selected model is '+str(type(net).__name__))
        if last_epoch != 0 :
            model_dict = net.state_dict()
            pretrained_dict = torch.load('models/'+NN_name+'_'+str(last_epoch)+'.pkl')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

        if torch.cuda.is_available():
            print('CUDA:'+str(dev)+' enabled.')
            net.cuda()
            if dev == -1 : net = nn.DataParallel(net,device_ids=[0,1,2])
        print("--- training network ---")
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(net.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
        optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'],
                                    weight_decay=param['weight_decay'],
                                    momentum=0.9)
        # lr_scheduler option_1 : StepLR
        scheduler = StepLR(optimizer = optimizer, step_size = 200, gamma = 0.5)
        # lr_scheduler option_2 : ReduceLROnPlateau
        """ In order to use ReduceLROnPlateau, You have to change
            1. ./pruning/utils.py scheduler.step() : This scheduler needs loss as argument
        """
        #scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.3, patience = 10, verbose = True, threshold = 1e-4, threshold_mode = 'rel', min_lr = 0.0000004)

        # Visualization Setup
        if vis_opt is 1:
            viz = vis_class(title = vis_title_gen(NN_name, "Training", text))
        else: viz = None
        train(net, criterion, optimizer, scheduler, param, loader_train, NN_name, loader_test, viz=viz, last_epoch = last_epoch, multi_phase_opt = multi_phase_opt)
        torch.save(net.state_dict(), 'models/'+NN_name+'_pretrained.pkl')

    # Load the pretrained model
    if pruning_opt != 0:
        code = compile('net2 = '+NN_name+'()','<string>','single')
        exec(code,globals(),ldict)
        net2 = ldict['net2']
        if pruning_opt == 1:
            model_dict = net2.state_dict()
            pretrained_dict = torch.load('models/'+NN_name+'_pretrained.pkl')
            print('Model Loaded complete: ' + 'models/'+NN_name+'_pretrained.pkl')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net2.load_state_dict(model_dict)
        elif pruning_opt == 2:
            model_dict = net2.state_dict()
            #pretrained_dict = torch.load('models/'+NN_name+'_multi_phased_400.pkl') # For Temp
            pretrained_dict = torch.load('models/'+NN_name+'_pruned.pkl')
            print('Model Loaded complete: ' + 'models/'+NN_name+'_pruned.pkl')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net2.load_state_dict(model_dict)

        if torch.cuda.is_available():
            print('CUDA:'+str(dev)+' enabled.')
            net2.cuda()
            if dev == -1 : net2 = nn.DataParallel(net2,device_ids=[0,1,2])
        print("--- Test rate for loaded Network ---")
        test(net2, loader_test)

    # Prune the weights
        if multi_phase_opt == 0:
            masks = layer_prune(net2, param['pruning_perc'])
            net2.set_masks(masks,multi_phase_opt)
            print("--- {}% parameters pruned ---".format(param['pruning_perc']))
            """
            print("Check for pruning state")
            net2_params = list(net2.parameters())
            print("First conv2 layer [4][0][0]")
            print(net2_params[4][0][0])
            """
            total_prune_rate = prune_rate(net2)
            data_save_pickle(NN_name + '_masks_' + str(int(total_prune_rate)) + '.pickle', masks)

        elif multi_phase_opt == 1:
            grad_masks = data_load_pickle(NN_name+'_masks_' +str(Maskfile_id) + '.pickle')
            # grad_masks Flipping and Generation
            flipped_grad_masks = flipped_grad_masks_gen(grad_masks, param['de-pruning_rate'])

            # net2 setting mask for inference usage

            print("Test network before assign inference masks")
            test(net2, loader_test)

            print("Generating Inference mask in MultiPhase option")
            inference_masks = gen_inf_masks(grad_masks, flipped_grad_masks)

            net2.set_masks(inference_masks, multi_phase_opt)
            print("--- {}% parameters addition ---".format(param['de-pruning_rate']))
            total_prune_rate = prune_rate(net2)
            data_save_pickle(NN_name + '_masks_' + str(int(total_prune_rate))  + '.pickle', inference_masks)
        test(net2, loader_test)

    # Retraining
    if retrain_opt == 1:
        criterion2 = nn.CrossEntropyLoss()
        #optimizer2 = torch.optim.Adam(net2.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
        if train_opt == 0 and multi_phase_opt == 0:
            optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],
                                        weight_decay=param['weight_decay'], 
                                        momentum=0.9)
        elif train_opt == 0 and multi_phase_opt == 1:
            optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],
                                        weight_decay=0, 
                                        momentum=0)
        elif train_opt == 1:
            optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate']*0.1,
                                        weight_decay=param['weight_decay'],
                                        momentum=0.9)

        # lr_scheduler : StepLR
        scheduler2 = StepLR(optimizer = optimizer2, step_size = 100, gamma = 0.1)
        # lr_sheduler : ReduceLROnPlateau
        #scheduler2 = ReduceLROnPlateau(optimizer = optimizer,verbose = True, factor = 0.1)

        if multi_phase_opt is 0:
            # Visualization Setup
            if vis_opt is 1:
                viz = vis_class(title = vis_title_gen(NN_name, "Pruning_Retrain", text))
            else: viz = None
            train(net2, criterion2, optimizer2, scheduler2, param, loader_train,NN_name, loader_test, viz=viz, last_epoch = last_epoch)

        elif multi_phase_opt is 1:
            # Visualization Setup
            if vis_opt is 1:
                viz = vis_class(title = vis_title_gen(NN_name, "MultiPhase_Retrain", text))
            else: viz = None
            train(net2, criterion2, optimizer2, scheduler2, param, loader_train, NN_name,loader_test, flipped_grad_masks, multi_phase_opt, viz = viz, last_epoch = last_epoch)

    # Check accuracy and nonzeros weights in each layer
        print("--- After retraining ---")
        test(net2, loader_test)
        prune_rate(net2)

    # Save and load the entire model
        if pruning_opt == 1:
            torch.save(net2.state_dict(), 'models/'+NN_name+'_pruned.pkl')
        elif pruning_opt == 2:
            torch.save(net2.state_dict(), 'models/'+NN_name+'_multi_phased.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'SqzNet Pruning Script')
    parser.add_argument('--T',
                        type = int,
                        default = 0,
                        help = 'Train Flag Option'
                        )
    parser.add_argument('--P',
                        type = int,
                        default = 0,
                        help = 'Pruning Flag Option'
                        )
    parser.add_argument('--R',
                        type = int,
                        default = 0,
                        help = 'Retraining Flag Option'
                        )
    parser.add_argument('--M',
                        type = int,
                        default = 0,
                        help = 'MultiPhase Flag Option'
                        )
    parser.add_argument('--Vis',
                        type = int,
                        default = 0,
                        help = 'Visualization On / Off '
                        )
    parser.add_argument('--Model',
                        type = str,
                        default = None,
                        help = 'Model name'
                        )
    parser.add_argument('--Epoch',
                        type = int,
                        default = 0,
                        help = 'last epoch of selected model'
                        )
    parser.add_argument('--P_rate',
                        type = int,
                        default = 90,
                        help = 'Pruning rate'
                        )
    parser.add_argument('--Dev',
                        type = int,
                        default = 0,
                        help = 'GPU Selection'
                        )
    parser.add_argument('--Train_Batch',
                        type = int,
                        default = 500,
                        help = 'Train Batch size'
                        )
    parser.add_argument('--Test_Batch',
                        type = int,
                        default = 500,
                        help = 'Test Batch size'
                        )
    parser.add_argument('--lr',
                        type = float,
                        default = 0.1,
                        help = 'initial learning rate'
                        )
    parser.add_argument('--dP_rate',
                        type = int,
                        default = 0,
                        help = 'de-Pruning rate'
                        )
    parser.add_argument('--Test',
                        type = str,
                        default = None,
                        help = 'Selected Model Test'
                        )
    parser.add_argument('--Mask',
                        type = int,
                        default = None,
                        help = 'To loading Mask pickle file'
                        )
    parser.add_argument('--Text',
                        type = str,
                        default = "",
                        help = 'Visdom Text Indexing'
                        )
    args = parser.parse_args()
    main(args.T, args.P, args.R, args.M, args.Vis, args.Model, args.Epoch, args.P_rate, args.Dev, args.Train_Batch,\
        args.Test_Batch, args.lr, args.dP_rate, args.Test, args.Mask, args.Text)
