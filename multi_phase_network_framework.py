"""
Pruning a MLP by weights with one shot
"""
# Dependencies
import os
import sys
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import time
import numpy as np
import pdb

from models import *
from zeroGradSGD import mySGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from pruning.methods import weight_prune, layer_prune, simd_prune, simd_deprune, kc_prune, kc_resnet_prune, KC_initialize, KC_de_pruning, simd_prune_scalpel, gpu_prune
from pruning.utils import weight_initialize, to_var, train, test, prune_rate, gen_grad_masks, vis_class, gen_inf_masks, data_save_pickle, data_load_pickle, vis_title_gen, save_metadata, save_metadata_scattered, print_specific_layer, count_specific_layer, log_to_txt, update_params, str2bool, cp_train, cp_simd_train
from pruning.layers import MaskedLinear, MaskedConv2d

from pprint import pprint # For print dictionary

# Dataset MEAN & STD
dataset_mean = {
'cifar10': (0.4914, 0.4822, 0.4465),
'cifar100': (0.4914, 0.4822, 0.4465),
}

dataset_std = {
'cifar10': (0.2023, 0.1994, 0.2010),
'cifar100': (0.2023, 0.1994, 0.2010),
}

# Hyperparameters
log_file_name = 'log.txt'
#folder_name = './models/date2020/'
#folder_name = 'models/dac2020/'
#folder_name = 'models/codes2020/'
#folder_name = 'models/iccad2020/'
#folder_name = 'models/kcc2020/'
#folder_name = 'models/ieee_access/'
folder_name = 'models/ieee_access_jjg/'
#folder_name = 'models/gpu_pruning/'
#folder_name = 'models/resnet/'
#folder_name = 'models/imagenet/'
param = {
    'pruning_perc': 90.,
    'batch_size': 64,
    'test_batch_size': 500,
    'num_epochs': 200,
    'learning_rate': 0.1,
    #'weight_decay': 5e-4,
    'weight_decay':1e-4,
    #'weight_decay':0,
    'betas' : (0.9,0.999),
    'eps' : 1e-8,
    'de-pruning_rate' : 0,
    'rbs' : 3008,
    'LR_step' : 50,
    'dropout_rate' : 0.5
}

def main(momentum, train_opt, pruning_opt, pruning_method, retrain_opt, multi_phase_opt, multi_phase_count, vis_opt, NN_name, last_epoch, training_epoch,\
        pruning_rate, dev, train_batch, test_batch, lr, dP_rate, test_opt, Maskfile_id, text, Meta_flag, Dataset, LR_step,\
        Weight_decay, RBS, with_fc, score_margin, dropout_rate, gamma, MaskTrain, pattern):

    # Hyper parameters setting
    print("\n==> Setting Hyperparameters..")
    param['rbs'] = RBS
    param['num_epochs'] = training_epoch
    param['pruning_perc'] = pruning_rate
    param['batch_size'] = train_batch
    param['test_batch_size'] = test_batch
    param['learning_rate'] = lr
    param['de-pruning_rate'] = float(dP_rate*0.01)
    param['LR_step'] = LR_step
    param['weight_decay'] = Weight_decay
    param['with_fc'] = with_fc
    param['dropout_rate'] = dropout_rate
    param['momentum'] = momentum
    param['gamma'] = gamma
    param['MaskTrain_flag'] = MaskTrain
    pprint(param)
    ldict = locals()

    # Setting Image preprocessing modules & Data loaders
    if Dataset == 10:
        print("\n==> Setting Data Loader for Training / Testing.. : CIFAR10")
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean['cifar10'], dataset_std['cifar10']),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean['cifar10'], dataset_std['cifar10']),
            ])
        train_dataset = datasets.CIFAR10(root='../data/',train=True, download=True,transform=transform_train)
    elif Dataset == 100:
        print("\n==> Setting Data Loader for Training / Testing.. : CIFAR100")
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean['cifar100'], dataset_std['cifar100']),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean['cifar100'], dataset_std['cifar100']),
            ])
        train_dataset = datasets.CIFAR100(root='../data/',train=True, download=True,transform=transform_train)
    loader_train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=param['batch_size'], shuffle=True, num_workers=2)

    if Dataset == 10:  test_dataset = datasets.CIFAR10(root='../data/', train=False, download=True,transform=transform_test)
    elif Dataset == 100:  test_dataset = datasets.CIFAR100(root='../data/', train=False, download=True,transform=transform_test)
    if score_margin == False:
        loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=param['test_batch_size'], shuffle=True, num_workers=2)
    else:
        loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=param['test_batch_size'], shuffle=False, num_workers=2)

    # GPU Device setting
    print("\n==> Setting GPU and Copy Network to GPU.. : ")
    if len(dev) ==  1:
        print(' * Selected Single GPU : CUDA '+str(dev)+' enabled.')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)[1:-1]
        torch.cuda.set_device(0)
    else:
        print(' * Selected Multi GPU : CUDA {} enabled.'.format(dev))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(dev)[1:-1]
        dev = [x for x in range(len(dev))]

    # Model import
    print("\n==> Model importing..")
    print(" * Selected Model : {}".format(NN_name))


    # Model Testing : SEQUENCE  #0 (MODEL TESTING)
    if test_opt != None:
        print("\n==> Model Testing (Check for Accuracy & Execution Time)..")

        if pruning_method == 'kc':
            if multi_phase_count ==0:
                pruned_net_info = data_load_pickle('./masks/' + NN_name + '_' + pruning_method + '_kc_info_' +str(Maskfile_id)+'%'+ '_pruned.pickle')
            else:
                pruned_net_info = data_load_pickle('./masks/' + NN_name + '_' + pruning_method + '_kc_info_' +str(Maskfile_id)+\
                        '_multi_phased_'+str(multi_phase_count)+'#.pickle')
            net = globals()[NN_name](cfg_kc=pruned_net_info['cfg_list'], with_fc=param['with_fc'])
        else:
            code = compile('net = '+NN_name+'()','<string>','single')
            exec(code,globals(),ldict)
            net = ldict['net']
        print(' * Selected model is ' + str(type(net).__name__))
        filename = folder_name + NN_name + '_' + test_opt + '.pkl'
        update_params(net, filename)
        print(net)
        prune_rate(net)

        print("\n==> Change format of Parameters to cuda Before test or training..")
        if torch.cuda.is_available():
            net.cuda()
            if len(dev) != 1:
                net = nn.DataParallel(net, device_ids = dev)

        time_sum, test_try = 0, 3
        for i in range(test_try):
            start_time = time.time()
            test(net, loader_test)
            ex_time = time.time() - start_time
            print("<--- {} seconds for test #{} --->".format(ex_time, i))
            time_sum += ex_time
        print("<--- Testing finished,  Avg time (%s seconds) --->" %(time_sum/test_try))

    # Training : SEQUENCE #1 (INITIAL TRAINING) 
    if train_opt == 1:
        print("\n==> Model Training Setup : < INITIAL TRAINING >")
        code = compile('net = ' + NN_name + '()', '<string>', 'single')
        exec(code, globals(), ldict)
        net = ldict['net']

        print(' * Selected model is ' + str(type(net).__name__))
        print(net)
        if last_epoch != 0 :
            filename = './checkpoint/' + NN_name + '_' + str(last_epoch) + '.pkl'
            update_params(net, filename)

        print("\n==> Change format of Parameters to cuda Before test or training..")
        if torch.cuda.is_available():
            net.cuda()
            if len(dev) != 1:
                net = nn.DataParallel(net, device_ids = dev)

        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(net.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
        optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'],
                                    weight_decay=param['weight_decay'],
                                    momentum=param['momentum'])
        # lr_scheduler option_1 : StepLR
        scheduler = 0
        if len(param['LR_step']) == 1: scheduler = StepLR(optimizer = optimizer, step_size = param['LR_step'][0], gamma = param['gamma'])
        else: scheduler = MultiStepLR(optimizer = optimizer,gamma= param['gamma'], milestones=param['LR_step'])
        # lr_scheduler option_2 : ReduceLROnPlateau
        """ In order to use ReduceLROnPlateau, You have to change
            1. ./pruning/utils.py scheduler.step() : This scheduler needs loss as argument
        """
        #scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.3, patience = 10, verbose = True,\
        #                              threshold = 1e-4, threshold_mode = 'rel', min_lr = 0.0000004)

        # Visualization Setup
        if vis_opt:  viz = vis_class(title = vis_title_gen(NN_name, "Training", text))
        else: viz = None
        print("\n==> Training Start.. : < INITIAL TRAINING >")
        train_time, nowtime_text =\
                train(net, criterion, optimizer, scheduler, param, loader_train, NN_name,\
                        loader_test, viz=viz, last_epoch = last_epoch, multi_phase_opt = multi_phase_opt, pruning_method = pruning_method,loader_test=loader_test)
        print('\n==> Training Finished. Trained Model will be saved ' + 'models/backup/' + NN_name + '_pretrained.pkl')
        torch.save(net.state_dict(), folder_name + NN_name+'_pretrained.pkl')
        log_to_txt(log_file_name, folder_name + NN_name+'_pretrained.pkl    ' + str(train_time) + '/' + nowtime_text)

    # Ready for SEQ 2, 3 : INITIAL PRUNING & MULTIPHASE (Load Network to Pruning & Retraining)
    if pruning_opt != 0: # pruning_opt ==1(Initial) / pruning_opt == 2(Multiphase)
        if pruning_method == 'kc' and multi_phase_opt == 1:
            # Load Masks to make gradient_masks
            print("\n==> Load Masks or Metadata before Pruning..")
            if multi_phase_count == 1: # Load Pruned pickle file
                pruned_net_info = data_load_pickle('./masks/' + NN_name + '_' + pruning_method + '_kc_info_' +str(Maskfile_id)+'%'+ '_pruned.pickle')
                pprint(pruned_net_info)
            else: # Load Multi_phase_pickle
                pruned_net_info = data_load_pickle('./masks/' + NN_name+ '_' + pruning_method + '_kc_info_'\
                                               +str(Maskfile_id) + '_multi_phased_'+str(multi_phase_count-1)+'#.pickle')
                pprint(pruned_net_info)

            net2 = globals()[NN_name](cfg_kc=pruned_net_info['cfg_list'], with_fc=param['with_fc'])

        else:# pruning_opt == 0(Initial pruning net2=VGG~)
            code = compile('net2 = '+NN_name+'()', '<string>', 'single')
            exec(code,globals(),ldict)
            net2 = ldict['net2']

        # SEQ2 : INITIAL PRUNING : Load pkl file and update parameters
        if pruning_opt == 1:
            print("\n==> Load Model file before Pruning.. : < INITIAL PRUNING >")
            filename = folder_name + NN_name + '_pretrained.pkl'
            update_params(net2, filename)

        # SEQ3 : MULTIPHASE
        elif pruning_opt == 2:
            print("\n==> Load Model file before Pruning.. : < MULTIPHASE >")
            # First De-Pruning
            if multi_phase_count == 0:
                print(" !!ERROR!! You should set M_count >= 1 to begin Multi-Phase process.")
                raise ValueError
            if multi_phase_count == 1:
                filename = folder_name + NN_name + '_' + pruning_method + '_pruned_' + str(Maskfile_id) + '%.pkl'
                update_params(net2, filename)

            elif multi_phase_count > 1:
                filename = folder_name + NN_name + '_' + pruning_method + '_multi_phased_' + str(Maskfile_id) + '%_' + str(multi_phase_count-1) + '#.pkl'
                update_params(net2, filename)

        print("\n==> Change format of Parameters to cuda Before test or training..")
        if torch.cuda.is_available():
            net2.cuda()
            if len(dev) != 1:
                net2 = nn.DataParallel(net2, device_ids = dev)

        print("\n==> Test for Loaded Network..")
        #test(net2, loader_test)

    # Pruning for SEQ2 : INITIAL PRUNING
        if multi_phase_opt == 0:
            print("\n==> Pruning Process Begins.. < INITIAL PRUNING > : You selected {} pruning ".format(pruning_method))
            if pruning_method == 'layer':
                masks = layer_prune(net2, param['pruning_perc'])
                net2.set_masks(masks, pruning_method)
                total_prune_rate = prune_rate(net2)
                data_save_pickle('./masks/' + NN_name + '_' + pruning_method + '_masks_' + str(int(total_prune_rate)) + '_pruned.pickle', masks)
                #pdb.set_trace()

            elif pruning_method == 'simd':
                masks = simd_prune(net2, param['pruning_perc'], RBS_MAX=param['rbs'], prune_fc = param['with_fc'], is_scattered = True)
                #save_metadata(masks, NN_name, multi_phase_opt, RBS_MAX = param['rbs'], is_encode = True, is_fc_inc = False)
                save_metadata_scattered(masks, NN_name, multi_phase_opt, RBS_MAX = param['rbs'], is_encode = True, is_fc_inc = param['with_fc'])
                if Meta_flag == 1:  raise ValueError
                net2.set_masks(masks, pruning_method, with_fc = param['with_fc'])
                total_prune_rate = prune_rate(net2)
                data_save_pickle('./masks/' + NN_name + '_' + pruning_method + '_masks_' + str(int(total_prune_rate)) + '_pruned.pickle', masks)

            elif pruning_method == 'gpu':
                masks = gpu_prune(net2, param['pruning_perc']/100, pruning_pattern = pattern)
                net2.set_masks(masks, pruning_method, with_fc = param['with_fc'])
                total_prune_rate = prune_rate(net2)
                data_save_pickle('./masks/' + NN_name+'_'+pruning_method+'_'+pattern+ '_masks_' + str(int(total_prune_rate)) + '_pruned.pickle', masks)
                #pdb.set_trace()

            elif pruning_method == 'simd_scalpel':
                masks = simd_prune_scalpel(net2, param['pruning_perc'], prune_fc = param['with_fc'], is_scattered = True)
                if Meta_flag == 1:  raise ValueError
                net2.set_masks(masks, pruning_method, with_fc = param['with_fc'])
                total_prune_rate = prune_rate(net2)
                data_save_pickle('./masks/' + NN_name + '_' + pruning_method + '_masks_' + str(int(total_prune_rate)) + '_pruned.pickle', masks)

            elif pruning_method == 'kc':
                pruned_weights_sequence, pruned_bias_sequence, pruned_net_info = kc_prune(net2, param['pruning_perc'], with_fc=param['with_fc'])#vgg?
                #pruned_weights_sequence, pruned_bias_sequence, pruned_net_info = kc_resnet_prune(net2, param['pruning_perc'], with_fc=param['with_fc'])#resnet?

                pprint(pruned_net_info)
                cfg_list = pruned_net_info['cfg_list']
                total_prune_rate = pruned_net_info['pruned_rate']
                print("<--- {}% parameters pruned --->".format(total_prune_rate))

                # Save info data to pickle
                filename = './masks/' + NN_name + '_' + pruning_method + '_kc_info_'+ str(int(total_prune_rate))+ '%_'+ 'pruned.pickle'
                data_save_pickle(filename, pruned_net_info)

                # Build net_kc & Initializing
                net2_kc = globals()[NN_name](cfg_list, with_fc=param['with_fc'])
                print(net2_kc)
                KC_initialize(net2_kc, pruned_weights_sequence, pruned_bias_sequence)
                #pdb.set_trace()

                if torch.cuda.is_available():
                    net2_kc.cuda()
                    if len(dev) != 1:
                        net2_kc = nn.DataParallel(net2, device_ids = dev)

            else:
                print(" !!ERROR!! You Selected Wrong Pruning Method! {}x!", pruning_method)
                raise ValueError

    # Pruning for SEQ3 : MULTI_PHASE
        elif multi_phase_opt == 1:
            print("\n==> MultiPhase Process Begins.. < MULTIPHASE > : You selected {} pruning".format(pruning_method))
            if pruning_method == 'layer':
                # Load Masks to make gradient_masks
                if multi_phase_count == 1:
                    masks = data_load_pickle('./masks/' + NN_name+ '_' + pruning_method + '_masks_' +str(Maskfile_id) + '_pruned.pickle')
                else:
                    masks = data_load_pickle('./masks/' + NN_name+ '_' + pruning_method + '_masks_' +str(Maskfile_id) + '_multi_phased_' + str(multi_phase_count-1) + '#.pickle')

                # grad_masks Flipping and Generation
                grad_masks = gen_grad_masks(masks, param['de-pruning_rate'], pruning_method, RBS_MAX = param['rbs'])

                # net2 setting mask for inference usage
                print(" * Generating Inference mask in MultiPhase option")
                inference_masks = gen_inf_masks(masks, grad_masks)

                net2.set_masks(inference_masks, pruning_method)
                total_prune_rate = prune_rate(net2)
                print("<--- {}% parameters added --->".format(param['de-pruning_rate']*100))

            elif pruning_method == 'simd':
                # Load golden_model : To get initial parameters. Always from original Network. (Always 100%)
                golden_model = globals()[NN_name]()
                update_params(golden_model, folder_name + NN_name + '_pretrained.pkl')

                # Load net_from : To initialize
                net_from = globals()[NN_name]()
                update_params(net_from, filename)
                if torch.cuda.is_available():
                    net_from.cuda()

                # Load Masks to make gradient_masks
                if multi_phase_count == 1:
                    masks = data_load_pickle('./masks/' + NN_name+ '_' + pruning_method + '_masks_' +str(Maskfile_id) + '_pruned.pickle')
                elif multi_phase_count > 1:
                    masks = data_load_pickle('./masks/' + NN_name+ '_' + pruning_method + '_masks_' +str(Maskfile_id) + '_multi_phased_' + \
                        str(multi_phase_count-1)+'#.pickle')
                else:
                    print("m_count should be larger than 1 : {}".format(multi_phase_count))
                    raise ValueError

                inference_masks = simd_deprune(masks, golden_model, param['de-pruning_rate']*100, with_fc=param['with_fc'])
                grad_masks = [(inference_masks[ind]-masks[ind]).cuda() for ind in range(len(masks))]
                # grad_masks Flipping and Generation

                #grad_masks = gen_grad_masks(masks, param['de-pruning_rate'], pruning_method, RBS_MAX = param['rbs'], is_scattered = True)

                # net2 setting mask for inference usage
                #print(" * Generating Inference mask in MultiPhase option")
                #inference_masks = gen_inf_masks(masks, grad_masks)
                #TODO Save Automatically
                #save_metadata(inference_masks, NN_name, multi_phase_opt, RBS_MAX=param['rbs'])
                #save_metadata_scattered(masks, NN_name, multi_phase_opt, RBS_MAX = param['rbs'], is_encode = True, is_fc_inc = param['with_fc'])
                if Meta_flag == 1:  raise ValueError
                net2.set_masks(inference_masks, pruning_method)

                print(" * Weight Initialize Process Begins....")
                net2 = weight_initialize(net2, golden_model, grad_masks, STD = 1e-10)
                total_prune_rate = prune_rate(net2)
                print("<--- {}% parameters added --->".format(param['de-pruning_rate']*100))

            elif pruning_method == 'kc':
                """
                    Multi_Phase Process for KC
                        - pruned_net_info : pruned_rate, cfg_list, pruned_shape_info, pruned_weights_sequence, pruned_bias_sequence
                        - Multi_phase_kc -> cfg_list(Model define), Mask
                """
                upper_pruned_net_info = data_load_pickle('./masks/' + NN_name + '_kc_kc_info_' + str(int(dP_rate)) + '%_pruned.pickle')
                pprint(upper_pruned_net_info)
                if dP_rate == 0:
                    upper_model = globals()[NN_name]()
                    update_params(upper_model, folder_name + NN_name +'_pretrained.pkl')
                elif dP_rate != 0:
                    upper_model = globals()[NN_name](cfg_kc=upper_pruned_net_info['cfg_list'], with_fc=param['with_fc'])
                    update_params(upper_model, folder_name+NN_name +'_kc_pruned_'+str(int(dP_rate))+'%.pkl')

                #---------------------------------------------------------------------
                masks, pruned_weights_sequence, pruned_bias_sequence,  pruned_net_info = KC_de_pruning(net2, pruned_net_info,\
                        param['de-pruning_rate'], upper_model, upper_pruned_net_info,  with_fc=param['with_fc'])
                pprint(pruned_net_info)

                # For Temporary Simd metadata Generation
                if Meta_flag == 1:
                    masks_temp = simd_prune(net2, 0, RBS_MAX = param['rbs'])
                    save_metadata_scattered(masks_temp, NN_name, multi_phase_opt, RBS_MAX=param['rbs'], is_encode = True)
                    raise ValueError

                # Initialize Multi-phased Network
                net2_kc = globals()[NN_name](cfg_kc=pruned_net_info['cfg_list'], with_fc=param['with_fc'])
                net2_kc = KC_initialize(net2_kc, pruned_weights_sequence, pruned_bias_sequence)
                grad_masks = gen_grad_masks(masks, param['de-pruning_rate'], pruning_method)

                # FIXME Is this prune_ratio represent only convolution layer??
                total_prune_rate = pruned_net_info['pruned_rate']
                print("<--- {}% parameters added! --->".format(param['de-pruning_rate']))

                if torch.cuda.is_available():
                    net2_kc.cuda()
                    if len(dev) != 1:
                        net2_kc = nn.DataParallel(net2_kc, device_ids = dev)

                # Save info data to pickle
                filename = './masks/' + NN_name + '_' + pruning_method + '_kc_info_'+str(Maskfile_id)+'-'+ str(int(total_prune_rate)) +\
                            '_multi_phased_'+str(multi_phase_count)+'#.pickle'
                data_save_pickle(filename, pruned_net_info)

            else:
                print(" !!ERROR!! You Selected Wrong Pruning Method! {}x!", pruning_method)
                raise ValueError

        print("\n==> Test after pruning")
        if torch.cuda.is_available():
            net2.cuda()
            if len(dev) != 1:
                net2 = nn.DataParallel(net2, device_ids = dev)
        if pruning_method == 'kc':
            test(net2_kc, loader_test)
        else:   test(net2, loader_test)

    # Retraining : Optimizer Setup
    if retrain_opt == 1:
        criterion2 = nn.CrossEntropyLoss()
        #optimizer2 = torch.optim.Adam(net2.parameters(), lr=param['learning_rate'],betas=param['betas'],eps=param['eps'],weight_decay=param['weight_decay']) 
        if train_opt == 0 and multi_phase_opt == 0:
            if pruning_method == 'kc':
                optimizer2 = torch.optim.SGD(net2_kc.parameters(), lr=param['learning_rate'],
                                            weight_decay=param['weight_decay'],
                                            momentum=param['momentum'])
            else:
                optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],
                                            weight_decay=param['weight_decay'],
                                            momentum=param['momentum'])
        elif train_opt == 0 and multi_phase_opt == 1:
            if pruning_method == 'kc':
                if param['MaskTrain_flag']:
                    optimizer2 = mySGD(net2_kc.parameters(), lr=param['learning_rate'],\
                                                weight_decay=param['weight_decay'],\
                                                momentum=param['momentum'])
                else:
                    optimizer2 = torch.optim.SGD(net2_kc.parameters(), lr=param['learning_rate'],\
                                                weight_decay=param['weight_decay'],\
                                                momentum=param['momentum'])
            elif pruning_method == 'layer':
                optimizer2 = mySGD(net2.parameters(), lr=param['learning_rate'],\
                                            weight_decay=param['weight_decay'],\
                                            momentum=param['momentum'])

            else:
                optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],\
                                            weight_decay=param['weight_decay'],\
                                            momentum=param['momentum'])
                #optimizer2 = mySGD(net2.parameters(), lr=param['learning_rate'],\
                #                            weight_decay=param['weight_decay'],\
                #                            momentum=param['momentum'])
        elif train_opt == 1:
            optimizer2 = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],
                                        weight_decay=param['weight_decay'],
                                        momentum=param['momentum'])

        # lr_scheduler : StepLR
        if param['LR_step'] == 1: scheduler2 = StepLR(optimizer = optimizer2, step_size = param['LR_step'][0], gamma = param['gamma'])
        else: scheduler2 = MultiStepLR(optimizer = optimizer2, gamma = param['gamma'], milestones=param['LR_step'])

        # lr_sheduler : ReduceLROnPlateau
        #scheduler2 = ReduceLROnPlateau(optimizer = optimizer,verbose = True, factor = 0.1)

        # Change format to CUDA
        print("\n==> Change format of Parameters to cuda Before test or training..")
        if torch.cuda.is_available():
            net2.cuda()
            if len(dev) != 1:
                net2 = nn.DataParallel(net2, device_ids = dev)

        if multi_phase_opt is 0:
            # Visualization Setup
            if vis_opt:
                viz = vis_class(title = vis_title_gen(NN_name, "Pruning_Retrain", text))
            else: viz = None
            if pruning_method == 'kc':
                train_time, nowtime_text =\
                train(net2_kc, criterion2, optimizer2, scheduler2, param, loader_train, NN_name, loader_test, viz=viz, last_epoch = last_epoch, pruning_method = pruning_method, loader_test = loader_test)
            else:
                train_time, nowtime_text =\
                train(net2, criterion2, optimizer2, scheduler2, param, loader_train, NN_name, loader_test, viz=viz, last_epoch = last_epoch, loader_test = loader_test)

        elif multi_phase_opt is 1:
            # Visualization Setup
            if vis_opt:
                viz = vis_class(title = vis_title_gen(NN_name, "MultiPhase_Retrain", text))
            else: viz = None

            if pruning_method == 'kc':
                #TODO Unlock this Comment
                if param['MaskTrain_flag']:
                    print("Training For Mask training\n")
                    train_time, nowtime_text =\
                    train(net2_kc, criterion2, optimizer2, scheduler2, param, loader_train, NN_name,\
                          loader_test, grad_masks, multi_phase_opt, viz = viz, last_epoch = last_epoch, pruning_method = pruning_method,\
                          with_fc = param['with_fc'], loader_test = loader_test)
                else:
                    print("Training For Copy-paste training\n")
                    optimizer_temp = torch.optim.SGD(net2.parameters(), lr=param['learning_rate'],
                                                weight_decay=param['weight_decay'],
                                                momentum=param['momentum'])
                    if param['LR_step'] == 1: scheduler_temp = StepLR(optimizer = optimizer_temp, step_size = param['LR_step'][0], gamma = gamma['gamma'])
                    else: scheduler_temp = MultiStepLR(optimizer = optimizer_temp, gamma = param['gamma'], milestones=param['LR_step'])

                    train_time, nowtime_text =\
                    cp_train(net2, net2_kc, criterion2, optimizer_temp, optimizer2, scheduler_temp,  scheduler2, param, loader_train, NN_name,\
                          pruning_method, loader_test, multi_phase_opt, viz = viz, last_epoch = last_epoch, loader_test = loader_test)

            elif pruning_method == 'layer':
                train_time, nowtime_text =\
                train(net2, criterion2, optimizer2, scheduler2, param, loader_train, NN_name,\
                      loader_test, grad_masks, multi_phase_opt, viz = viz, last_epoch = last_epoch, pruning_method = pruning_method,\
                      with_fc=param['with_fc'], loader_test = loader_test)

            else:
                #train_time, nowtime_text =\
                #train(net2, criterion2, optimizer2, scheduler2, param, loader_train, NN_name,\
                #      loader_test, grad_masks, multi_phase_opt, viz = viz, last_epoch = last_epoch, pruning_method = pruning_method,\
                #      with_fc=param['with_fc'], loader_test = loader_test)

                optimizer_temp = torch.optim.SGD(net_from.parameters(), lr=param['learning_rate'],
                                            weight_decay=param['weight_decay'],
                                            momentum=param['momentum'])
                if len(param['LR_step']) == 1: scheduler_temp = StepLR(optimizer = optimizer_temp, step_size = param['LR_step'][0], gamma =gamma['gamma'])
                else: scheduler_temp = MultiStepLR(optimizer = optimizer_temp, gamma = param['gamma'], milestones=param['LR_step'])

                train_time, nowtime_text =\
                cp_simd_train(net_from, net2, criterion2, optimizer_temp, optimizer2, scheduler_temp,  scheduler2, param, loader_train, NN_name,\
                        pruning_method, loader_test, multi_phase_opt, viz = viz, last_epoch = last_epoch, loader_test = loader_test, grad_masks=grad_masks)

    # Check accuracy and nonzeros weights in each layer After Retraining
        print("\n==> Test after retraining")
        if pruning_method == 'kc':  test(net2_kc, loader_test)
        else:   test(net2, loader_test, score_margin = score_margin)

        if pruning_method == 'kc':  total_prune_rate = pruned_net_info['pruned_rate']
        else:   total_prune_rate = prune_rate(net2)

    # Save and load the entire model After Retraining
        print("\n==> Save model & masks after retraining.. ")
        if pruning_opt == 1:
            filename = folder_name + NN_name + "_" + pruning_method + '_pruned_' + str(int(total_prune_rate)) +'%.pkl'
            if pruning_method == 'kc':
                torch.save(net2_kc.state_dict(), filename)
                log_to_txt(log_file_name, filename + '  ' + str(train_time) + '/' + nowtime_text)
            elif pruning_method == 'gpu':
                filename = folder_name + NN_name + "_" + pruning_method +'_'+pattern+ '_pruned_' + str(int(total_prune_rate)) +'%.pkl'
                torch.save(net2.state_dict(), filename)
                log_to_txt(log_file_name, filename + '  ' + str(train_time) + '/' + nowtime_text)
            else:
                torch.save(net2.state_dict(), filename)
                log_to_txt(log_file_name, filename + '  ' + str(train_time) + '/' + nowtime_text)
            print(' * Model save completed: '+ filename)

        elif pruning_opt == 2:
            filename = folder_name + NN_name + "_" + pruning_method + '_multi_phased_' + \
                        str(Maskfile_id)+ '-' + str(int(total_prune_rate))+'%_' + str(multi_phase_count)+'#.pkl'
            if pruning_method == 'kc':
                torch.save(net2_kc.state_dict(), filename)
                log_to_txt(log_file_name, filename + '  ' + str(train_time) + '/' + nowtime_text )
            else:
                data_save_pickle('./masks/'+NN_name+'_'+pruning_method+'_masks_'\
                                 +str(Maskfile_id)+'-'+str(int(total_prune_rate))+'_multi_phased_'+str(multi_phase_count)+'#.pickle', inference_masks)
                torch.save(net2.state_dict(), filename)
                log_to_txt(log_file_name, filename + '  ' + str(train_time) + '/' + nowtime_text )
            print(' * Model save completed: ' + filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Multi-Phase Pruning & Retraining Framework')
    parser.add_argument('--T',
                        type = int,
                        default = 0,
                        help = 'Initial Train Flag Option'
                        )
    parser.add_argument('--P',
                        type = int,
                        default = 0,
                        help = 'Pruning Flag Option'
                        )
    parser.add_argument('--M',
                        type = int,
                        default = 0,
                        help = 'MultiPhase Flag Option'
                        )
    parser.add_argument('--M_count',
                        type = int,
                        default = 0,
                        help = 'MultiPhase Counting Parameter'
                        )
    parser.add_argument('--Method',
                        type = str,
                        default = 'layer',
                        help = 'Pruning Method Select Flag'
                        )
    parser.add_argument('--R',
                        type = int,
                        default = 0,
                        help = 'Retraining Flag Option'
                        )
    parser.add_argument('--Vis',
                        type = str2bool,
                        default = 'f',
                        help = 'Visualization On / Off '
                        )
    parser.add_argument('--Model',
                        type = str,
                        default = None,
                        help = 'Model name'
                        )
    parser.add_argument('--T_Epoch',
                        type = int,
                        default = 1000,
                        help = 'Setting Parameter for Training Epoch'
                        )
    parser.add_argument('--L_Epoch',
                        type = int,
                        default = 0,
                        help = 'Last epoch of selected model'
                        )
    parser.add_argument('--P_rate',
                        type = float,
                        default = 90.0,
                        help = 'Pruning rate'
                        )
    parser.add_argument('--Dev',
                        type = int,
                        nargs = '+',
                        default = [3],
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
                        type = float,
                        default = 0,
                        help = 'de-Pruning rate'
                        )
    parser.add_argument('--Test',
                        type = str,
                        default = None,
                        help = 'Selected Model Test'
                        )
    parser.add_argument('--Mask',
                        type = str,
                        default = None,
                        help = 'To loading Mask pickle file'
                        )
    parser.add_argument('--Text',
                        type = str,
                        default = "",
                        help = 'Visdom Text Indexing'
                        )
    parser.add_argument('--Meta',
                        type = int,
                        default = 0,
                        help = 'Generate Metadata in each pruning'
                        )
    parser.add_argument('--Dataset',
                        type = int,
                        default = 10,
                        help = 'CIFAR 10 : type 10, CIFAR 100: type 100'
                        )
    parser.add_argument('--LR_step',
                        type = int,
                        nargs = '+',
                        default = [50],
                        help = 'Scheduler Learning Rate Decay Step Size'
                        )
    parser.add_argument('--Weight_decay',
                        type = float,
                        default = 1e-4,
                        help = 'Optimizer Weight Decay Term'
                        )
    parser.add_argument('--RBS',
                        type = int,
                        default = 3008,
                        help = 'Size of Reduction block'
                        )
    parser.add_argument('--with_fc',
                        type = str2bool,
                        default = 'f',
                        help = 'Prune FC Layer with CONV layer'
                        )
    parser.add_argument('--score_margin',
                        type = str2bool,
                        default = 'f',
                        help = 'Measure Score margin from selected epoch'
                        )
    parser.add_argument('--dropout_rate',
                        type = float,
                        default = 0.5,
                        help = 'Drop out rate'
                        )
    parser.add_argument('--momentum',
                        type = float,
                        default = 0,
                        help = 'momentum'
                        )
    parser.add_argument('--gamma',
                        type = float,
                        default = 0.1,
                        help = 'lr_step gamma'
                        )
    parser.add_argument('--MaskTrain',
                        type = str2bool,
                        default = 'f',
                        help = 'Training Method with Gradient Mask Method'
                        )
    parser.add_argument('--pattern',
                        type = str,
                        default = None,
                        help = 'Pruning Pattern for GPU Pruning. ex) micro, block, naive'
                        )
    args = parser.parse_args()
    main(args.momentum, args.T, args.P, args.Method,  args.R, args.M, args.M_count, args.Vis, args.Model, args.L_Epoch, args.T_Epoch, args.P_rate, args.Dev,\
        args.Train_Batch, args.Test_Batch, args.lr, args.dP_rate, args.Test, args.Mask, args.Text, args.Meta, args.Dataset, \
        args.LR_step, args.Weight_decay, args.RBS, args.with_fc, args.score_margin, args.dropout_rate, args.gamma, args.MaskTrain, args.pattern)
