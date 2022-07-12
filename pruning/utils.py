import visdom
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#from . import lr_scheduler
import random
import pickle
import gzip
import copy
import time
import tarfile
import argparse
import pdb

class vis_class():
    """Visualize loss and Acc"""
    def __init__(self, title):
        self.vis = visdom.Visdom()
        self.loss = self.vis.line(X = np.array([0]), Y = np.array([0]), opts = dict(title = title + "Loss"))
        self.acc = self.vis.line(X = np.array([0]), Y = np.array([0]), opts = dict(title = title + "Acc"))
        self.lr = self.vis.line(X = np.array([0]) , Y = np.array([0]), opts = dict(title = title + "LR"))

    def update_graph(self, mode, epoch, item, name=None):
        if mode == 'l':
            self.vis.line(X = np.array([epoch]), Y = np.array([item]), win = self.loss, update = "append", name = name)
        elif mode == 'a':
            self.vis.line(X = np.array([epoch]), Y = np.array([item]) , win = self.acc, update = "append", name = name)
        elif mode == 'lr':
            self.vis.line(X = np.array([epoch]), Y = np.array([item]), win=self.lr, update = "append", name = name)
        else:
            print("Should give mode {} or {} or {} which means ""loss"" and ""accuracy""".format('l', 'a', 'lr'))

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def paste_params(net_from, net_to, pruning_method, grad_masks = None):
    # Check identity of selected network from & to
    if count_specific_layer(net_from, 'Linear') != count_specific_layer(net_to, 'Linear') or \
            count_specific_layer(net_from, 'Conv') != count_specific_layer(net_to, 'Conv'):
        print("\nCan't Copy parameters with two different Network!\n")
        raise ValueError

    # For KC De-pruning Copy-paste
    if grad_masks == None and pruning_method == 'kc':
        """
            KC de-pruning doesn't need grad_masks, Just Copy-paste
        """
        # Define Iterator of net_from & to
        iter_from = net_from.named_parameters()
        iter_to = net_to.named_parameters()
        for name_from, param_from in iter_from:
            name_to, param_to = next(iter_to)

            shape_to = param_to.shape
            shape_from = param_from.shape
            if len(shape_from) == 4: # CONV
                if shape_from[0]+shape_from[1] < shape_to[0]+shape_to[1]:
                    param_to.data[:shape_from[0],:shape_from[1],:,:].copy_(param_from.data)
                else:
                    param_to.data.copy_(param_from.data[:shape_to[0],:shape_to[1],:,:])

            elif len(shape_from) == 2: # FIX for FC
                if shape_from[1] < shape_to[1]:
                    param_to.data[:shape_from[0],:shape_from[1]].copy_(param_from.data)
                else:
                    #raise ValueError # TODO Is Error????? #jjg
                    param_to.data.copy_(param_from.data[:,:shape_to[1]])

            #elif len(shape_from) == 1:# and not('classifier' in name_from): # Batch or Bias Except in classifier's bias
                #if shape_from[0] < shape_to[0]:
                #    param_to.data[:shape_from[0]].copy_(param_from.data)
                #else:
                #    param_to.data.copy_(param_from.data[:shape_to[0]])

    elif grad_masks != None and pruning_method == 'simd':
        """
            SIMD de-pruning need gradient masks
        """
        cnt = 0
        iter_from, iter_to = net_from.modules(), net_to.modules()
        for layer_ind, module_from in enumerate(iter_from):
            module_to = next(iter_to)

            if type(module_from).__name__ == 'MaskedConv2d':
                grad_mask = grad_masks[cnt].cpu().data; cnt += 1
                if torch.cuda.is_available():
                    grad_mask = grad_mask.cuda()
                # Copy the fixed region = module_to(Trained params + module_from (Fixed params)
                module_to.weight.data.copy_(torch.add(torch.mul(module_to.weight.data, grad_mask), module_from.weight.data))

            elif type(module_from).__name__ == 'MaskedLinear':
                grad_mask = grad_masks[cnt].cpu().data; cnt += 1
                if torch.cuda.is_available():
                    grad_mask = grad_mask.cuda()
                # Copy the fixed region = module_to(Trained params + module_from (Fixed params)
                module_to.weight.data.copy_(torch.add(torch.mul(module_to.weight.data, grad_mask), module_from.weight.data))

    else:
        print("Put Wrong pruning_method = {}!".format(pruning_method))
        raise ValueError

    return

def print_specific_layer(net, layer_name, flag, layer_ind=0):
    """
    Print specific layer's flag part of network
    Print 'First' layer of layer_name
    layer_name : 'Linear' or 'Conv'
    flag : 'grad' or 'grad_mask' or 'param'
    """
    np.set_printoptions(threshold=1000000)
    if layer_name == 'batchnorm':
        layer_cnt = 0
        for m in net.modules():
            if type(m).__name__ == 'BatchNorm2d':
                if layer_cnt == layer_ind:
                    print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                    if flag == 'weight': print(m.weight.cpu().data.numpy())
                    elif flag == 'bias': print(m.bias.cpu().data.numpy())
                    elif flag == 'mean': print(m.running_mean.cpu().data.numpy())
                    elif flag == 'var': print(m.running_var.cpu().data.numpy())
                    break
                else : layer_cnt += 1
    elif flag == 'grad':
        if layer_name == 'Linear':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedLinear':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        m.get_current_grad()
                        break
                    else : layer_cnt += 1
        elif layer_name == 'Conv':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedConv2d':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        m.get_current_grad()
                        break
                    else : layer_cnt += 1
    elif flag == 'grad_mask':
        if layer_name == 'Linear':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedLinear':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        m.get_grad_mask()
                        break
                    else : layer_cnt += 1
        elif layer_name == 'Conv':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedConv2d':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        m.get_grad_mask()
                        break
                    else : layer_cnt += 1

    elif flag == 'param':
        if layer_name == 'Linear':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedLinear':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        print(m.weight.cpu().data.shape)
                        print(m.weight.cpu().data.numpy()[0].reshape(-1))
                        break
                    else : layer_cnt += 1

        elif layer_name == 'Conv':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedConv2d':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        print(m.weight.cpu().data.shape)
                        temp_from = m.weight[0].reshape(-1).cpu().detach().numpy()
                        #temp_to = m.weight[36:].reshape(-1).cpu().detach().numpy()
                        print(temp_from)
                        break
                    else : layer_cnt += 1

    elif flag == 'bias':
        if layer_name == 'Linear':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedLinear':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        print(m.bias.cpu().data.numpy())
                        break
                    else : layer_cnt += 1
        elif layer_name == 'Conv':
            layer_cnt = 0
            for m in net.modules():
                if type(m).__name__ == 'MaskedConv2d':
                    if layer_cnt == layer_ind:
                        print("--- Current {} of {}'th {} layer ---".format(flag, str(layer_cnt), layer_name))
                        print(m.bias.cpu().data.numpy())
                        break
                    else : layer_cnt += 1


def weight_initialize(model, super_model, grad_masks, STD = 1e-10):
    # Incremental Initialization : Not use STD 
    cnt = 0
    model_gen, super_model_gen = model.modules(), super_model.modules()

    print(" * Selected Incremental Initialization")
    for layer_ind, module in enumerate(model_gen):
        super_module = next(super_model_gen)
        if type(module).__name__ == 'MaskedConv2d':
            print(str(layer_ind) + 'th Conv layer parameter initializing....')
            if module.weight.shape != super_module.weight.shape:
                print("!!ERROR!! {} != {}".format(module.weight.shape, super_module.weight.shape))
                raise ValueError
            grad_mask = grad_masks[cnt].cpu().data; cnt += 1 #FloatTensor

            init_weight = torch.mul(super_module.weight.data, grad_mask)
            if torch.cuda.is_available():
                init_weight = init_weight.cuda()
            module.weight.data.copy_(torch.add(module.weight.data,init_weight))

        elif type(module).__name__ == 'MaskedLinear': #TODO With fc decision??
            print(str(layer_ind) + 'th FC layer parameter initializing....')
            if module.weight.shape != super_module.weight.shape:
                print("!!ERROR!! {} != {}".format(module.weight.shape, super_module.weight.shape))
                raise ValueError
            grad_mask = grad_masks[cnt].cpu().data; cnt += 1 #FloatTensor

            init_weight = torch.mul(super_module.weight.data, grad_mask)
            if torch.cuda.is_available():
                init_weight = init_weight.cuda()

            module.weight.data.copy_(torch.add(module.weight.data,init_weight))

    """
    # Temporary commented : Random-Normal Initialization
    cnt = 0
    print("* Selected Random Normal Initialization")
    for layer_ind, params in enumerate(model.parameters()):
        if len(params.data.size()) == 4:
            print(str(layer_ind) + 'th Conv layer parameter initializing....')
            layer_shape = params.shape
            rand_normal = np.random.normal(0, STD, layer_shape[0]*layer_shape[1]*layer_shape[2]*layer_shape[3]).reshape(layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3])
            #if cnt == 1: print("     Before Initialize\n", params.cpu().data.numpy().reshape(-1)[:576*2])

            grad_mask_numpy = grad_masks[cnt].cpu().data.numpy(); cnt += 1
            rand_normal = torch.from_numpy(np.multiply(rand_normal, grad_mask_numpy)).type(torch.FloatTensor)
            if torch.cuda.is_available():
                 rand_normal = rand_normal.cuda()
            params.data.copy_(torch.add(rand_normal, params.data))

            #if cnt == 1: print("     After Initialize\n", params.cpu().data.numpy()[-1].reshape(-1))

        elif len(params.data.size()) == 2:
            print(str(layer_ind) + 'th layer parameter initializing....')
            layer_shape = params.shape
            rand_normal = np.random.normal(0, STD, layer_shape[0]*layer_shape[1]).reshape(layer_shape[0], layer_shape[1])
            #if cnt == 1: print("     Before Initialize\n", params.cpu().data.numpy().reshape(-1)[:576*2])

            grad_mask_numpy = grad_masks[cnt].cpu().data.numpy(); cnt += 1
            rand_normal = torch.from_numpy(np.multiply(rand_normal, grad_mask_numpy)).type(torch.FloatTensor)
            if torch.cuda.is_available():
                 rand_normal = rand_normal.cuda()
            params.data.copy_(torch.add(rand_normal, params.data))
    """
    return model

def gen_inf_masks(grad_masks, flipped_grad_masks):
    inference_masks = []
    print("Generating inference Mask Generation Process Begins___________________________________")
    for layer_num, mask in enumerate(grad_masks):
        if torch.cuda.is_available():
            mask = mask.cuda()
        inference_mask = torch.add(mask, flipped_grad_masks[layer_num])
        print("     Mask for Layer %d Complete" % layer_num)
        inference_masks.append(inference_mask)
    return inference_masks

def test(model, loader, score_margin = False, tag="test"):
    model.eval()
    cnt, num_correct, num_samples = 0, 0, len(loader.dataset)
    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)

        #y_var = to_var(y.long())
        #loss = loss_fn(scores, y_var)

        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        cnt += 1
        if score_margin: break
    if score_margin:
        softmax_score = F.softmax(scores, dim=1)
        print("scores_after_softmax!!!\n",softmax_score)
        print("score_margin!!\n", softmax_score.sort(1)[0][0][-1] - softmax_score.sort(1)[0][0][-2])
    acc = float(num_correct) / num_samples
    print(' <--- '+tag+' accuracy : {:.2f}% ({}/{}) --->'.format(
        100.*acc,
        num_correct,
        num_samples))
    return acc

def _gen_is_flipping_list(flipped_mask, de_prune_rate,  pruning_method, layer_shape,  RBS_MAX = 256, is_scattered = True):
    # Flipped Mask is Not FLIPPED YET!!!!!
    # Temp
    np.set_printoptions(threshold = 100000)

    # Initialization
    zero_cnt = np.count_nonzero(flipped_mask.cpu().data.numpy()==0)
    is_flipping_list = []

    # Layer Pruning 
    if pruning_method == 'layer' or (pruning_method == 'simd' and is_scattered == True):
        for i in range(int(zero_cnt*de_prune_rate)):
            is_flipping_list.append(True)
        for i in range(zero_cnt - int(zero_cnt*de_prune_rate)):
            is_flipping_list.append(False)
        random.shuffle(is_flipping_list)

    # Simd Pruning
    elif pruning_method == 'simd':
        # Extract Shpae of Layer
        num_filter = layer_shape[0]
        filter_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        print("     * Generate Gradient_Mask for Conv : shape ({}, {}, {}, {})".format(layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3]))
        print("        - Zero Parameter (Count, Percent) : ({}#, {:.2f}%)".format(zero_cnt, zero_cnt / (num_filter * filter_size) * 100 ))

        # Set Reduction block size
        if filter_size < RBS_MAX:
            reduction_block_size = filter_size
            num_reduction_block = 1
        else:
            reduction_block_size = RBS_MAX
            num_reduction_block = filter_size // reduction_block_size

        # Make is_flipping_list
        for filter_ind in range(num_filter):
            for reduction_block_ind in range(num_reduction_block):
                reduction_block_start = reduction_block_ind * reduction_block_size
                cnt = 0
                for ind in range(reduction_block_size):
                    if flipped_mask[filter_ind*filter_size + reduction_block_start + ind] == 0:
                        cnt = reduction_block_size - ind
                        break
                for ind in range(cnt):
                    if ind < int(cnt*de_prune_rate):
                        is_flipping_list.append(True)
                    else:
                        is_flipping_list.append(False)

            if filter_size % reduction_block_size != 0:
                leftover_block_start = (reduction_block_ind + 1) * reduction_block_size
                leftover_block_size = filter_size - leftover_block_start
                cnt = 0
                for ind in range(leftover_block_size):
                    if flipped_mask[filter_ind * filter_size + leftover_block_start + ind] == 0:
                        cnt = leftover_block_size - ind
                        break
                for ind in range(cnt):
                    if ind < int(cnt*de_prune_rate):
                        is_flipping_list.append(True)
                    else:
                        is_flipping_list.append(False)

        zero_cnt = np.count_nonzero(np.array(is_flipping_list)==0)
        print("        - After De-Pruning, Zero Parameter (Count, Percent) : ({}#, {:.2f}%)".format(zero_cnt, zero_cnt / (num_filter * filter_size) * 100 ))

    else:
        print("ERROR! : Selected Wrong Pruning Method!")
        raise ValueError

    return is_flipping_list


def gen_grad_masks(masks, de_prune_rate, pruning_method, RBS_MAX = 256, is_scattered = True, pruned_net_info = None):
    """
        Masking gradient
        masks should flip : 0 -> 1, 1 -> 0
        During Iteration, model.parameters.grad should multiplied with masks
    """
    # Flipping Masks : Mask Generation
    print("Generating flipped Gradient Mask Generation Process Begins" + "="*30)
    flipped_masks = []
    conv_cnt = -1; fc_cnt = -1

    for layer_num, mask in enumerate(masks):
        # Zero count operation & Initialization
        flip_cnt = 0
        layer_shape = mask.shape
        if pruning_method == 'kc+simd':
            flipped_mask = copy.deepcopy(mask)
        else:
            flipped_mask = copy.deepcopy(mask.reshape(-1))

        if pruning_method == 'kc':
            for ind, data in enumerate(flipped_mask):
                if data == 1:
                    flipped_mask[ind] = 1 - data
                elif data == 0:
                    flipped_mask[ind] = 1 - data

        # SIMD pruing in kc+simd process (Not SIMD de-pruning!)
        elif pruning_method == 'kc+simd' and pruned_net_info != None:
            if len(layer_shape) == 2: #FC : TODO Node Pruning? / This is for only Single FC
                fc_cnt += 1
                num_sub_channel = pruned_net_info['sub_pruned_shape_info'][conv_cnt + fc_cnt]
                flipped_mask[:,:num_sub_channel] = 0

            elif len(layer_shape) == 4: #Conv
                conv_cnt += 1
                # Gen shape of masks
                upper_shape_num_filter, upper_shape_channel, upper_shape_width, upper_shape_height = layer_shape
                sub_shape_num_filter = pruned_net_info['sub_pruned_shape_info'][layer_num]
                print("Upper, sub shape", layer_shape, sub_shape_num_filter)
                if layer_num == 0:
                    flipped_mask[:sub_shape_num_filter,:,:,:] = 0
                else:
                    sub_shape_num_channel = pruned_net_info['sub_pruned_shape_info'][layer_num-1]
                    flipped_mask[:sub_shape_num_filter,:sub_shape_num_channel,:,:] = 0

        else:
            # Stochastic Select for Layer_pruning
            is_flipping_list = _gen_is_flipping_list(flipped_mask, de_prune_rate, pruning_method, layer_shape,  RBS_MAX = RBS_MAX, is_scattered = is_scattered)
            for ind, data in enumerate(flipped_mask):
                if data == 1:
                    flipped_mask[ind] = 1 - data
                elif data == 0:
                    if is_flipping_list[flip_cnt]:
                        flipped_mask[ind] = 1 - data
                    flip_cnt += 1

        #if layer_num == 1:
        #    print("len(is_flipping_list) for layer #{}".format(len(is_flipping_list)))
        #    print("is_flipping_list for layer #{} \n{}".format(layer_num, np.array(is_flipping_list).astype(float)))
        if torch.cuda.is_available():
            flipped_mask = flipped_mask.cuda()

        flipped_masks.append(flipped_mask.reshape(layer_shape))
    return flipped_masks

def set_grad_masks(model, flipped_masks, pruning_method, with_fc=False):
    cnt = 0
    if pruning_method == 'kc' or pruning_method == 'simd':
        for m in model.modules():
            if type(m).__name__ ==  'MaskedConv2d':
                m.set_grad_mask(flipped_masks[cnt])
                cnt += 1
            if type(m).__name__ ==  'MaskedLinear' and with_fc:
                m.set_grad_mask(flipped_masks[cnt])
                cnt += 1

    elif pruning_method == 'layer':
        for m in model.modules():
            if type(m).__name__ ==  'MaskedConv2d':
                m.set_grad_mask(flipped_masks[cnt])
                cnt += 1
            elif type(m).__name__ ==  'MaskedLinear':
                m.set_grad_mask(flipped_masks[cnt])
                cnt += 1


def count_specific_layer(net, layer_name):
    if layer_name == 'Linear':
        layer_cnt = 0
        for m in net.modules():
            if type(m).__name__ == 'MaskedLinear':
                layer_cnt +=1
        return layer_cnt

    elif layer_name == 'Conv':
        layer_cnt = 0
        for m in net.modules():
            if type(m).__name__ == 'MaskedConv2d':
                layer_cnt +=1
        return layer_cnt

    else:
        print("You should Select Specific Layer name!!")
        raise ValueError

def train(model, loss_fn, optimizer, scheduler, param, loader_train, NN_name, loader_val = None, grad_masks = None,\
          multi_phase_opt = 0, viz = None, last_epoch=0, pruning_method = 'layer', with_fc=False, loader_test=None):
    now_time = time.localtime()
    now_time_text = "%02d/%02d:(%02d:%02d)" % (now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min)
    train_ex_start = time.time()

    #print_specific_layer(model, 'Conv', 'param')
    #if multi_phase_opt == 1:
        #print("---------------------- init []-----------------------1111111111111111111111")
        #print_specific_layer(model, 'Linear', 'grad')
        #print_specific_layer(model, 'Linear', 'param')
        #print_specific_layer(model, 'Linear', 'bias')

    model.train()
    for epoch in range(param['num_epochs']):
        # Learning_rate Adjustment : Others
        scheduler.step()

        # Get Learning_rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # Training Per Batch : Default 500
        for t, (x, y) in enumerate(loader_train):
            x_var, y_var = to_var(x), to_var(y.long())
            scores = model(x_var)
            loss = loss_fn(scores, y_var)

            if (t + 1) <= 1:
                for x_test, y_test in loader_test:
                    x_var_test, y_var_test = to_var(x_test), to_var(y_test.long())
                    scores_test = model(x_var_test)
                    loss_test = loss_fn(scores_test, y_var_test)
                #print(' | EPOCH [%d / %d] : (t = %d) (train_loss = %.8f) (lr = %f) |' % (epoch + 1, param['num_epochs'], t + 1, loss.item(), lr))
                print(' | EPOCH [%d / %d] : (t = %d) (train_loss = %.8f)(test_loss = %.8f) (lr = %f) |' % (epoch + 1, param['num_epochs'], t + 1, loss.item(), loss_test.item(), lr))


            # For Temp : De-prun Check
            #if t < 1 and epoch > 0 :
                #print("---------------------- init [{}/{}]-----------------------".format(epoch, t))
                #print_specific_layer(model, 'Conv', 'grad')
                #print_specific_layer(model, 'Conv', 'param')
                #print_specific_layer(model, 'Linear', 'grad')
            optimizer.zero_grad()

            # For Temp : De-prun Check
            #if t < 3 and multi_phase_opt == 1:
                #print("---------------------- after zero_grad [{}/{}]-----------------------".format(epoch, t))
                #print_specific_layer(model, 'Conv', 'grad')
                #print_specific_layer(model, 'Conv', 'param')

            loss.backward()

            # For Temp : De-prun Check
            #if t < 1 and multi_phase_opt == 1:
            #    print("---------------------- after backward [{}/{}]-----------------------".format(epoch, t))
            #    print_specific_layer(model, 'Conv', 'grad')
            #    print_specific_layer(model, 'Conv', 'param')

            # Setting grad_masking Using Flipped masks
            #if multi_phase_opt == 1 and grad_masks != None:
            if grad_masks != None:
                set_grad_masks(model, grad_masks, pruning_method, with_fc=with_fc)

            # For Temp : De-prun Check
            #if t < 1 and multi_phase_opt == 1:
            #    print("---------------------- after set_grad_mask [{}/{}]-----------------------".format(epoch, t))
            #    print_specific_layer(model, 'Conv', 'grad')
            #    print_specific_layer(model, 'Conv', 'param')
                #print_specific_layer(model, 'Conv', 'grad_mask')
            #if multi_phase_opt == 1 and grad_masks != None:
            if grad_masks != None:
                optimizer.step(grad_masks)
            else:
                optimizer.step()
            #For Temp : De-prun Check
            #if epoch % 2 == 0 and (t + 1) % 100 == 0:
            #    print("---------------------- after step [{}/{}]-----------------------".format(epoch, t+1))
            #    print_specific_layer(model, 'Conv', 'grad')
            #    print_specific_layer(model, 'Conv', 'grad_mask')
            #    print_specific_layer(model, 'Conv', 'param')

        # Learning_rate Adjustment : ReduceOnPlateau
        #scheduler.step(loss.item())
        # Visualization
        if viz is not None:
            viz.update_graph('l', epoch, loss.item(), name='train_loss')
            viz.update_graph('l', epoch, loss_test.item(), name='test_loss')
            viz.update_graph('lr', epoch, lr, name='learning_rate')

        # Test the model
        if epoch % 2 == 0 and loader_val is not None:
            acc = test(model, loader_train, tag="train")
            acc_test = test(model, loader_test, tag="test")
            if viz is not None:
                viz.update_graph('a', epoch, acc, name='train_acc')
                viz.update_graph('a', epoch, acc_test, name='test_acc')

        # Save the model
        if epoch % 100 == 0 and epoch != 0:
            if multi_phase_opt == 0 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/' + NN_name+'_' + str(epoch+last_epoch)))
                torch.save(model.state_dict(), './checkpoint/'+NN_name+'_' + str(epoch+last_epoch) + '.pkl')
            elif multi_phase_opt == 1 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/'+NN_name+'_multi_phased_' + str(epoch+last_epoch)))
                torch.save(model.state_dict(), './checkpoint/'+NN_name+'_multi_phased_' + str(epoch+last_epoch) + '.pkl')

    train_ex_time = time.time() - train_ex_start
    return train_ex_time, now_time_text

def cp_train(model_from, model_to, loss_fn, optimizer_from, optimizer_to, scheduler_from, scheduler_to, param, loader_train, NN_name,\
          pruning_method, loader_val = None, multi_phase_opt = 0, viz = None, last_epoch=0, loader_test=None, grad_masks = None):
    print("Start Copy-paste training...\n")
    sub_epoch = 1
    now_time = time.localtime()
    now_time_text = "%02d/%02d:(%02d:%02d)" % (now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min)
    train_ex_start = time.time()

    #model_from.train()
    model_to.train()
    for epoch in range(param['num_epochs']):
        # Learning_rate Adjustment : Others
        scheduler_from.step()
        scheduler_to.step()

        # Get Learning_rate
        for param_group in optimizer_from.param_groups:
            lr = param_group['lr']

        # Training Per Batch: Model_to
        for t, (x, y) in enumerate(loader_train):

            x_var, y_var = to_var(x), to_var(y.long())
            scores = model_to(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) <= 1:
                for x_test, y_test in loader_test:
                    x_var_test, y_var_test = to_var(x_test), to_var(y_test.long())
                    scores_test = model_to(x_var_test)
                    loss_test = loss_fn(scores_test, y_var_test)
                print(' | EPOCH [%d / %d] for net_from : (t = %d) (train_loss = %.8f)(test_loss = %.8f) (lr = %f) |' % (epoch + 1, param['num_epochs'], t + 1, loss.item(), loss_test.item(), lr))

            optimizer_to.zero_grad()
            loss.backward()
            optimizer_to.step()


        #paste_params(model_to, model_from, pruning_method)
        paste_params(model_from, model_to, pruning_method)
        #print_specific_layer(model_to, 'Conv', 'param')

        """
        # Training Per Batch: Model_from
        for t, (x, y) in enumerate(loader_train):
            if t%5 == 0 :
                for x_test, y_test in loader_test:
                    x_var_test, y_var_test = to_var(x_test), to_var(y_test.long())
                    scores_test = model_from(x_var_test)
                    loss_test = loss_fn(scores_test, y_var_test)

            x_var, y_var = to_var(x), to_var(y.long())
            scores = model_from(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) <= 1:
                print(' | EPOCH [%d / %d] for net_to : (t = %d) (train_loss = %.8f)(test_loss = %.8f) (lr = %f) |' % (epoch + 1, param['num_epochs'], t + 1, loss.item(), loss_test.item(), lr))

            optimizer_from.zero_grad()
            loss.backward()
            optimizer_from.step()

        paste_params(model_from, model_to, pruning_method)
        """
        # Learning_rate Adjustment : ReduceOnPlateau
        #scheduler.step(loss.item())
        # Visualization
        if viz is not None:
            viz.update_graph('l', epoch, loss.item(), name = 'train_loss')
            viz.update_graph('l', epoch, loss_test.item(), name = 'test_loss')
            viz.update_graph('lr', epoch, lr, name = 'learning_rate')

        # Test the model
        if epoch % 2 == 0 and loader_val is not None:
            #acc = test(model_from, loader_train,tag="train")
            #acc_test = test(model_from, loader_test,tag="test")
            acc = test(model_to, loader_train,tag="train")
            acc_test = test(model_to, loader_test,tag="test")
            if viz is not None:
                viz.update_graph('a', epoch, acc, name='train_acc')
                viz.update_graph('a', epoch, acc_test, name='test_acc')

        # Save the model
        if epoch % 100 == 0 and epoch != 0:
            if multi_phase_opt == 0 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/' + NN_name+'_' + str(epoch+last_epoch)))
                torch.save(model_to.state_dict(), './checkpoint/'+NN_name+'_' + str(epoch+last_epoch) + '.pkl')
            elif multi_phase_opt == 1 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/'+NN_name+'_multi_phased_' + str(epoch+last_epoch)))
                torch.save(model_to.state_dict(), './checkpoint/'+NN_name+'_multi_phased_' + str(epoch+last_epoch) + '.pkl')

    train_ex_time = time.time() - train_ex_start
    return train_ex_time, now_time_text

def cp_simd_train(model_from, model_to, loss_fn, optimizer_from, optimizer_to, scheduler_from, scheduler_to, param, loader_train, NN_name,\
        pruning_method, loader_val = None, multi_phase_opt = 0, viz = None, last_epoch=0, loader_test=None, grad_masks = None):
    print("Start Copy-paste training for SIMD De-pruning...\n")
    sub_epoch = 1
    now_time = time.localtime()
    now_time_text = "%02d/%02d:(%02d:%02d)" % (now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min)
    train_ex_start = time.time()
    model_to.train()
    for epoch in range(param['num_epochs']):
        # Learning_rate Adjustment : Others
        scheduler_from.step()
        scheduler_to.step()

        # Get Learning_rate
        for param_group in optimizer_from.param_groups:
            lr = param_group['lr']

        # Training Per Batch: Model_to
        for t, (x, y) in enumerate(loader_train):

            x_var, y_var = to_var(x), to_var(y.long())
            scores = model_to(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) <= 1:
                for x_test, y_test in loader_test:
                    x_var_test, y_var_test = to_var(x_test), to_var(y_test.long())
                    scores_test = model_to(x_var_test)
                    loss_test = loss_fn(scores_test, y_var_test)
                print(' | EPOCH [%d / %d] for net_from : (t = %d) (train_loss = %.8f)(test_loss = %.8f) (lr = %f) |' % (epoch + 1, param['num_epochs'], t + 1, loss.item(), loss_test.item(), lr))

            optimizer_to.zero_grad()
            loss.backward()
            optimizer_to.step()


        paste_params(model_from, model_to, pruning_method, grad_masks = grad_masks)
        #print_specific_layer(model_to, 'Conv', 'param')
        # Learning_rate Adjustment : ReduceOnPlateau
        #scheduler.step(loss.item())
        # Visualization
        if viz is not None:
            viz.update_graph('l', epoch, loss.item(), name = 'train_loss')
            viz.update_graph('l', epoch, loss_test.item(), name = 'test_loss')
            viz.update_graph('lr', epoch, lr, name = 'learning_rate')

        # Test the model
        if epoch % 2 == 0 and loader_val is not None:
            #acc = test(model_from, loader_train,tag="train")
            #acc_test = test(model_from, loader_test,tag="test")
            acc = test(model_to, loader_train,tag="train")
            acc_test = test(model_to, loader_test,tag="test")
            if viz is not None:
                viz.update_graph('a', epoch, acc, name='train_acc')
                viz.update_graph('a', epoch, acc_test, name='test_acc')

        # Save the model
        if epoch % 100 == 0 and epoch != 0:
            if multi_phase_opt == 0 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/' + NN_name+'_' + str(epoch+last_epoch)))
                torch.save(model_to.state_dict(), './checkpoint/'+NN_name+'_' + str(epoch+last_epoch) + '.pkl')
            elif multi_phase_opt == 1 :
                print(" <--- Saving Network to {}.pkl --->".format('./models/'+NN_name+'_multi_phased_' + str(epoch+last_epoch)))
                torch.save(model_to.state_dict(), './checkpoint/'+NN_name+'_multi_phased_' + str(epoch+last_epoch) + '.pkl')

    train_ex_time = time.time() - train_ex_start
    return train_ex_time, now_time_text

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0
    layer_id = 0
    print("\n==> Ratio of Zero parameters from Network ..")
    for parameter in model.parameters():
        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print(" * Layer {} | {} layer | {:.2f}% parameters pruned | {} zeros {} nonzeros in {} parameters " \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        zero_param_this_layer,
                        param_this_layer-zero_param_this_layer,
                        param_this_layer
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}% | in {} parameters".format(pruning_perc,total_nb_param))
    return pruning_perc

def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix

def data_save_pickle(filename, data):
    print(" * Data Saving as pickle file.... ")
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(" * Save Successfully ! : {}".format(filename))

def data_load_pickle(filename):
    print(" * Mask Loading from pickle file.... ")
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    print(" * Load Successfully ! : {}".format(filename))
    return data

def vis_title_gen(NN_name, train_mode, text):
    sep = "_"
    if text is "":
        now_time = time.localtime()
        text = "%02d/%02d:(%02d:%02d)" % (now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min)
        vis_title = NN_name + sep +  train_mode + sep + text + sep
    else : vis_title = NN_name + sep + train_mode + sep + text + sep
    return vis_title

def log_to_txt(filename, data):
    print("Logging to {}",filename)
    with open(filename, 'a') as f:
        f.write(str(data) + '\n')

def _encode_binary2int(meta_buffer):
    # Get list of string, return list of string
    zero_cnt = 0
    meta_buffer_int = []
    for binary_str in meta_buffer:
        for ind, binary_bit in enumerate(binary_str):
            if ind == 0:
                if binary_bit == '0':
                    if binary_str[1:].find('1') == -1:
                        #int_str = str(len(binary_str)-1) + ','
                        int_str = ''
                        break
                    else: int_str = str(binary_str[1:].find('1') + 1) + ','
                elif binary_bit == '1':
                    int_str = '0,'

            if (ind+1) != len(binary_str) and binary_bit == '1':
                if binary_str[ind+1:].find('1')== -1:
                    #int_str += str(len(binary_str[ind+1:])) + ','
                    pass
                else: int_str += str(binary_str[ind+1:].find('1') + 1) + ','

        # Add size of binary_string
        binary_size = int_str.count(',')
        int_str = str(binary_size) + ',' + int_str

        meta_buffer_int.append(int_str[:-1])

    return meta_buffer_int

def _encode_binary2int_gpu2(binary_str):
    toggle_str = '10'; tog = 0
    int_str = ''
    sliced_str = binary_str
    end = 0
    # Get string, return string
    while(True):
        start = sliced_str.find(toggle_str[tog%2])
        end = sliced_str.find(toggle_str[(tog+1)%2])
        if(start == -1): # Not firstly found '1'
            int_str += '0,'
        elif(end == -1):
            if(toggle_str[tog%2] == '0'):
                break
            else:
                int_str += str(len(sliced_str)) + ','
                break
        elif(end-start<0):
            int_str += '0,'
        else:
            int_str += str(end-start) + ','
        sliced_str = sliced_str[end:]
        tog += 1

    # Add size of binary_string
    binary_size = int_str.count(',')
    int_str = str(binary_size) + ',' + int_str


    return int_str[:-1] #Return string

def _encode_binary2int_gpu(binary_str):
    # Get string, return string
    meta_buffer_int = []
    for ind, binary_bit in enumerate(binary_str):
        if ind == 0:
            if binary_bit == '0':
                if binary_str[1:].find('1') == -1:
                    #int_str = str(len(binary_str)-1) + ','
                    int_str = ''
                    break
                else: int_str = str(binary_str[1:].find('1') + 1) + ','
            elif binary_bit == '1':
                int_str = '0,'

        if (ind+1) != len(binary_str) and binary_bit == '1':
            if binary_str[ind+1:].find('1')== -1:
                #int_str += str(len(binary_str[ind+1:])) + ','
                pass
            else: int_str += str(binary_str[ind+1:].find('1') + 1) + ','

    # Add size of binary_string
    binary_size = int_str.count(',')
    int_str = str(binary_size) + ',' + int_str

    #meta_buffer_int.append(int_str[:-1])

    return int_str[:-1] #Return string

def save_metadata_gpu(masks, NN_name, pruned_layer, pruning_pattern, num_workgroup, num_conv):
    #pruned_layer_flag = [1,2,3]
    pruned_layer_flag = [e for e in range(num_conv)]
    num_microtile = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] #VGG16, Conv 13
    num_iter = [1] + [16 for e in range(num_conv-1)]

    for layer_ind, mask in enumerate(masks): #Make metadata per layer
        if layer_ind in pruned_layer_flag: # pruned_layer_flag
            num_macrotile = num_workgroup * num_microtile[layer_ind]
            # Save CONV layer's mask
            fname = './metadata/' + NN_name + 'conv' + str(layer_ind) + '.csv'
            num_filter, channel, width, height = mask.shape
            filter_size = channel * width * height
            data_2D = mask.reshape(num_filter, filter_size)

            meta_buffer = [] # Metadata buffer
            blocksize_buffer = []
            blockpoint_buffer = []
            if pruning_pattern == 'block' or layer_ind == 0:
                print("Saving Metadata for" + str(layer_ind) + "th block Conv Layer..")
                single_line_str = ''
                for filter_offset in range(0, num_filter, num_macrotile):
                    block_str = ''
                    for block_offset in range(0, filter_size, num_iter[layer_ind]):
                       block_data = data_2D[filter_offset, block_offset:block_offset+num_iter[layer_ind]].numpy().reshape(-1).astype(int).tolist()
                       block_str += str(block_data[0])
                    temp_str = ','+_encode_binary2int_gpu(block_str)
                    blocksize_buffer.append(temp_str.count(','))
                    single_line_str += temp_str
                meta_buffer.append(''.join(str(_) + ',' for _ in blocksize_buffer) + str(max(blocksize_buffer)))
                meta_buffer.append(single_line_str[1:])

            elif pruning_pattern == 'pattern':
                print("Saving Metadata for" + str(layer_ind) + "th {}Conv Layer..".format(pruning_pattern))
                pattern_list = [[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14], [3, 7, 11, 15]]
                single_line_str = ''
                for filter_offset in range(0, num_filter, num_workgroup*2):
                    for block_offset in range(0, filter_size, num_workgroup):
                        block_mask = data_2D[filter_offset, block_offset: block_offset+num_workgroup].numpy().reshape(-1).astype(int).tolist()
                        pat_ind = []
                        for ind, m in enumerate(block_mask):
                            if m==1: pat_ind.append(ind)
                        single_line_str += ',' + str(pattern_list.index(pat_ind))

                    blocksize_buffer.append(filter_size//num_workgroup)
                meta_buffer.append(''.join(str(_) + ',' for _ in blocksize_buffer) + str(max(blocksize_buffer)))
                meta_buffer.append(single_line_str[1:])

            elif pruning_pattern == 'micro':
                INT_SIZE = 32
                print("Saving Metadata for" + str(layer_ind) + "th {} Conv Layer..".format(pruning_pattern))
                num_block = filter_size // num_workgroup
                num_int = num_block//INT_SIZE if num_block%INT_SIZE ==0 else num_block//INT_SIZE+1
                single_line_str = ''
                for filter_offset in range(0, num_filter, num_macrotile):
                    workitem_str_list = []
                    for workitem_offset in range(num_workgroup):
                        workitem_data = data_2D[filter_offset][[e*num_workgroup+workitem_offset for e in range(num_block)]].numpy().reshape(-1).astype(int).tolist()
                        workitem_str_list.append(''.join(str(_) for _ in workitem_data))

                    workitem_int_list = []
                    num_one = workitem_str_list[0].count('1')
                    workitem_int_list.append(str(num_one))
                    workitem_int_list.append(str(num_int))
                    for workitem_str in workitem_str_list:
                        num_one = workitem_str.count('1')
                        for offset in range(0, num_block, 32):
                            int_str = workitem_str[offset:offset+32]
                            if len(int_str) != 32:
                                int_str += ''.join('0' for i in range(32-len(int_str)))
                            if(int(int_str, base=2) >= 4294967296): raise ValueError
                            workitem_int_list.append(str(int(int_str, base=2)))


                    blocksize_buffer.append(len(workitem_int_list))
                    single_line_str += ''.join(str(_) + ',' for _ in workitem_int_list)

                meta_buffer.append(''.join(str(_) + ',' for _ in blocksize_buffer) + str(max(blocksize_buffer)))
                meta_buffer.append(single_line_str[:-1])
            """
            elif pruning_pattern == 'micro':
                print("Saving Metadata for" + str(i) + "th {}Conv Layer..".format(pruning_pattern))
                single_line_str = ''
                for filter_offset in range(0, num_filter, num_workgroup*2):
                   block_str = ''
                   for block_offset in range(0, filter_size, num_workgroup):
                       block_data = data_2D[filter_offset, block_offset:block_offset+num_workgroup].numpy().reshape(-1).astype(int).tolist()
                       temp_str = ',' + _encode_binary2int_gpu2(''.join(str(_) for _ in block_data)) # String!
                       block_str += temp_str
                   blocksize_buffer.append(block_str.count(','))
                   single_line_str += ',' + block_str[1:]
                print("Blockcolumn size = ", blocksize_buffer)
                # Make blockpoint_buffer
                for ind, size in enumerate(blocksize_buffer):
                    if ind == 0:  blockpoint_buffer.append(0)
                    else:  blockpoint_buffer.append(blockpoint_buffer[ind-1] + blocksize_buffer[ind-1])
                for i in range(max(blocksize_buffer)-blocksize_buffer[-1]):
                    single_line_str += ',0'
                #meta_buffer.append(''.join(str(_) + ',' for _ in blocksize_buffer) + str(max(blocksize_buffer)))
                meta_buffer.append(''.join(str(_) + ',' for _ in blockpoint_buffer) + str(max(blocksize_buffer)))
                meta_buffer.append(single_line_str[1:])
            """

            np.savetxt(fname, np.array(meta_buffer), delimiter = '\n', fmt='%s')
    return

def save_metadata(masks, NN_name, multi_phase_opt, RBS_MAX = 256, is_encode = False, is_fc_inc = False):
    """
    Save Metadata of pruned network. Use when right-sided SIMD Pruning.
    """
    for i, layer in enumerate(masks):
        # Save FC layer's metadata
        if len(layer.shape) == 2 and is_fc_inc == True:
            meta_buffer = []
            print("Saving Metadata for" + str(i) + "th FC Layer..")
            num_class, num_input_channel = layer.shape

            if multi_phase_opt == 0:    fname = './metadata/' + NN_name + 'fc' + str(i) + '.csv'
            else:   fname = './metadata/' + NN_name + 'fc' + str(i) + '.csv'

            for class_ind, input_channel_set in enumerate(layer):
                cnt = ''
                for input_ind, input_channel_data in enumerate(input_channel_set):
                    if input_ind % 4 == 0:
                        if input_channel_data == 0:
                            cnt += '0'
                        else: cnt += '1'
                meta_buffer.append(cnt)

            # Metadata Count method
            if is_encode:
                np.savetxt(fname, np.array(_encode_binary2int(meta_buffer)), delimiter = '\n', fmt='%s')
            else:
                np.savetxt(fname, np.array(meta_buffer).astype(str), delimiter = '\n', fmt='%s')
            break # Finish when save metadata of  fully connected layer

        # Save CONV layer's metadata 
        print("Saving Metadata for" + str(i) + "th Conv Layer..")
        meta_buffer = []
        if multi_phase_opt == 0:    fname = './metadata/' + NN_name + 'conv' + str(i) + '.csv'
        else:   fname = './metadata/' + NN_name + 'conv' + str(i) + '.csv'

        num_filter, channel, width, height = layer.shape
        filter_size = channel * width * height
        temp_layer = layer.reshape(num_filter, filter_size)
        num_filter_divided = num_filter // 4
        print("num_filter_divided_4 and filter size = {}, {}".format(num_filter_divided, filter_size))

        # Set Reduction Block Size
        if filter_size < RBS_MAX:
            reduction_block_size = filter_size
            num_reduction_block = 1
        else:
            reduction_block_size = RBS_MAX
            num_reduction_block = filter_size // reduction_block_size

        print("num_reduction_block = {}".format(num_reduction_block))
        # Extract Metadata information from RBS
        for reduction_block_ind in range(num_reduction_block):
            reduction_block_start = reduction_block_ind * reduction_block_size
            for filter_ind_f in range(0, num_filter, 4):
                cnt = 0
                # filter size < RBS_MAX
                if filter_size < RBS_MAX:
                    for ind in range(reduction_block_size):
                        if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                            cnt += 1
                        else: break
                # filter size >= RBS_MAX
                else:
                    for ind in range(reduction_block_size):
                        if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                            cnt += 1
                        else: break
                meta_buffer.append(cnt)

        # Extract Metadata information from LeftOver
        if filter_size % reduction_block_size !=0:
            leftover_block_start = (reduction_block_ind+1) * reduction_block_size
            leftover_block_size = filter_size - leftover_block_start
            for filter_ind_f in range(0, num_filter, 4):
                cnt = 0
                for ind in range(leftover_block_size):
                    if temp_layer[filter_ind_f][leftover_block_start + ind] == 1:
                        cnt += 1
                    else: break
                meta_buffer.append(cnt)
        #print(meta_buffer)

        np.savetxt(fname, np.array(meta_buffer).astype(int), delimiter = '\n', fmt='%d')
    return


def save_metadata_scattered(masks, NN_name, multi_phase_opt, RBS_MAX = 256, is_encode = False, is_fc_inc = False):
    """
    Save Metadata of pruned network. Use when scattered SIMD Pruning Only.
    """
    for i, layer in enumerate(masks):
        # Save FC layer's metadata 
        print(i,layer)
        if len(layer.shape) == 2 and is_fc_inc == True:
            meta_buffer = []
            print("Saving Metadata for" + str(i) + "th FC Layer..")
            num_class, num_input_channel = layer.shape

            if multi_phase_opt == 0:    fname = './metadata/' + NN_name + 'fc' + str(i) + '.csv'
            else:   fname = './metadata/' + NN_name + 'fc' + str(i) + '.csv'

            for class_ind, input_channel_set in enumerate(layer):
                cnt = ''
                for input_ind, input_channel_data in enumerate(input_channel_set):
                    if input_ind % 4 == 0:
                        if input_channel_data == 0:
                            cnt += '0'
                        else: cnt += '1'
                meta_buffer.append(cnt)

            # Metadata Count method
            if is_encode:
                np.savetxt(fname, np.array(_encode_binary2int(meta_buffer)), delimiter = '\n', fmt='%s')
            else:
                np.savetxt(fname, np.array(meta_buffer).astype(str), delimiter = '\n', fmt='%s')

        # Save CONV layer's metadata 
        if len(layer.shape) == 4:
            print("Saving Metadata for" + str(i) + "th Conv Layer..")
            meta_buffer = []
            if multi_phase_opt == 0:    fname = './metadata/' + NN_name + 'conv' + str(i) + '.csv'
            else:   fname = './metadata/' + NN_name + 'conv' + str(i) + '.csv'

            num_filter, channel, width, height = layer.shape
            filter_size = channel * width * height
            temp_layer = layer.reshape(num_filter, filter_size)
            num_filter_divided = num_filter // 4
            print("num_filter_divided_4 and filter size = {}, {}".format(num_filter_divided, filter_size))

            # Set Reduction Block Size
            if filter_size < RBS_MAX:
                reduction_block_size = filter_size
                num_reduction_block = 1
            else:
                reduction_block_size = RBS_MAX
                num_reduction_block = filter_size // reduction_block_size

            print("num_reduction_block = {}".format(num_reduction_block))
            # Extract Metadata information from RBS
            for reduction_block_ind in range(num_reduction_block):
                reduction_block_start = reduction_block_ind * reduction_block_size
                for filter_ind_f in range(0, num_filter, 4):
                    cnt = ''
                    # filter size < RBS_MAX
                    if filter_size < RBS_MAX:
                        for ind in range(reduction_block_size):
                            if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                                cnt += '1'
                            else:
                                cnt += '0'
                    # filter size >= RBS_MAX
                    else:
                        for ind in range(reduction_block_size):
                            if temp_layer[filter_ind_f][reduction_block_start + ind] == 1:
                                cnt += '1'
                            else:
                                cnt += '0'

                    meta_buffer.append(cnt)

            # Extract Metadata information from LeftOver
            if filter_size % reduction_block_size !=0:
                leftover_block_start = (reduction_block_ind+1) * reduction_block_size
                leftover_block_size = filter_size - leftover_block_start
                for filter_ind_f in range(0, num_filter, 4):
                    cnt = ''
                    for ind in range(leftover_block_size):
                        if temp_layer[filter_ind_f][leftover_block_start + ind] == 1:
                            cnt += '1'
                        else:
                            cnt += '0'

                    meta_buffer.append(cnt)

            if is_encode:
                np.savetxt(fname, np.array(_encode_binary2int(meta_buffer)), delimiter = '\n', fmt='%s')
            else:
                np.savetxt(fname, np.array(meta_buffer), delimiter = '\n', fmt='%s')
    return

def update_params(net, filename):
    print(" * Update Parameters from pkl Model file.. : {}".format(filename))
    model_dict = net.state_dict()
    loaded_dict = torch.load(filename, map_location = 'cpu')
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    net.load_state_dict(model_dict)
    return


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def rand_bin_array(total_num, zero_portion):
    total_zeros = int(total_num*zero_portion)
    total_ones = total_num-total_zeros
    rand_bin_array = np.array([0]*total_zeros + [1]*total_ones)
    np.random.shuffle(rand_bin_array)

    return rand_bin_array

def check_masks(masks):
    for mask in masks:
            temp = mask.cpu().detach().numpy().reshape(-1)
            print(temp[temp==2])
