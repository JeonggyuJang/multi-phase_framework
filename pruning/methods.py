import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pruning.utils import prune_rate, arg_nonzero_min, count_specific_layer, rand_bin_array, save_metadata_gpu
import pdb

def gpu_prune(model, pruning_perc, pruning_pattern):
    num_conv = count_specific_layer(model, 'Conv')
    prune_layer_flag = [e for e in range(num_conv)] # Prune all Conv, ConvNet
    num_workgroup = 16 # Need to check!
    #num_microtile = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] # VGG16, Conv 13
    num_microtile = [4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2] # VGG16, Conv 13
    num_iter = [1] + [16 for e in range(num_conv-1)] #VGG16, Conv 13, K
    masks = []
    conv_cnt = 0
    for module in model.modules():
        if type(module).__name__ == 'MaskedConv2d':
            if pruning_pattern == 'block' or conv_cnt == 0:
                print("*Selected block gpu pruning\n")
                num_macrotile = num_workgroup * num_microtile[conv_cnt]
                shape = module.weight.shape
                num_filter, filter_size = shape[0], shape[1]*shape[2]*shape[3]
                mask = np.ones((num_filter, filter_size))
                layer_param = module.weight.cpu().data.abs().numpy().reshape((num_filter, filter_size))
                if conv_cnt in prune_layer_flag:
                    for filter_offset in range(0, num_filter, num_macrotile): # Calculate l2Norm
                        l2_matrix = []
                        for block_offset in range(0, filter_size, num_iter[conv_cnt]):
                            l2_matrix.append((LA.norm(layer_param[filter_offset:filter_offset+num_macrotile, block_offset:block_offset+num_iter[conv_cnt]], 2) / (num_macrotile * num_iter[conv_cnt])))

                        l2_matrix = np.array(l2_matrix)
                        threshold = np.percentile(l2_matrix, pruning_perc*100)
                        l2_matrix_pruned_inds = l2_matrix > threshold

                        for block_ind, prune_flag in enumerate(l2_matrix_pruned_inds):
                            if not prune_flag: # if True, means alive
                                block_offset = block_ind*num_iter[conv_cnt]
                                mask[filter_offset:filter_offset + num_macrotile, block_offset: block_offset+num_iter[conv_cnt]] = 0

                mask = torch.from_numpy((mask.reshape(shape)).astype(float)).type(torch.FloatTensor)
                masks.append(mask)
                conv_cnt += 1

            elif pruning_pattern == 'micro':
                print("*Selected micro_tile pruning\n")
                num_macrotile = num_workgroup * num_microtile[conv_cnt]
                shape = module.weight.shape
                num_filter, filter_size = shape[0], shape[1]*shape[2]*shape[3]
                mask = np.ones((num_filter, filter_size))
                layer_param = module.weight.cpu().data.abs().numpy().reshape((num_filter, filter_size))

                if conv_cnt in prune_layer_flag:
                    for filter_offset in range(0, num_filter, num_macrotile): # Filter 0 ~ 32
                        for workitem_offset in range(num_workgroup): # 0 ~ 15, workitem idx location
                            l2_matrix = []
                            for block_offset in range(0, filter_size, num_workgroup): # 0, 16, 32, ... -> 0+0, 16+0, ..., prune for w.i 0
                                l2_matrix.append((LA.norm(layer_param[filter_offset:filter_offset+num_macrotile, block_offset+workitem_offset], 2) / num_macrotile))

                            l2_matrix = np.array(l2_matrix)
                            if pruning_perc == 0:
                                threshold = np.percentile(l2_matrix, 0) - 1
                            else:
                                threshold = np.percentile(l2_matrix, pruning_perc*100)
                            l2_matrix_pruned_inds = l2_matrix > threshold

                            for block_ind, prune_flag in enumerate(l2_matrix_pruned_inds):
                                if not prune_flag: # if True, means alive
                                    mask[filter_offset:filter_offset + num_macrotile, block_ind*num_workgroup + workitem_offset] = 0

                mask = torch.from_numpy((mask.reshape(shape)).astype(float)).type(torch.FloatTensor)
                masks.append(mask)
                conv_cnt += 1

            elif pruning_pattern == 'pattern':
                print("*Selected pattern pruning\n")
                pattern_list = np.array([[0, 4, 8, 12],[1, 5, 9, 13],[2, 6, 10, 14], [3, 7, 11, 15]])
                shape = module.weight.shape
                num_filter, filter_size = shape[0], shape[1]*shape[2]*shape[3]
                mask = np.zeros((num_filter, filter_size)) if conv_cnt in prune_layer_flag else np.ones((num_filter, filter_size))
                layer_param = module.weight.cpu().data.abs().numpy().reshape((num_filter, filter_size))

                if conv_cnt in prune_layer_flag:
                    for filter_offset in range(0, num_filter, num_workgroup*2): # Calculate l2Norm
                        for ind, block_offset in enumerate(range(0, filter_size, num_workgroup)):
                            l2_list =[]
                            for pat in pattern_list:
                                l2_sum = 0
                                for ind in pat:
                                    l2_sum += LA.norm(layer_param[filter_offset:filter_offset+num_workgroup*2, block_offset+ind], 2)
                                l2_list.append(l2_sum)
                            pat_num = l2_list.index(max(l2_list)) #Decide pattern
                            #pdb.set_trace()

                            for i in range(num_workgroup):
                                if(i in pattern_list[pat_num]):
                                    mask[filter_offset:filter_offset+num_workgroup*2, block_offset+i] = 1
                                    #pdb.set_trace()
                mask = torch.from_numpy((mask.reshape(shape)).astype(float)).type(torch.FloatTensor)
                masks.append(mask)
                conv_cnt += 1

            elif pruning_pattern == 'topbottom': # Too bad
                print("*Selected top-bottom pruning\n")
                shape = module.weight.shape
                num_filter, filter_size = shape[0], shape[1]*shape[2]*shape[3]
                mask = np.zeros((num_filter, filter_size)) if conv_cnt in prune_layer_flag else np.ones((num_filter, filter_size))
                layer_param = module.weight.cpu().data.abs().numpy().reshape((num_filter, filter_size))

                if conv_cnt in prune_layer_flag:
                    for filter_offset in range(0, num_filter, num_workgroup*2): # Calculate l2Norm
                        num_block = filter_size//num_workgroup #If not sliced, Error!!
                        num_prune = int(filter_size*pruning_perc)
                        block_ptr = [num_workgroup-1 for i in range(num_block)] # 0 ~ 15
                        while(num_prune > 0):
                            l2_matrix = []
                            for ind, block_offset in enumerate(range(0, filter_size, num_workgroup)):
                                if block_ptr[ind]<0:    l2_matrix.append(sys.maxsize)
                                else:
                                    l2_matrix.append((LA.norm(layer_param[filter_offset:filter_offset+num_workgroup*2, block_offset+block_ptr[ind]], 2)))
                            block_ptr[l2_matrix.index(min(l2_matrix))] -= 1
                            num_prune -= 1
                        # Make masks
                        block_ptr = np.array(block_ptr) + 1 #ptr -> num
                        for ind, block_offset in enumerate(range(0, filter_size, num_workgroup)):
                            mask[filter_offset:filter_offset+num_workgroup*2, block_offset: block_offset+block_ptr[ind]] = 1

                mask = torch.from_numpy((mask.reshape(shape)).astype(float)).type(torch.FloatTensor)
                masks.append(mask)
                conv_cnt += 1

            elif pruning_pattern == 'naive':
                print("*Selected naive gpu pruning\n")
                shape = module.weight.shape
                num_filter, filter_size = shape[0], shape[1]*shape[2]*shape[3]
                num_prune  = int(num_workgroup * pruning_perc)
                print("Debug, Num_prune {}\n".format(num_prune))
                mask = np.ones((num_filter, filter_size))

                if conv_cnt in prune_layer_flag: #Prune out this conv layer
                    for offset in range(filter_size):
                        if offset % num_workgroup == 0:
                            mask[:, offset+num_workgroup-num_prune : offset+num_workgroup] = 0
                mask = torch.from_numpy((mask.reshape(shape)).astype(float)).type(torch.FloatTensor)
                masks.append(mask)
                conv_cnt += 1

            else:
                print("Error ! : You Should pick GPU pruning pattern!\n")
                raise ValueError

    save_metadata_gpu(masks, model.__class__.__name__, prune_layer_flag, pruning_pattern, num_workgroup, num_conv)
    return masks


def simd_prune(model, pruning_perc, RBS_MAX = 256, prune_fc = False, is_scattered = False):
    """
    model -> parameters -> l2cal -> l2matrix -> pruning
    masks + l2_matrix
    """
    #np.set_printoptions(threshold = 10000000) # Set print num thoreshold - Debug
    leftover_filter_size = 0
    first_layer_flag = 0; masks = []; #l2_masks = []; #Uncomment this when use l2_masks
    for layer_ind, p in enumerate(model.parameters()):
        # Prune fc layer if prune_fc = True
        if prune_fc == True and len(p.data.size()) == 2:
            # Get shape of selected layer
            mask = []
            l2_matrix = []
            layer_param = p.cpu().data.abs().numpy()
            num_class, num_input_channel = layer_param.shape
            print(" * Shape of {}'th fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, num_class, num_input_channel))

            # STEP 1 : Make l2_matrix per RBS
            # TODO Expand this code for when it's not 4's multiple
            unit_l2_set = []
            for class_ind, input_params in enumerate(layer_param):
                for param_ind, param in enumerate(input_params): #512
                    unit_l2_set.append(param)
                    if (param_ind+1) % 4 == 0:
                        l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                        unit_l2_set = []

            # STEP 2 : Pruning stage based on l2_norm threshold
            l2_matrix = np.array(l2_matrix)
            threshold = np.percentile(l2_matrix, pruning_perc)
            if pruning_perc == 0:
                threshold -= 0.1

            l2_matrix_pruned_inds =  l2_matrix > threshold
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()
            print("     !! <DEBUG> l2_mask = ", np.array(l2_mask))
            for input_ind_div4, mask_data in enumerate(l2_mask):
                for i in range(4):
                    mask.append(mask_data)

            mask = torch.from_numpy(np.reshape(np.array(mask), (num_class, num_input_channel)).astype(float)).type(torch.FloatTensor)
            masks.append(mask)

        # Per Convolution Layer
        if len(p.data.size()) == 4:
            # Initialization
            mask = []
            l2_matrix = []

            # Get layer shape
            layer_param = p.cpu().data.abs().numpy()
            num_filter, channel, width, height = layer_param.shape
            filter_size = channel * width * height
            print(" * Shape of {}'th Conv layer (#filters,#channels,width,height,filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, num_filter, channel, width, height, filter_size))

            # STEP 1 : Make l2_matrix per RBS
            for kernel_ind, data in enumerate(layer_param):
                kernel_data = []
                if kernel_ind%4 == 0 and kernel_ind+4 <= num_filter:
                    for i in range(4):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])

                    # Copy Block Region
                    for i in range(filter_size):
                        # Simd 단위
                        unit_l2_set = []
                        for j in range(4):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)

                elif kernel_ind % 4 == 0 and kernel_ind + 4 > num_filter:
                    leftover_filter_size = num_filter - kernel_ind
                    for i in range(leftover_filter_size):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])
                    for i in range(filter_size):
                        unit_l2_set = []
                        for j in range(leftover_filter_size):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/leftover_filter_size)

            l2_matrix = np.array(l2_matrix)
            #print("    !! <DEBUG> l2_matrix.shape = ", l2_matrix.shape)

            # STEP 2 : Pruning stage based on l2_norm threshold
            #   1. Decide l2_norm threshold value
            if first_layer_flag == 0:
                threshold = np.percentile(l2_matrix, pruning_perc/2)
                first_layer_flag +=1
            else : threshold = np.percentile(l2_matrix, pruning_perc)

            if pruning_perc == 0:
                threshold -= 0.1

            #   2. Set Reduction block size for each filter
            if filter_size < RBS_MAX :
                reduction_block_size = filter_size
                num_reduction_block = 1
            else :
                reduction_block_size = RBS_MAX
                num_reduction_block = filter_size // reduction_block_size

            if num_filter % 4 != 0:
                l2_matrix_filter_size = num_filter//4+1
            else:
                l2_matrix_filter_size = num_filter//4

            #   3. Pruning consider Reduction_block_size
            l2_matrix_pruned_inds =  l2_matrix > threshold
            print(l2_matrix_pruned_inds.shape)
            #print("    !! <DEBUG> After thresholding = ",l2_matrix_pruned_inds)
            l2_matrix_pruned_inds = np.reshape(l2_matrix_pruned_inds, (l2_matrix_filter_size, filter_size))
            #print("    !! <DEBUG> l2_matrix.shape = ", l2_matrix.shape)
            #print("    !! <DEBUG> After Reshape, l2_matrix_pruned_inds.shape = ", l2_matrix_pruned_inds)

            # STEP 3 : Pick out actual pruned parameter (From right to left)
            if not is_scattered:
                for filter_ind in range(l2_matrix_filter_size):
                    for reduction_block_ind in range(num_reduction_block):
                        reduction_block_start = reduction_block_ind * reduction_block_size
                        ind = reduction_block_size - 1
                        while ind != 0:
                            if l2_matrix_pruned_inds[filter_ind][reduction_block_start + ind]:
                                l2_matrix_pruned_inds[filter_ind][reduction_block_start:(reduction_block_start + ind + 1)] = True
                                break
                            ind -= 1
                    if filter_size % reduction_block_size !=0:
                        leftover_block_start = (reduction_block_ind+1) * reduction_block_size
                        leftover_block_size = filter_size - leftover_block_start
                        ind = leftover_block_size - 1
                        while ind != 0:
                            if l2_matrix_pruned_inds[filter_ind][leftover_block_start + ind]:
                                l2_matrix_pruned_inds[filter_ind][leftover_block_start:(leftover_block_start + ind + 1)] = True
                                break
                            ind -= 1
            # STEP 4 : Change data type ( Bool -> Float ) 
            print("    !! <DEBUG> Result of Pick out l2_matrix_pruned_inds : ", l2_matrix_pruned_inds)
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()
            #print(l2_mask)
            #l2_masks.append(l2_mask)
            print("    !! <DEBUG> l2_mask.shape = ", np.array(l2_mask).shape)
            #print("    !! <DEBUG> l2_masks.shape = ", np.array(l2_mask).shape)
            print("    !! <DEBUG> leftover_filter_size", leftover_filter_size)

            # STEP 5 : Generate Masks based on l2_mask
            for filter_ind, filter_data in enumerate(l2_matrix_pruned_inds):
                if (filter_ind+1)*4 <= num_filter:
                #if num_filter % 4 == 0:
                    for i in range(4):
                        mask.append(filter_data)
                else:
                    for i in range(leftover_filter_size):
                        mask.append(filter_data)

            print("    !! <DEBUG> mask.shape ", np.array(mask).shape)

            #for i in range(num_filter%4):
            #    mask.append(list(np.ones(filter_size)))

            #mask = np.reshape(np.array(mask), (num_filter, channel, width, height)).astype(float).tolist()
            mask = torch.from_numpy(np.reshape(np.array(mask), (num_filter, channel, width, height)).astype(float)).type(torch.FloatTensor)
            #print("mask.shape = \n", np.array(mask).shape)
            #print("Typeof mask", type(mask))
            masks.append(mask)
            #print("masks.shape \n", np.array(masks).shape)
            #print("typeof Masks", type(masks))

    return masks
    # Should we return l2_masks?

def simd_deprune(masks, golden_model, de_pruning_perc, with_fc = False):
    """
        De-pruning between KC_sub ~ KC_upper.
        Decide de-prune candidate from golden_model-> golden_model
        상위 de_pruning_perc만큼 살리는 대신, (100-de_pruning_erc)만큼 없애는 것으로. 남는 것이 상위
        CAUTION!!
          0 < de_pruning_perc <= 100
          KC_depruning 이후 SIMD(0->1, 0->2, 0->3)순서로 생성, 1->2순서 불가.
    """
    np.set_printoptions(threshold = 10000000) # Set print num thoreshold - Debug
    leftover_filter_size = 0
    first_layer_flag = 0;
    inference_masks = []; grad_masks = [];
    conv_layer_cnt = -1; fc_layer_cnt = -1

    for layer_ind, p in enumerate(golden_model.parameters()):
        # FC Layer (+ If with_fc = True)
        if with_fc and len(p.data.size()) == 2:
            fc_layer_cnt += 1
            # Get shape of selected layer
            l2_matrix = []
            layer_param = torch.mul(p.cpu().data, 1-masks[conv_layer_cnt+fc_layer_cnt+1]).abs().numpy()

            #layer_param = p.cpu().data.abs().numpy()
            num_class, num_input_channel = layer_param.shape
            print(" * Shape of {}'th fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, num_class, num_input_channel))

            # STEP 1 : Make l2_matrix per RBS : It is considered when not 4's multiply
            unit_l2_set = []
            for class_ind, input_params in enumerate(layer_param):
                for param_ind in range(num_input_channel-num_input_channel%4):
                    unit_l2_set.append(input_params[param_ind])
                    if (param_ind+1) % 4 == 0:
                        l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                        unit_l2_set = []
                if num_input_channel%4 != 0:
                    l2_matrix.append(LA.norm(input_params[num_input_channel-num_input_channel%4:], 2) / (num_input_channel%4))

            # STEP 2 : Pruning stage based on l2_norm threshold
            l2_matrix = np.array(l2_matrix) # No zero value included
            #threshold = np.percentile(l2_matrix[l2_matrix!=0], 100-de_pruning_perc) #Except zero values

            #KYUU : Random Selection
            l2_matrix[l2_matrix!=0] = np.multiply(l2_matrix[l2_matrix!=0], rand_bin_array(l2_matrix[l2_matrix!=0].size, 1-de_pruning_perc/100))
            l2_matrix[l2_matrix!=0] = 1

            #l2_matrix_pruned_inds =  l2_matrix >= threshold
            #l2_mask = l2_matrix_pruned_inds.astype(float).tolist()

            inference_mask = torch.ones((num_class, num_input_channel)).type(torch.FloatTensor)
            param_cnt = 0

            for class_ind in range(num_class):
                for param_ind in range(num_input_channel-num_input_channel%4):
                    if param_ind%4==0:
                        for i in range(4):
                            inference_mask[class_ind][param_ind+i] = l2_matrix[param_cnt]
                        param_cnt +=1
                if num_input_channel%4 != 0:
                    inference_mask[class_ind][num_input_channel-num_input_channel%4:] = l2_matrix[param_cnt]
                    param_cnt+=1

            #print(mask[0])
            inference_masks.append(inference_mask + masks[conv_layer_cnt + fc_layer_cnt + 1])

        # Per Convolution Layer
        if len(p.data.size()) == 4:
            # Initialization
            conv_layer_cnt+=1
            inference_mask = []
            l2_matrix = []

            # Get layer shape
            layer_param = torch.mul(p.cpu().data, 1-masks[conv_layer_cnt]).abs().numpy()
            #layer_param = p.cpu().data.abs().numpy()
            num_filter, channel, width, height = layer_param.shape
            filter_size = channel * width * height


            print(" * Shape of {}'th Conv layer (#filters, #channels, width, height, filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, num_filter, channel, width, height, filter_size))

            # STEP 1 : Make l2_matrix per RBS
            for kernel_ind, data in enumerate(layer_param):
                kernel_data = []
                if kernel_ind % 4 == 0 and kernel_ind+4 <= num_filter:
                    for i in range(4):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])

                    # Copy Block Region
                    for i in range(filter_size):
                        # SIMD 단위
                        unit_l2_set = []
                        for j in range(4):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)
                    #print(len(l2_matrix))

                elif kernel_ind % 4 == 0 and kernel_ind + 4 > num_filter:
                    leftover_filter_size = num_filter - kernel_ind
                    for i in range(leftover_filter_size):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])
                    for i in range(filter_size):
                        unit_l2_set = []
                        for j in range(leftover_filter_size):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/leftover_filter_size)

            # STEP 2 : Pruning stage based on l2_norm threshold
            #   1. Decide l2_norm threshold value
            l2_matrix = np.array(l2_matrix) # No zero value included
            #threshold = np.percentile(l2_matrix[l2_matrix!=0], 100-de_pruning_perc) #Except zero values


            if num_filter % 4 != 0:
                l2_matrix_filter_size = num_filter//4+1
            else:
                l2_matrix_filter_size = num_filter//4

            #   3. Pruning consider Reduction_block_size
            #l2_matrix_pruned_inds =  l2_matrix >= threshold

            # KYUU : Random Selection
            l2_matrix[l2_matrix!=0] = np.multiply(l2_matrix[l2_matrix!=0], rand_bin_array(l2_matrix[l2_matrix!=0].size, 1-de_pruning_perc/100))
            l2_matrix[l2_matrix!=0] = 1
            l2_matrix_pruned_inds = l2_matrix


            # STEP 4 : Generate Masks based on l2_mask
            filter_start = 0
            for l2_filter_ind in range(l2_matrix_filter_size):
                if (l2_filter_ind+1)*4<=num_filter:
                    for i in range(4):
                        inference_mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

                else:
                    for i in range(leftover_filter_size):
                        inference_mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

            inference_mask = np.array(inference_mask)

            inference_mask = torch.from_numpy(np.reshape(inference_mask, (num_filter, channel, width, height)).astype(float)).type(torch.FloatTensor)
            inference_masks.append(inference_mask + masks[conv_layer_cnt])

    return inference_masks

def simd_prune_scalpel(model, pruning_perc, prune_fc=False, is_scattered=False):
    """
        Scalpel pruning to Measure Accuracy
    """
    #np.set_printoptions(threshold = 10000000) # Set print num thoreshold - Debug
    leftover_filter_size = 0
    first_layer_flag = 0; masks = []; #l2_masks = []; #Uncomment this when use l2_masks
    for layer_ind, p in enumerate(model.parameters()):
        # Prune fc layer if prune_fc = True
        if prune_fc == True and len(p.data.size()) == 2:
            # Get shape of selected layer
            mask = []
            l2_matrix = []
            layer_param = p.cpu().data.abs().numpy()
            num_class, num_input_channel = layer_param.shape
            print(" * Shape of {}'th fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, num_class, num_input_channel))

            # STEP 1 : Make l2_matrix per RBS
            # TODO Expand this code for when it's not 4's multiple
            unit_l2_set = []
            for class_ind, input_params in enumerate(layer_param):
                for param_ind, param in enumerate(input_params): #512
                    unit_l2_set.append(param)
                    if (param_ind+1) % 4 == 0:
                        l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                        unit_l2_set = []

            # STEP 2 : Pruning stage based on l2_norm threshold
            l2_matrix = np.array(l2_matrix)
            threshold = np.percentile(l2_matrix, pruning_perc)
            if pruning_perc == 0:
                threshold -= 0.1

            l2_matrix_pruned_inds =  l2_matrix > threshold
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()
            print("     !! <DEBUG> l2_mask = ", np.array(l2_mask))
            for input_ind_div4, mask_data in enumerate(l2_mask):
                for i in range(4):
                    mask.append(mask_data)

            mask = torch.from_numpy(np.reshape(np.array(mask), (num_class, num_input_channel)).astype(float)).type(torch.FloatTensor)
            masks.append(mask)

        # Per Convolution Layer
        if len(p.data.size()) == 4:
            # Initialization
            mask = []
            l2_matrix = []

            # Get layer shape
            layer_param = p.cpu().data.abs().numpy()
            num_filter, channel, width, height = layer_param.shape
            filter_size = channel * width * height
            print(" * Shape of {}'th Conv layer (#filters,#channels,width,height,filter_size!!)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, num_filter, channel, width, height, filter_size))

            layer_param = np.reshape(layer_param, (num_filter, channel*width*height))
            # STEP 1 : Make l2_matrix per RBS
            for filter_data in layer_param:
                for ind in range(0, filter_size, 4):
                    unit_l2_set = []
                    if (ind + 3) < (filter_size):
                        unit_l2_set = np.append(unit_l2_set, filter_data[ind:ind+4])
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)

            l2_matrix = np.array(l2_matrix)

            # STEP 2 : Pruning stage based on l2_norm threshold
            #   1. Decide l2_norm threshold value
            if first_layer_flag == 0:
                threshold = np.percentile(l2_matrix, pruning_perc/2)
                first_layer_flag +=1
            else : threshold = np.percentile(l2_matrix, pruning_perc)

            if pruning_perc == 0:
                threshold -= 0.1

            l2_matrix_filter_size = filter_size//4

            #   3. Pruning consider Reduction_block_size
            l2_matrix_pruned_inds =  l2_matrix > threshold
            l2_matrix_pruned_inds = np.reshape(l2_matrix_pruned_inds, (num_filter, l2_matrix_filter_size))


            # STEP 3 : Change data type ( Bool -> Float ) 
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()

            # STEP 4 : Generate Masks based on l2_mask
            for filter_data in l2_mask:
                temp_list = []
                for data in filter_data:
                    for i in range(4):
                        temp_list = np.append(temp_list, data)

                if temp_list.size < filter_size:
                    leftover_filter_size = filter_size - temp_list.size
                    for i in range(leftover_filter_size):
                        temp_list = np.append(temp_list, 1)
                mask.append(temp_list)

            mask = torch.from_numpy(np.reshape(np.array(mask), (num_filter, channel, width, height)).astype(float)).type(torch.FloatTensor)
            masks.append(mask)

    return masks
    # Should we return l2_masks?

def kc_simd_deprune(masks, golden_model, de_pruning_perc, pruned_net_info, with_fc = False):
    """
        De-pruning between KC_sub ~ KC_upper.
        Decide de-prune candidate from golden_model-> golden_model
        상위 de_pruning_perc만큼 살리는 대신, (100-de_pruning_erc)만큼 없애는 것으로. 남는 것이 상위
        CAUTION!!
          0 < de_pruning_perc <= 100
          KC_depruning 이후 SIMD(0->1, 0->2, 0->3)순서로 생성, 1->2순서 불가.
    """
    np.set_printoptions(threshold = 10000000) # Set print num thoreshold - Debug
    leftover_filter_size = 0
    first_layer_flag = 0;
    inference_masks = []; grad_masks = [];
    conv_layer_cnt = -1; fc_layer_cnt = -1

    sub_pruned_shape_info = pruned_net_info['sub_pruned_shape_info']
    for layer_ind, p in enumerate(golden_model.parameters()):
        # FC Layer (+ If with_fc = True)
        if with_fc and len(p.data.size()) == 2:
            fc_layer_cnt += 1
            # Get shape of selected layer
            l2_matrix = []
            layer_param = torch.mul(p.cpu().data, 1-masks[conv_layer_cnt+fc_layer_cnt+1]).abs().numpy()

            #layer_param = p.cpu().data.abs().numpy()
            num_class, num_input_channel = layer_param.shape
            sub_input_channel = sub_pruned_shape_info[conv_layer_cnt + fc_layer_cnt]
            sub_output_channel = sub_pruned_shape_info[conv_layer_cnt + fc_layer_cnt + 1]
            print(" * Shape of {}'th fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, num_class, num_input_channel))
            print(" * Shape of {}'th sub fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, sub_output_channel, sub_input_channel))

            # STEP 1 : Make l2_matrix per RBS : It is considered when not 4's multiply
            unit_l2_set = []
            for class_ind, input_params in enumerate(layer_param):
                if (class_ind+1)<=sub_output_channel:
                    for param_ind in range(sub_input_channel + 4 - sub_input_channel%4, num_input_channel-num_input_channel%4):
                        unit_l2_set.append(input_params[param_ind])
                        if (param_ind+1) % 4 == 0:
                            l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                            unit_l2_set = []
                    if num_input_channel%4 != 0:
                        l2_matrix.append(LA.norm(input_params[num_input_channel-num_input_channel%4:], 2) / (num_input_channel%4))
                else:
                    for param_ind in range(num_input_channel-num_input_channel%4):
                        unit_l2_set.append(input_params[param_ind])
                        if (param_ind+1) % 4 == 0:
                            l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                            unit_l2_set = []
                    if num_input_channel%4 != 0:
                        l2_matrix.append(LA.norm(input_params[num_input_channel-num_input_channel%4:], 2) / (num_input_channel%4))

            # STEP 2 : Pruning stage based on l2_norm threshold
            l2_matrix = np.array(l2_matrix) # No zero value included
            #threshold = np.percentile(l2_matrix[l2_matrix!=0], 100-de_pruning_perc) #Except zero values
            #KYUU : Random Selection
            l2_matrix[l2_matrix!=0] = np.multiply(l2_matrix[l2_matrix!=0], rand_bin_array(l2_matrix[l2_matrix!=0].size, 1-de_pruning_perc/100))
            l2_matrix[l2_matrix!=0] = 1

            #l2_matrix_pruned_inds =  l2_matrix >= threshold
            #l2_mask = l2_matrix_pruned_inds.astype(float).tolist()

            inference_mask = torch.zeros((num_class, num_input_channel)).type(torch.FloatTensor)
            param_cnt = 0
            for class_ind in range(num_class):
                if (class_ind+1)<=sub_output_channel:
                    for param_ind in range(sub_input_channel + 4 - sub_input_channel%4, num_input_channel-num_input_channel%4):
                        if param_ind%4==0:
                            for i in range(4):
                                inference_mask[class_ind][param_ind+i] = l2_matrix[param_cnt]
                            param_cnt +=1
                    if num_input_channel%4 != 0:
                        inference_mask[class_ind][num_input_channel-num_input_channel%4:] = l2_matrix[param_cnt]
                        param_cnt+=1
                    if sub_input_channel%4 != 0:
                        inference_mask[class_ind][sub_input_channel : sub_input_channel +4 - sub_input_channel%4] = 1
                else:
                    for param_ind in range(num_input_channel-num_input_channel%4):
                        if param_ind%4==0:
                            for i in range(4):
                                inference_mask[class_ind][param_ind+i] = l2_matrix[param_cnt]
                            param_cnt +=1
                    if num_input_channel%4 != 0:
                        inference_mask[class_ind][num_input_channel-num_input_channel%4:] = l2_matrix[param_cnt]
                        param_cnt+=1
                    if sub_input_channel%4 != 0:
                        inference_mask[class_ind][sub_input_channel : sub_input_channel +4 - sub_input_channel%4] = 1
            if de_pruning_perc >=100:
                inference_mask[:,:] = 1 # Need to check
                inference_masks.append(inference_mask)
            else:
                inference_mask = inference_mask + masks[conv_layer_cnt + fc_layer_cnt + 1]
                inference_mask[:,sub_input_channel : sub_input_channel +4 - sub_input_channel%4] = 1
                inference_masks.append(inference_mask)

        # Per Convolution Layer
        if len(p.data.size()) == 4:
            # Initialization
            conv_layer_cnt+=1
            inference_mask = []
            l2_matrix = []

            # Get layer shape
            layer_param = torch.mul(p.cpu().data, 1-masks[conv_layer_cnt]).abs().numpy()
            #layer_param = p.cpu().data.abs().numpy()
            num_filter, channel, width, height = layer_param.shape
            filter_size = channel * width * height

            sub_num_filter = sub_pruned_shape_info[conv_layer_cnt]
            sub_channel = channel # Use channel when first layer comes
            if conv_layer_cnt!=0:   sub_channel = sub_pruned_shape_info[conv_layer_cnt-1]
            sub_filter_size = sub_channel * width * height

            print(" * Shape of {}'th Conv layer (#filters, #channels, width, height, filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, num_filter, channel, width, height, filter_size))
            print(" * Shape of {}'th Sub layer (#filters, #channels, width,height, filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, sub_num_filter, sub_channel, width, height, sub_filter_size))

            # STEP 1 : Make l2_matrix per RBS
            for kernel_ind, data in enumerate(layer_param):
                kernel_data = []
                if kernel_ind % 4 == 0 and kernel_ind+4 <= num_filter:
                    for i in range(4):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])

                    # Copy Block Region
                    for i in range(filter_size):
                        # SIMD 단위
                        unit_l2_set = []
                        for j in range(4):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)
                    #print(len(l2_matrix))

                elif kernel_ind % 4 == 0 and kernel_ind + 4 > num_filter:
                    leftover_filter_size = num_filter - kernel_ind
                    for i in range(leftover_filter_size):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])
                    for i in range(filter_size):
                        unit_l2_set = []
                        for j in range(leftover_filter_size):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/leftover_filter_size)

            # STEP 2 : Pruning stage based on l2_norm threshold
            #   1. Decide l2_norm threshold value
            l2_matrix = np.array(l2_matrix) # No zero value included
            #threshold = np.percentile(l2_matrix[l2_matrix!=0], 100-de_pruning_perc) #Except zero values


            if num_filter % 4 != 0:
                l2_matrix_filter_size = num_filter//4+1
            else:
                l2_matrix_filter_size = num_filter//4

            #   3. Pruning consider Reduction_block_size
            #l2_matrix_pruned_inds =  l2_matrix >= threshold

            # KYUU : Random Selection
            l2_matrix[l2_matrix!=0] = np.multiply(l2_matrix[l2_matrix!=0], rand_bin_array(l2_matrix[l2_matrix!=0].size, 1-de_pruning_perc/100))
            l2_matrix[l2_matrix!=0] = 1
            l2_matrix_pruned_inds = l2_matrix


            # STEP 4 : Generate Masks based on l2_mask
            filter_start = 0
            for l2_filter_ind in range(l2_matrix_filter_size):
                if (l2_filter_ind+1)*4<=num_filter:
                    for i in range(4):
                        inference_mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

                else:
                    for i in range(leftover_filter_size):
                        inference_mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

            inference_mask = np.array(inference_mask) ##mask 2D matrix (Not reshaped)
            inference_mask[:sub_num_filter: , :sub_filter_size] = 0

            # Cut when pruning perc == 100, because of 4's multiply SIMD characteristic : 다짜를땐 짜르고, 아닐땐 살리기.
            inference_mask = torch.from_numpy(np.reshape(inference_mask, (num_filter, channel, width, height)).astype(float)).type(torch.FloatTensor)
            inference_mask = inference_mask + masks[conv_layer_cnt]

            if sub_num_filter%4 != 0:
                inference_mask[sub_num_filter: sub_num_filter + 4 - sub_num_filter%4, :sub_channel] = 1

            inference_masks.append(inference_mask)

    return inference_masks

def kc_simd_prune(model, pruning_perc, pruned_net_info,  RBS_MAX = 256, with_fc = False):
    """
    model -> parameters -> l2cal -> l2matrix -> pruning
    TODO : Only consider pruning with fully-connected layer
    """
    #np.set_printoptions(threshold = 10000000) # Set print num thoreshold - Debug
    leftover_filter_size = 0
    first_layer_flag = 0;
    masks = []; #l2_masks = []; #Uncomment this when use l2_masks
    conv_layer_cnt = -1; fc_layer_cnt = -1

    sub_pruned_shape_info = pruned_net_info['sub_pruned_shape_info']
    for layer_ind, p in enumerate(model.parameters()):
        # Prune fc layer if with_fc = True
        if with_fc and len(p.data.size()) == 2 :
            fc_layer_cnt += 1
            # Get shape of selected layer
            mask = []
            l2_matrix = []
            layer_param = p.cpu().data.abs().numpy()
            num_class, num_input_channel = layer_param.shape
            sub_input_channel = sub_pruned_shape_info[conv_layer_cnt + fc_layer_cnt]
            sub_output_channel = sub_pruned_shape_info[conv_layer_cnt + fc_layer_cnt + 1]
            print(" * Shape of {}'th fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, num_class, num_input_channel))
            print(" * Shape of {}'th sub fc layer (#class, #input_channel) = ({}, {})".format(layer_ind, sub_output_channel, sub_input_channel))

            # STEP 1 : Make l2_matrix per RBS
            # TODO Expand this code for when it's not 4's multiple
            unit_l2_set = []
            for class_ind, input_params in enumerate(layer_param):
                for param_ind in range(sub_input_channel + 4 - sub_input_channel%4, num_input_channel-num_input_channel%4):
                    unit_l2_set.append(input_params[param_ind])
                    if (param_ind+1) % 4 == 0:
                        l2_matrix.append(LA.norm(unit_l2_set, 2) / 4)
                        unit_l2_set = []

            # STEP 2 : Pruning stage based on l2_norm threshold
            l2_matrix = np.array(l2_matrix)
            # jjg modified (if~elif~else)
            if pruning_perc == 100 :
                threshold = 9999999
            elif pruning_perc == 0:
                threshold -= 0.1
            else :
                threshold = np.percentile(l2_matrix, pruning_perc)

            l2_matrix_pruned_inds =  l2_matrix > threshold
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()

            mask = torch.ones((num_class, num_input_channel)).type(torch.FloatTensor)
            param_cnt = 0

            for class_ind in range(num_class):
                for param_ind in range(sub_input_channel + 4 - sub_input_channel%4, num_input_channel-num_input_channel%4):
                    if param_ind%4==0:
                        for i in range(4):
                            mask[class_ind][param_ind+i] = l2_mask[param_cnt]
                        param_cnt +=1
                if sub_input_channel%4 != 0:
                    mask[class_ind][sub_input_channel : sub_input_channel +4 - sub_input_channel%4] = 0

            if pruning_perc >=100:
                mask[:sub_output_channel, sub_input_channel:] = 0
                mask[sub_output_channel:] = 0
            masks.append(mask)

        # Per Convolution Layer
        if len(p.data.size()) == 4:
            # Initialization
            conv_layer_cnt+=1
            mask = []
            l2_matrix = []

            # Get layer shape
            layer_param = p.cpu().data.abs().numpy()
            num_filter, channel, width, height = layer_param.shape
            filter_size = channel * width * height

            sub_num_filter = sub_pruned_shape_info[conv_layer_cnt]
            sub_channel = channel # Use channel when first layer comes
            if conv_layer_cnt!=0:   sub_channel = sub_pruned_shape_info[conv_layer_cnt-1]
            sub_filter_size = sub_channel * width * height

            print(" * Shape of {}'th Conv layer (#filters, #channels, width, height, filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, num_filter, channel, width, height, filter_size))
            print(" * Shape of {}'th Sub layer (#filters, #channels, width,height, filter_size)=({}, {}, {}, {}, {})".format\
                                                              (layer_ind, sub_num_filter, sub_channel, width, height, sub_filter_size))

            # STEP 1 : Make l2_matrix per RBS
            for kernel_ind, data in enumerate(layer_param):
                kernel_data = []
                if kernel_ind % 4 == 0 and kernel_ind+4 <= num_filter and kernel_ind < sub_num_filter:
                    for i in range(4):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])

                    # Copy Block Region
                    for i in range(sub_filter_size, filter_size):
                        # Simd 단위
                        unit_l2_set = []
                        for j in range(4):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)
                    #print(len(l2_matrix))

                elif kernel_ind % 4 == 0 and kernel_ind+4 <= num_filter and kernel_ind >= sub_num_filter:
                    for i in range(4):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])

                    # Copy Block Region
                    for i in range(filter_size):
                        # Simd 단위
                        unit_l2_set = []
                        for j in range(4):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/4)
                    #print(len(l2_matrix))

                elif kernel_ind % 4 == 0 and kernel_ind + 4 > num_filter:
                    leftover_filter_size = num_filter - kernel_ind
                    for i in range(leftover_filter_size):
                        kernel_data = np.append(kernel_data, layer_param[kernel_ind+i])
                    for i in range(filter_size):
                        unit_l2_set = []
                        for j in range(leftover_filter_size):
                            unit_l2_set = np.append(unit_l2_set, kernel_data[j*filter_size + i])
                        # Calculate l2norm
                        l2_matrix.append(LA.norm(unit_l2_set, 2)/leftover_filter_size)

            l2_matrix = np.array(l2_matrix)

            # STEP 2 : Pruning stage based on l2_norm threshold
            #   1. Decide l2_norm threshold value
            if first_layer_flag == 0:
                #threshold = np.percentile(l2_matrix, pruning_perc/2)
                threshold = np.percentile(l2_matrix, pruning_perc)
                first_layer_flag +=1
            else : threshold = np.percentile(l2_matrix, pruning_perc)

            if pruning_perc == 0:
                threshold -= 0.1

            if num_filter % 4 != 0:
                l2_matrix_filter_size = num_filter//4+1
            else:
                l2_matrix_filter_size = num_filter//4

            #   3. Pruning consider Reduction_block_size
            l2_matrix_pruned_inds =  l2_matrix > threshold
            #print(l2_matrix_pruned_inds.shape)
            #l2_matrix_pruned_inds = np.reshape(l2_matrix_pruned_inds[], (l2_matrix_filter_size, filter_size)) # No use with reshape

            # STEP 4 : Change data type ( Bool -> Float ) 
            l2_mask = l2_matrix_pruned_inds.astype(float).tolist()

            # STEP 5 : Generate Masks based on l2_mask
            filter_start = 0
            for l2_filter_ind in range(l2_matrix_filter_size):
                if (l2_filter_ind+1)*4<=num_filter and (l2_filter_ind*4+1) <= sub_num_filter:
                    temp_mask = np.append(np.ones(sub_filter_size), l2_matrix_pruned_inds[filter_start:filter_start + filter_size - sub_filter_size])
                    for i in range(4):
                        mask.append(temp_mask)
                    filter_start += filter_size - sub_filter_size

                elif (l2_filter_ind+1)*4<=num_filter and (l2_filter_ind*4+1) > sub_num_filter:
                    for i in range(4):
                        mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

                else:
                    for i in range(leftover_filter_size):
                        mask.append(l2_matrix_pruned_inds[filter_start : filter_start + filter_size])
                    filter_start += filter_size

            mask = np.array(mask)

            # Cut when pruning perc == 100, because of 4's multiply SIMD characteristic : 다짜를땐 짜르고, 아닐땐 살리기.
            if sub_num_filter%4 != 0 and pruning_perc >= 100:
                mask[sub_num_filter: sub_num_filter + 4 - sub_num_filter%4,:sub_channel] = 0
            mask = torch.from_numpy(np.reshape(mask, (num_filter, channel, width, height)).astype(float)).type(torch.FloatTensor)
            masks.append(mask)

    return masks

def kc_prune(model, pruning_perc, with_fc = False):
    """
    pruning_perc : Target pruning rate per layer.
    with_fc : Prune out fc layer with conv_layer
    """
    total_conv_num = count_specific_layer(model, 'Conv')
    total_fc_num = count_specific_layer(model, 'Linear')
    print("    * Number of conv | fc of this network : (#{} | #{})\n".format(total_conv_num, total_fc_num))
    pruned_weights_sequence=[]
    pruned_bias_sequence=[]
    Prune_check_sequence=[]

    # Network info metadata
    pruned_net_info = {}
    pruned_shape_info = []
    cfg_list = []

    conv_layer_ind = 0
    fc_layer_ind = 0

    last_p_rate = 0
    now_p_rate = 0

    for module in model.modules():
        if type(module).__name__ == 'MaskedConv2d':
            conv_layer_ind += 1

            # Ready to Pruning : Torch format to Numpy
            weights = module.weight.cpu().data.numpy()
            bias = module.bias.cpu().data.numpy()
            layer_shape = module.weight.size()
            num_total_param = weights.size
            num_total_bias = bias.size
            print('     * KC_Pruning For Conv{} : ({}, {}, {}, {})'.format(conv_layer_ind,layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3]))

            # Pruning : #1 Channel Pruning_______________________________________________________________________________________________________
            channel_cnt=0
            if conv_layer_ind != 1:
                channel_pruned_weights = np.zeros((layer_shape[0], layer_shape[1]-len(Prune_check_sequence), layer_shape[2], layer_shape[3]))
                for i in range(0, layer_shape[1]):
                    if ((i in Prune_check_sequence) == False):
                        channel_pruned_weights[:,channel_cnt,:,:] = weights[:,i,:,:]
                        channel_cnt += 1
                    else:
                        Prune_check_sequence.remove(i)
                print("         - #{} Channels Pruned".format(layer_shape[1] - channel_cnt))
            else:
                channel_pruned_weights = weights

                #       - Parameter Save to Layer Buffer & Make cfg_list
            if conv_layer_ind == total_conv_num and not(with_fc):
                cfg_list.append(layer_shape[0])
                pruned_weights_sequence.append(channel_pruned_weights)
                pruned_bias_sequence.append(bias)

            # -----------------------------------------------------------------------------------------------------------------------------

            # Pruning : #2 Kernel Pruning__________________________________________________________________________________________________
            if conv_layer_ind != total_conv_num+1 or with_fc:
                #       - Calculate norm_list
                num_filter = channel_pruned_weights.shape[0]
                normlist = np.zeros(num_filter)
                for i in range(0, num_filter):
                    temp = channel_pruned_weights[i,:,:,:]
                    normlist[i] = (np.sum(temp**2))**0.5

                #       - Determine pruning_ratio of this layer
                now_p_rate =1 - (1-pruning_perc/100)/(1-last_p_rate)
                print("     !!DEBUG!! pruning_rate {}, last_p_rate {}, p_rate {} \n".format(pruning_perc, last_p_rate, now_p_rate))

                #       - Determine Threshold Value
                if conv_layer_ind == 1:
                    #threshold = np.percentile(normlist, now_p_rate*100/2)
                    threshold = np.percentile(normlist, now_p_rate*100)
                    #last_p_rate = now_p_rate/2
                    last_p_rate = now_p_rate
                else:
                    threshold = np.percentile(normlist, now_p_rate*100)
                    last_p_rate = now_p_rate

                #       - Kernel_Pruning
                for i in range(0, num_filter):
                    if(normlist[i]<threshold):
                        Prune_check_sequence.append(i)

                pruned_weights = np.zeros((channel_pruned_weights.shape[0]-len(Prune_check_sequence), channel_pruned_weights.shape[1],\
                                           channel_pruned_weights.shape[2], channel_pruned_weights.shape[3]), dtype='float32')
                pruned_bias = np.zeros((channel_pruned_weights.shape[0]-len(Prune_check_sequence)), dtype='float32')

                kernel_cnt=0
                for i in range(0, channel_pruned_weights.shape[0]):
                    if((i in Prune_check_sequence) == False):
                        pruned_weights[kernel_cnt,:,:,:] = channel_pruned_weights[i,:,:,:]
                        pruned_bias[kernel_cnt] = bias[i]
                        kernel_cnt+=1

                #       - Parameter Save to Layer Buffer
                pruned_shape_info.append((channel_pruned_weights.shape[0],channel_pruned_weights.shape[0]-len(Prune_check_sequence)))
                pruned_weights_sequence.append(pruned_weights)
                pruned_bias_sequence.append(pruned_bias)

                #       - Make cfg_list
                cfg_list.append(pruned_weights.shape[0])
                print("         - #{} Kernels Pruned : ".format(channel_pruned_weights.shape[0]-kernel_cnt), Prune_check_sequence)
            # ----------------------------------------------------------------------------------------------------------------------------------------------
            if conv_layer_ind == total_conv_num and (not with_fc):
                pruned_shape_info.append((channel_pruned_weights.shape[0], channel_pruned_weights.shape[0]))
                pruned_weights = channel_pruned_weights
                pruned_bias = bias

            # Pruned Parameters & Shape counting
            num_total_param_pruned = pruned_weights.size
            num_total_bias = pruned_bias.size
            pruned_layer_shape = pruned_weights.shape
            print('         - After Pruning conv{} : ({}, {}, {}, {})'.format(conv_layer_ind, pruned_layer_shape[0],\
                            pruned_layer_shape[1], pruned_layer_shape[2], pruned_layer_shape[3]))

        elif type(module).__name__ == "MaskedLinear":
            fc_layer_ind += 1
            # Get shape of FC layer
            layer_shape = module.weight.size()
            weights = module.weight.cpu().data.numpy()
            bias = module.bias.cpu().data.numpy()

            if not with_fc: #If not prune FC layer
                pruned_weights_sequence.append(weights)
                pruned_bias_sequence.append(bias)
                pruned_shape_info.append((layer_shape[0], layer_shape[0])) # Not pruned

            elif with_fc: #Input pruning
                image_size = 7*7
                print('     * KC_Pruning for FC{} : ({}, {})'.format(fc_layer_ind, layer_shape[0], layer_shape[1]))
                fc_input_pruned_weights = np.zeros((layer_shape[0],layer_shape[1]-len(Prune_check_sequence)*image_size))
                print("         - #{} Parameter Pruned : ".format(layer_shape[1]-fc_input_pruned_weights.shape[1]), Prune_check_sequence)
                param_cnt = 0
                for fc_param_ind in range(0, layer_shape[1], image_size): #layer_shape 25088 0, 49, 98, ...
                    if ((fc_param_ind//image_size in Prune_check_sequence) == False):
                        fc_input_pruned_weights[:,param_cnt*image_size:param_cnt*image_size+image_size] = weights[:,fc_param_ind:fc_param_ind+image_size]
                        param_cnt += 1
                    else:
                        Prune_check_sequence.remove(fc_param_ind//image_size)

                pruned_shape_info.append((layer_shape[0], fc_input_pruned_weights.shape[0]))
                cfg_list.append((fc_input_pruned_weights.shape[0], fc_input_pruned_weights.shape[1]))
                pruned_weights_sequence.append(fc_input_pruned_weights)
                pruned_bias_sequence.append(bias)
                #pdb.set_trace()
                with_fc = False

                """
                if fc_layer_ind == total_fc_num:
                    cfg_list.append((layer_shape[0], fc_input_pruned_weights.shape[1])) # cfg_fc = [(output), (input)]
                    pruned_shape_info.append((layer_shape[0], fc_input_pruned_weights.shape[0]))
                    pruned_weights_sequence.append(fc_input_pruned_weights)
                    pruned_bias_sequence.append(bias)

                elif fc_layer_ind != total_fc_num: # Node Pruning

                    print('     * KC_node_pruning for FC{} : ({}, {})'.format(fc_layer_ind, layer_shape[0], layer_shape[1]))
                    if len(Prune_check_sequence) != 0:
                        print("!!ERROR!! # of pruned filters({}) is not removed".format(len(Prune_check_sequence)))
                        raise ValueError

                    # Get norm_value of nodes
                    normlist = np.zeros(layer_shape[0])
                    for i in range(layer_shape[0]):
                        temp = fc_input_pruned_weights[i,:]
                        normlist[i] = (np.sum(temp**2))**0.5

                    # Get prune rate for this layer
                    now_p_rate = 1 - (1-pruning_perc/100)/(1-last_p_rate)
                    threshold = np.percentile(normlist, now_p_rate*100)
                    last_p_rate = now_p_rate
                    print("now_p_rate {} ".format(now_p_rate))

                    for i in range(layer_shape[0]):
                        if(normlist[i]<threshold):
                            Prune_check_sequence.append(i)

                    fc_pruned_weights = np.zeros((layer_shape[0]-len(Prune_check_sequence), fc_input_pruned_weights.shape[1]), dtype='float32')
                    fc_pruned_bias = np.zeros(layer_shape[0]-len(Prune_check_sequence), dtype='float32')

                    node_cnt = 0
                    for i in range(layer_shape[0]):
                        if((i in Prune_check_sequence) == False):
                            fc_pruned_weights[node_cnt,:] = fc_input_pruned_weights[i,:]
                            fc_pruned_bias[node_cnt] = bias[i]
                            node_cnt+=1


                    pruned_shape_info.append((layer_shape[0], fc_pruned_weights.shape[0]))
                    cfg_list.append((fc_pruned_weights.shape[0], fc_pruned_weights.shape[1]))
                    pruned_weights_sequence.append(fc_pruned_weights)
                    pruned_bias_sequence.append(fc_pruned_bias)
                """

        elif type(module).__name__ == "MaxPool2d":
            cfg_list.append("M")
        elif type(module).__name__ == "AvgPool2d":
            pass

        elif type(module).__name__ == "BatchNorm2d":
            if conv_layer_ind != total_conv_num or with_fc:
                weights = module.weight.cpu().data.numpy()
                bias = module.bias.cpu().data.numpy()
                running_var = module.running_var.cpu().data.numpy()
                running_mean = module.running_mean.cpu().data.numpy()

                num_pruned_kernel = weights.shape[0] - len(Prune_check_sequence)

                pruned_weights = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_bias = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_running_mean = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_running_var = np.zeros(num_pruned_kernel, dtype='float32')

                kernel_cnt=0
                for i in range(0, weights.shape[0]):
                    if((i in Prune_check_sequence) == False):
                        pruned_weights[kernel_cnt] = bias[i]
                        pruned_bias[kernel_cnt] = bias[i]
                        pruned_running_mean[kernel_cnt] = bias[i]
                        pruned_running_var[kernel_cnt] = bias[i]
                        kernel_cnt+=1

                pruned_weights_sequence.append(pruned_weights)
                pruned_weights_sequence.append(pruned_running_var)
                pruned_weights_sequence.append(pruned_running_mean)
                pruned_bias_sequence.append(pruned_bias)

                print("         - BatchNorm Pruned({})->({}) : weights({}), var({}), mean({}), bias({}) ".format(weights.shape[0], num_pruned_kernel,\
                                                                                                                 pruned_weights.shape, pruned_running_var.shape,\
                                                                                                                 pruned_running_mean.shape, pruned_bias.shape))
            else:
                print("         - Last BatchNorm Layer Not Pruned")
                pruned_weights_sequence.append(module.weight.cpu().data.numpy())
                pruned_weights_sequence.append(module.running_var.cpu().data.numpy())
                pruned_weights_sequence.append(module.running_mean.cpu().data.numpy())
                pruned_bias_sequence.append(module.bias.cpu().data.numpy())

    # Calculate Pruned_rate
    pruned_rate = 100.0*(1-num_total_param_pruned/num_total_param)

    # Pruned_net_info
    pruned_net_info['cfg_list'] = cfg_list
    pruned_net_info['pruned_shape_info'] = pruned_shape_info
    pruned_net_info['pruned_rate'] = pruned_rate

    return pruned_weights_sequence, pruned_bias_sequence, pruned_net_info

def kc_resnet_prune(model, pruning_perc, with_fc = False):
    """
    ResNet kc_pruning for temporary
    """
    total_conv_num = 2
    total_fc_num = count_specific_layer(model, 'Linear')
    print("    * Number of conv | fc of this network : (#{} | #{})\n".format(total_conv_num, total_fc_num))
    pruned_weights_sequence=[]
    pruned_bias_sequence=[]
    Prune_check_sequence=[]

    # Network info metadata
    pruned_net_info = {}
    pruned_shape_info = []
    cfg_list = []

    conv_layer_ind = 0
    fc_layer_ind = 0

    last_p_rate = 0
    now_p_rate = 0
    prune_flag = False

    for module in model.modules():
        if type(module).__name__ == 'BasicBlock':
            prune_flag = True
        if type(module).__name__ == 'Sequential':
            prune_flag = False
        if type(module).__name__ == 'MaskedConv2d' and not prune_flag:
            # Ready to Pruning : Torch format to Numpy
            weights = module.weight.cpu().data.numpy()
            layer_shape = module.weight.size()
            num_total_param = weights.size

            pruned_weights_sequence.append(weights)

        if type(module).__name__ == 'MaskedLinear' and not prune_flag:
            # Ready to Pruning : Torch format to Numpy
            weights = module.weight.cpu().data.numpy()
            bias = module.bias.cpu().data.numpy()
            layer_shape = module.weight.size()
            num_total_param = weights.size

            pruned_weights_sequence.append(weights)
            pruned_bias_sequence.append(bias)

        if type(module).__name__ == 'MaskedConv2d' and prune_flag:
            conv_layer_ind += 1

            # Ready to Pruning : Torch format to Numpy
            weights = module.weight.cpu().data.numpy()
            layer_shape = module.weight.size()
            num_total_param = weights.size
            print('     * KC_Pruning For Conv{} : ({}, {}, {}, {})'.format(conv_layer_ind,layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3]))

            # Pruning : #1 Channel Pruning_______________________________________________________________________________________________________
            channel_cnt=0
            if conv_layer_ind != 1 : #  First_layer of conv, Channel pruning
                channel_pruned_weights = np.zeros((layer_shape[0], layer_shape[1]-len(Prune_check_sequence), layer_shape[2], layer_shape[3]))
                for i in range(0, layer_shape[1]):
                    if ((i in Prune_check_sequence) == False):
                        channel_pruned_weights[:,channel_cnt,:,:] = weights[:,i,:,:]
                        channel_cnt += 1
                    else:
                        Prune_check_sequence.remove(i)
                print("         - #{} Channels Pruned".format(layer_shape[1] - channel_cnt))
            else:
                channel_pruned_weights = weights

                #       - Parameter Save to Layer Buffer & Make cfg_list
            if conv_layer_ind == total_conv_num and not(with_fc):
                cfg_list.append(layer_shape[0])
                pruned_weights_sequence.append(channel_pruned_weights)
                conv_layer_ind == 0

            # -----------------------------------------------------------------------------------------------------------------------------

            # Pruning : #2 Kernel Pruning__________________________________________________________________________________________________
            if conv_layer_ind != total_conv_num or with_fc: # 0 ~ Last_conv_layer with_fc = True
                #       - Calculate norm_list
                num_filter = channel_pruned_weights.shape[0]
                normlist = np.zeros(num_filter)
                for i in range(0, num_filter):
                    temp = channel_pruned_weights[i,:,:,:]
                    normlist[i] = (np.sum(temp**2))**0.5

                #       - Determine pruning_ratio of this layer
                now_p_rate =1 - (1-pruning_perc/100)/(1-last_p_rate)
                print("     !!DEBUG!! pruning_rate {}, last_p_rate {}, p_rate {} \n".format(pruning_perc, last_p_rate, now_p_rate))

                #       - Determine Threshold Value
                if conv_layer_ind == 1:
                    #threshold = np.percentile(normlist, now_p_rate*100/2)
                    threshold = np.percentile(normlist, now_p_rate*100)
                    #last_p_rate = now_p_rate/2
                    last_p_rate = now_p_rate
                else:
                    threshold = np.percentile(normlist, now_p_rate*100)
                    last_p_rate = now_p_rate

                #       - Kernel_Pruning
                for i in range(0, num_filter):
                    if(normlist[i]<threshold):
                        Prune_check_sequence.append(i)

                pruned_weights = np.zeros((channel_pruned_weights.shape[0]-len(Prune_check_sequence), channel_pruned_weights.shape[1],\
                                           channel_pruned_weights.shape[2], channel_pruned_weights.shape[3]), dtype='float32')

                kernel_cnt=0
                for i in range(0, channel_pruned_weights.shape[0]):
                    if((i in Prune_check_sequence) == False):
                        pruned_weights[kernel_cnt,:,:,:] = channel_pruned_weights[i,:,:,:]
                        kernel_cnt+=1

                #       - Parameter Save to Layer Buffer
                pruned_shape_info.append((channel_pruned_weights.shape[0],channel_pruned_weights.shape[0]-len(Prune_check_sequence)))
                pruned_weights_sequence.append(pruned_weights)

                #       - Make cfg_list
                cfg_list.append(pruned_weights.shape[0])
                print("         - #{} Kernels Pruned : ".format(channel_pruned_weights.shape[0]-kernel_cnt), Prune_check_sequence)
            # ----------------------------------------------------------------------------------------------------------------------------------------------
            if conv_layer_ind == total_conv_num : # Last Conv layer with_fc = False
                pruned_shape_info.append((channel_pruned_weights.shape[0], channel_pruned_weights.shape[0]))
                pruned_weights = channel_pruned_weights

            # Pruned Parameters & Shape counting
            num_total_param_pruned = pruned_weights.size
            pruned_layer_shape = pruned_weights.shape
            print('         - After Pruning conv{} : ({}, {}, {}, {})'.format(conv_layer_ind, pruned_layer_shape[0],\
                            pruned_layer_shape[1], pruned_layer_shape[2], pruned_layer_shape[3]))

        elif type(module).__name__ == "BatchNorm2d":
            if conv_layer_ind != total_conv_num or with_fc:
                weights = module.weight.cpu().data.numpy()
                bias = module.bias.cpu().data.numpy()
                running_var = module.running_var.cpu().data.numpy()
                running_mean = module.running_mean.cpu().data.numpy()

                num_pruned_kernel = weights.shape[0] - len(Prune_check_sequence)

                pruned_weights = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_bias = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_running_mean = np.zeros(num_pruned_kernel, dtype='float32')
                pruned_running_var = np.zeros(num_pruned_kernel, dtype='float32')

                kernel_cnt=0
                for i in range(0, weights.shape[0]):
                    if((i in Prune_check_sequence) == False):
                        pruned_weights[kernel_cnt] = bias[i]
                        pruned_bias[kernel_cnt] = bias[i]
                        pruned_running_mean[kernel_cnt] = bias[i]
                        pruned_running_var[kernel_cnt] = bias[i]
                        kernel_cnt+=1

                pruned_weights_sequence.append(pruned_weights)
                pruned_weights_sequence.append(pruned_running_var)
                pruned_weights_sequence.append(pruned_running_mean)
                pruned_bias_sequence.append(pruned_bias)

                print("         - BatchNorm Pruned({})->({}) : weights({}), var({}), mean({}), bias({}) ".format(weights.shape[0], num_pruned_kernel,\
                                                                                                                 pruned_weights.shape, pruned_running_var.shape,\
                                                                                                                 pruned_running_mean.shape, pruned_bias.shape))
            else:
                print("         - Last BatchNorm Layer Not Pruned")
                pruned_weights_sequence.append(module.weight.cpu().data.numpy())
                pruned_weights_sequence.append(module.running_var.cpu().data.numpy())
                pruned_weights_sequence.append(module.running_mean.cpu().data.numpy())
                pruned_bias_sequence.append(module.bias.cpu().data.numpy())

    # Calculate Pruned_rate
    pruned_rate = 100.0*(1-num_total_param_pruned/num_total_param)

    # Pruned_net_info
    pruned_net_info['cfg_list'] = cfg_list
    pruned_net_info['pruned_shape_info'] = pruned_shape_info
    pruned_net_info['pruned_rate'] = pruned_rate

    return pruned_weights_sequence, pruned_bias_sequence, pruned_net_info

def KC_initialize(model, pruned_weights_sequence, pruned_bias_sequence):
    layer_ind = 0
    layer_weight_ind = 0
    layer_bias_ind = 0
    print(model)
    for module in model.modules():
        if type(module).__name__ == 'MaskedConv2d':
            bias_flag = False if module.bias == None else True
            print("model size for conv{}_weights : ".format(layer_ind), module.weight.size())
            if bias_flag: print("model size for conv{}_bias : ".format(layer_ind), module.bias.size())

            print("pruned_size for conv{}_weights : ".format(layer_ind), pruned_weights_sequence[layer_weight_ind].shape)
            if bias_flag: print("pruned_size for conv{}_bias : ".format(layer_ind), pruned_bias_sequence[layer_bias_ind].shape)
            initialize_weight_shape = pruned_weights_sequence[layer_weight_ind].shape
            if bias_flag: initialize_bias_shape = pruned_bias_sequence[layer_bias_ind].shape

            if bias_flag: module.bias.data[:initialize_bias_shape[0]] = torch.from_numpy(pruned_bias_sequence[layer_bias_ind])
            for i in range(initialize_weight_shape[0]):
                    for j in range(initialize_weight_shape[1]):
                        for k in range(initialize_weight_shape[2]):
                            module.weight.data[i][j][k] = torch.from_numpy(pruned_weights_sequence[layer_weight_ind][i][j][k])

            layer_ind += 1
            layer_weight_ind += 1
            if bias_flag: layer_bias_ind += 1

        elif type(module).__name__ == 'MaskedLinear':
            bias_flag = False if module.bias == None else True
            #pdb.set_trace()
            print("pruned_size for fc{}_weights : ".format(layer_ind), pruned_weights_sequence[layer_weight_ind].shape)
            if bias_flag: print("pruned_size for fc{}_bias : ".format(layer_ind), pruned_bias_sequence[layer_bias_ind].shape)
            initialize_shape = pruned_weights_sequence[layer_weight_ind].shape
            if bias_flag: module.bias.data[:initialize_shape[0]] = torch.from_numpy(pruned_bias_sequence[layer_bias_ind])

            for i in range(initialize_shape[0]):
                #if bias_flag: module.bias.data[i] = torch.from_numpy(pruned_bias_sequence[layer_bias_ind][i])
                #for j in range(module.weight.shape[1]):
                module.weight.data[i] = torch.from_numpy(pruned_weights_sequence[layer_weight_ind][i])

            layer_ind += 1
            layer_weight_ind += 1
            if bias_flag: layer_bias_ind += 1

        elif type(module).__name__ == 'BatchNorm2d':
            bias_flag = False if module.bias == None else True
            initialize_shape = pruned_weights_sequence[layer_weight_ind].shape

            module.weight.data[:initialize_shape[0]] = torch.from_numpy(pruned_weights_sequence[layer_weight_ind])
            module.running_var.data[:initialize_shape[0]] = torch.from_numpy(pruned_weights_sequence[layer_weight_ind+1])
            module.running_mean.data[:initialize_shape[0]] = torch.from_numpy(pruned_weights_sequence[layer_weight_ind+2])
            if bias_flag: module.bias.data[:initialize_shape[0]] = torch.from_numpy(pruned_bias_sequence[layer_bias_ind])

            layer_ind += 1
            layer_weight_ind += 3
            if bias_flag: layer_bias_ind += 1

    return model

def KC_de_pruning(model, pruned_net_info, dP_rate, super_model, super_pruned_net_info, STD = 1e-1, with_fc=False):
    #Kyuu temp
    temp_features = super_model.features
    temp_classifier = super_model.classifier
    temp_cnt = -3

    pruned_shape_info = pruned_net_info['pruned_shape_info']
    #new_cfg_list = pruned_net_info['cfg_list']
    new_cfg_list = super_pruned_net_info['cfg_list']

    new_pruned_shape_info = super_pruned_net_info['pruned_shape_info']
    sub_pruned_shape_info = [y for x, y in pruned_shape_info]
    masks = []

    pruned_weights_sequence=[]
    pruned_bias_sequence=[]
    w_sum = 0
    new_w_sum = 0

    pruned_weights_sequence = []; pruned_bias_sequence = []

    # Change cfg_list & pruned_shape_info
    """
    for ind, (original_num_kernel, pruned_num_kernel) in enumerate(pruned_shape_info):
        #diff_num_kernel = original_num_kernel - pruned_num_kernel
        #new_pruned_num_kernel = pruned_num_kernel + int(diff_num_kernel*dP_rate)
        sub_pruned_shape_info.append(pruned_num_kernel)
        if ind == 0:
            new_pruned_num_kernel = original_num_kernel - int(original_num_kernel*dP_rate/2)
        else:
            new_pruned_num_kernel = original_num_kernel - int(original_num_kernel*dP_rate)
        if pruned_num_kernel >= new_pruned_num_kernel:
            print(" !!ERROR!! Inserted de_pruned shape is smaller than original.\n ")
            raise ValueError
        new_pruned_shape_info.append((original_num_kernel, new_pruned_num_kernel))
        for ind, shape in enumerate(new_cfg_list):
            if shape == pruned_num_kernel:
                new_cfg_list[ind] = new_pruned_num_kernel
    """
    print(new_pruned_shape_info, new_cfg_list)

    # Generate masks & pruned_weights
    total_conv_num = count_specific_layer(model, 'Conv')
    total_fc_num = count_specific_layer(model, 'Linear')
    layer_ind = -1
    fc_layer_ind = -1
    for module in model.modules():
        temp_cnt += 1
        if type(module).__name__ == 'MaskedConv2d':
            layer_ind += 1
            # Decide shape of mask
            if layer_ind == 0:
                # For Weight de-pruning
                mask_shape = (new_pruned_shape_info[layer_ind][-1], module.weight.shape[1], module.weight.shape[2], module.weight.shape[3])
                mask = torch.zeros(mask_shape)
                #kyuu temp
                print(temp_features[temp_cnt])
                pruned_weights = temp_features[temp_cnt].weight.cpu().data.numpy()
                #pruned_weights =\
                #np.random.normal(0, STD, mask_shape[0]*mask_shape[1]*mask_shape[2]*mask_shape[3]).reshape(\
                #    mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3])
                print("mask_shape ", mask_shape)
                print("module.weight.shape[0]", module.weight.shape[0])
                for kernel_ind in range(module.weight.shape[0]):
                    mask[kernel_ind,:,:,:] = torch.ones(mask[kernel_ind,:,:,:].shape)
                    pruned_weights[kernel_ind,:,:,:] = module.weight.cpu().data.numpy()[kernel_ind,:,:,:]

                new_w_sum += mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3]
                w_sum += new_pruned_shape_info[layer_ind][0] * module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]

            elif layer_ind == total_conv_num - 1 and (not with_fc):
                mask_shape = (module.weight.shape[0], new_pruned_shape_info[layer_ind-1][-1], module.weight.shape[2], module.weight.shape[3])
                mask = torch.zeros(mask_shape)
                #kyuu temp
                pruned_weights = temp_features[temp_cnt].weight.cpu().data.numpy()
                #pruned_weights = \
                #np.random.normal(0, STD, mask_shape[0]*mask_shape[1]*mask_shape[2]*mask_shape[3]).reshape(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3])
                for channel_ind in range(module.weight.shape[1]):
                    mask[:,channel_ind,:,:] = torch.ones(mask[:,channel_ind,:,:].shape)
                    pruned_weights[:,channel_ind,:,:] = module.weight.cpu().data.numpy()[:,channel_ind,:,:]
                new_w_sum += mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3]
                w_sum += new_pruned_shape_info[layer_ind][0] * new_pruned_shape_info[layer_ind-1][0] * module.weight.shape[2] * module.weight.shape[3]

            else:
                mask_shape = (new_pruned_shape_info[layer_ind][-1], new_pruned_shape_info[layer_ind-1][-1], module.weight.shape[2], module.weight.shape[3])
                mask = torch.zeros(mask_shape)
                #kyuu temp
                pruned_weights = temp_features[temp_cnt].weight.cpu().data.numpy()
                #pruned_weights =\
                #np.random.normal(0, STD, mask_shape[0]*mask_shape[1]*mask_shape[2]*mask_shape[3]).reshape(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3])
                for channel_ind in range(module.weight.shape[1]):
                    mask[:module.weight.shape[0],channel_ind,:,:] = 1
                    pruned_weights[:module.weight.shape[0],channel_ind,:,:] = module.weight.cpu().data.numpy()[:module.weight.shape[0],channel_ind,:,:]
                new_w_sum += mask_shape[0] * mask_shape[1] * mask_shape[2] * mask_shape[3]
                w_sum += new_pruned_shape_info[layer_ind][0] * new_pruned_shape_info[layer_ind-1][0] * module.weight.shape[2] * module.weight.shape[3]

            num_filter = mask_shape[0]
            num_filter_pruned = module.weight.shape[0]
            #kyuu temp
            pruned_bias = temp_features[temp_cnt].bias.cpu().data.numpy()
            #pruned_bias = np.random.normal(0, STD, mask_shape[0])
            pruned_bias[:module.weight.shape[0]] = module.bias.cpu().data.numpy()

            pruned_weights_sequence.append(pruned_weights)
            #pruned_bias_sequence.append(module.bias.cpu().data.numpy())
            pruned_bias_sequence.append(pruned_bias) # Initialization problem What is better?
            masks.append(mask)

        if type(module).__name__ == 'MaskedLinear':
            fc_layer_ind += 1
            fc_shape = module.weight.shape
            #mask_shape = (fc_shape[0], new_pruned_shape_info[total_conv_num+fc_layer_ind-1][-1])
            mask_shape = (new_pruned_shape_info[total_conv_num+fc_layer_ind][-1], new_pruned_shape_info[total_conv_num+fc_layer_ind-1][-1])

            #TODO Revise if node pruning added
            if with_fc:
                mask = torch.zeros(mask_shape)
                if total_fc_num == 1:
                    pruned_fc_weights = temp_classifier.weight.cpu().data.numpy()
                    pruned_bias = temp_classifier.bias.cpu().data.numpy()
                else:
                    pruned_fc_weights = temp_classifier[fc_layer_ind].weight.cpu().data.numpy()
                    pruned_bias = temp_classifier[fc_layer_ind].bias.cpu().data.numpy()

                for node_ind in range(fc_shape[0]):
                    mask[node_ind][:fc_shape[1]] = torch.ones(fc_shape[1])
                    pruned_fc_weights[node_ind][:fc_shape[1]] = module.weight.cpu().data.numpy()[node_ind]

                pruned_weights_sequence.append(pruned_fc_weights)

                pruned_bias[:module.weight.shape[0]] = module.bias.cpu().data.numpy()
                pruned_bias_sequence.append(pruned_bias)
                masks.append(mask)
            else:
                pruned_weights_sequence.append(module.weight.cpu().data.numpy())
                pruned_bias = np.random.normal(0, STD, mask_shape[0])
                pruned_bias[:module.weight.shape[0]] = module.bias.cpu().data.numpy()
                pruned_bias_sequence.append(pruned_bias)

        if type(module).__name__ == 'BatchNorm2d':
            # TODO Test for fixing BatchNorm2d
            #mask_shape = num_filter
            #mask_bn = torch.zeros(mask_shape)
            #mask_bn[:num_filter_pruned] = 1


            pruned_weights = temp_features[temp_cnt].weight.cpu().data.numpy()
            pruned_running_var = temp_features[temp_cnt].running_var.cpu().data.numpy()
            pruned_running_mean = temp_features[temp_cnt].running_mean.cpu().data.numpy()
            pruned_bias = temp_features[temp_cnt].bias.cpu().data.numpy()
            #pruned_weights = np.random.normal(0, STD, mask_shape[0])
            #pruned_running_var = np.random.normal(0, STD, mask_shape[0])
            #pruned_running_mean = np.random.normal(0, STD, mask_shape[0])
            #pruned_bias = np.random.normal(0, STD, mask_shape[0])

            pruned_weights[:module.weight.shape[0]] = module.weight.cpu().data.numpy()
            pruned_running_var[:module.running_var.shape[0]] = module.running_var.cpu().data.numpy()
            pruned_running_mean[:module.running_mean.shape[0]] = module.running_mean.cpu().data.numpy()
            pruned_bias[:module.bias.shape[0]] = module.bias.cpu().data.numpy()

            pruned_weights_sequence.append(pruned_weights)
            pruned_weights_sequence.append(pruned_running_var)
            pruned_weights_sequence.append(pruned_running_mean)
            pruned_bias_sequence.append(pruned_bias)
            """
            pruned_weights_sequence.append(module.weight.cpu().data.numpy())
            pruned_weights_sequence.append(module.running_var.cpu().data.numpy())
            pruned_weights_sequence.append(module.running_mean.cpu().data.numpy())
            pruned_bias_sequence.append(module.bias.cpu().data.numpy())
            """
            #TODO Test for fix BN For Weight bias of BN Layer
            #masks.append(mask_bn)
            #masks.append(mask_bn)


    new_pruned_rate = 100.0*(1-new_w_sum/w_sum)
    print("Total Pruned Percentage : {}.3%".format(new_pruned_rate))

    # Change pruned_net_info for new Multi-phased information
    pruned_net_info['cfg_list'] = new_cfg_list
    pruned_net_info['pruned_shape_info'] = new_pruned_shape_info
    pruned_net_info['pruned_rate'] = new_pruned_rate
    pruned_net_info['sub_pruned_shape_info'] = sub_pruned_shape_info

    return masks, pruned_weights_sequence, pruned_bias_sequence, pruned_net_info

def layer_prune(model, pruning_perc):
    #Calc Threshold per layer and precede pruning
    first_layer_flag = 0; masks = []
    for ind, p in enumerate(model.parameters()):
        #print(str(ind)+'th req_grad = '+str(p.requires_grad)+' '+str(p.data.size()))
        if len(p.data.size()) != 1:
            print(' * Weight Pruning Per layer : ' + str(ind)+'th parameters prunning...')
            if first_layer_flag == 0 :
                threshold = np.percentile(p.cpu().data.abs().numpy().flatten(), pruning_perc)
                first_layer_flag = 1
            # Delete zeros
            #p_non_zero = np.trim_zeros(p.cpu().data.abs().numpy().flatten())
            elif len(p.data.size()) == 2 :
                threshold = np.percentile(p.cpu().data.abs().numpy().flatten(), 0)-1
            else : threshold = np.percentile(p.cpu().data.abs().numpy().flatten(), pruning_perc)

        # Generate mask
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind,
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks
