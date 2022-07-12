import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.optim.lr_scheduler import StepLR

# Device configuration
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def sqzing_model(model_original, model, model_original_ckpt, sqz_filter_list):
    #model_original.load_state_dict(torch.load('./model.ckpt'))
    model_original.load_state_dict(torch.load(model_original_ckpt))
    layer_cnt = 0
    layer_weight_list = []
    for m in model_original.modules():
        if isinstance(m,nn.Conv2d):
            layer_weight_list.append(m.weight.data.numpy())
            layer_cnt += 1

    print('the number of layers : '+str(len(layer_weight_list)))

    layer_cnt = 0
    layer_weight_list_pruned = []
    for i in range(len(layer_weight_list)):
        shape1 = np.shape(layer_weight_list[i])
        weight_1 = []
        target_filters = sqz_filter_list[i]
        for j in range(shape1[0]):
            weight_2 = []
            if j in target_filters:
                print('{}_layer {}_filter '.format(i,j))
                pass
            else:
                for k in range(shape1[1]):
                    weight_3 = []
                    if i == 0:
                        target_channels = []
                    else :
                        target_channels = sqz_filter_list[i-1]
                    if k in target_channels:
                        print('{}_layer {}_filter {}_channel'.format(i,j,k))
                        pass
                    else:
                        for l in range(shape1[2]):
                            weight_4 = []
                            for m in range(shape1[3]):
                                t_array = layer_weight_list[i]
                                weight_4.append(t_array[j][k][l][m])
                            weight_3.append(weight_4)
                        weight_2.append(weight_3)
                weight_1.append(weight_2)
        layer_weight_list_pruned.append(weight_1)

    shape2 = np.array(layer_weight_list_pruned).shape
    print('the shape of squeezed model : '+str(shape2))
    layer_cnt = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            for i in range(m.weight.shape[0]):
                for j in range(m.weight.shape[1]):
                    for k in range(m.weight.shape[2]):
                        m.weight.data[i][j][k] = torch.from_numpy(layer_weight_list_pruned[layer_cnt][i][j][k])
            layer_cnt += 1

    for i,m in enumerate(model.modules()):
        if isinstance(m,nn.Conv2d):
            print('the shape of each squeezed layers [layer_{}] : {} '.format(i,m.weight.data.shape))

    return model

def masking(in_list,mean):
    output = []
    for data in in_list:
        if data > mean*1.1:
            output.append('H')
        elif data < mean*0.8:
            output.append('O')
        elif data < mean*0.9:
            output.append('O')
        else:
            output.append('-')
    return output

def find_target_filter(model):
    f = open('target_filter_finding_log.txt','w') 
    filter_lists = []
    channel_lists_list = []
    for i, m in enumerate(model.modules()):
            if isinstance(m,nn.Conv2d):
                filter_list = [] 
                channel_lists = []
    #           print("Conv2d_{} : {}".format(i,m.weight.shape))
                f.write("Conv2d_{} : {}\n".format(i,m.weight.shape))
                for k in range(m.weight.shape[0]):
    #               print("Conv2d_{} : {}".format(i,m.weight[k]))
                    norm = round(float(m.weight[k].norm(2)),3)
    #               print("Conv2d_{} filter[{}] 2nd norm: {}".format(i,k,norm))
                    f.write("Conv2d_{} filter[{}] 2nd norm: {}\n".format(i,k,norm))
                    filter_list.append(norm)
                    channel_list = [] 
                    for j in range(m.weight.shape[1]):
                        norm_c = round(float(m.weight[k][j].norm(2)),3)
    #                   print("Conv2d_{} filter[{}] channel[{}] 2nd norm: {}".format(i,k,j,norm_c))
                        f.write("Conv2d_{} filter[{}] channel[{}] 2nd norm: {}\n".format(i,k,j,norm_c))
                        channel_list.append(norm_c)
                    channel_lists.append(channel_list)
                filter_lists.append(filter_list)
                channel_lists_list.append(channel_lists)
            elif isinstance(m,nn.Linear):
                pass
    #           print("Linear_{} : {}".format(i,m.weight.shape))

    mask_list = [[]]
    for i, filter_ in enumerate(filter_lists):
        filter_array = np.array(filter_)
        mean_t = filter_array.mean()
        mask = masking(filter_,mean_t)
        mask_list.append(mask)
        print("[{}] {}".format(i,filter_))
        print("[{}] {}".format(i,mask))
        f.write("[{}] {}\n".format(i,filter_))
        f.write("[{}] {}\n".format(i,mask))
    for i, channel_lists_ in enumerate(channel_lists_list):
        print("[{}/M] {}".format(i,mask_list[i]))
        f.write("[{}/M] {}\n".format(i,mask_list[i]))
        for j, channel_list in enumerate(channel_lists_):
            channel_list_ = np.array(channel_list)
            mean_t = channel_list_.mean()
            mask = masking(channel_list,mean_t)
    #        print("[{}/{}] {} mean : {}".format(i,j,channel_list,mean_t))
            print("[{}/{}] {}".format(i,j,mask))
            f.write("[{}/{}] {}\n".format(i,j,mask))
    print("mask_list") 
    print(mask_list) 
    return mask_list
    #torch.save(model.state_dict(), 'model_jjg_pruned2.ckpt')
    #'''
