import torch
from models import ConvNet
import numpy as np

Model_name = "ConvNet"
ldict = locals()

def model_load(target_file):
    code = compile('net = ' + Model_name + '()', '<string>', 'single')
    exec(code, globals(), ldict)
    net = ldict['net']
    model_dict = net.state_dict()
    pretrained_dict = torch.load('models/' + Model_name + '_' + target_file + '.pkl')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def cal_percent_of_zero(net_params, total_parameter):
    zero_parameter = sum(np.count_nonzero(p.cpu().data.numpy()) for p in net_params)
    return (total_parameter - zero_parameter) / total_parameter * 100

def total_parameters(net):
    return sum(p.numel() for p in net.parameters())

def main(layer_num):
    # Model Loading
    pre_net = model_load('pretrained')
    #mul_net = model_load('multi_phased')
    pruned_net = model_load('simd_pruned_87%_sc')


    # Parameter Count
    num_parameters = total_parameters(pre_net)
    print("Total parameter number : #{}".format(num_parameters))
    pre_params = list(pre_net.parameters())
    mul_params= list(mul_net.parameters())
    pruned_params = list(pruned_net.parameters())

    print("Zero portion of parameters of network")
    print("    Pretrained_network : {:.2f}%".format(cal_percent_of_zero(pre_params, num_parameters)))
    print("    Pruned_network : {:.2f}%".format(cal_percent_of_zero(pruned_params, num_parameters)))
    print("    Multi_phased_network : {:.2f}%".format(cal_percent_of_zero(mul_params, num_parameters)))

    print("State of Network for specific layer")
    print("    Pretrained_network" + "_"*10)
    print(pre_params[layer_num][0][0])
    print("    Pruned_network" + "_"*10)
    print(pruned_params[layer_num][0][0])
    print("    Multi_phased_network" + "_"*10)
    print(mul_params[layer_num][0][0])

main(4)
