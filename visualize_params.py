import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import torch
import numpy as np
import argparse
from pruning.utils import update_params, count_specific_layer

def draw_params(net, filename = None):
    sns.set()
    num_cols = 3
    num_conv = count_specific_layer(net, 'Conv')
    num_fc = count_specific_layer(net, 'Linear')
    total_num_layer = num_conv + num_fc
    f, axis = plt.subplots(total_num_layer//num_cols, num_cols, figsize=(21,14))
    print(axis)
    cmap = ListedColormap(['y'])
    layer_count = 0
    for m in net.modules():
        if type(m).__name__ == 'MaskedConv2d':
            shape = m.weight.shape
            assert len(shape) == 4
            params = m.weight.detach().abs().numpy()
            params = params.reshape(shape[0], shape[1]*shape[2]*shape[3])
            sns.heatmap(params, fmt='f', ax = axis[layer_count//num_cols][layer_count%num_cols], xticklabels=False, yticklabels=False)
            sns.heatmap(params, fmt='f', mask=params != 0.0, cmap=cmap,
                ax=axis[layer_count//num_cols][layer_count%num_cols], cbar=False, xticklabels=False, yticklabels=False)
            layer_count += 1
        elif type(m).__name__ == 'MaskedLinear':
            shape = m.weight.shape
            assert len(shape) == 2
            params = m.weight.detach().abs().numpy()
            sns.heatmap(params, fmt='f', ax=axis[layer_count//num_cols][layer_count%num_cols], xticklabels=False, yticklabels=False)
            sns.heatmap(params, fmt='f', mask=params != 0.0, cmap=cmap,
                ax=axis[layer_count//num_cols][layer_count%num_cols], cbar=False, xticklabels=False, yticklabels=False)
            layer_count += 1

    plt.show()

    if filename != None:
        plt.savefig('./imgs/' + filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Draw Parameters from selected CNN Architecture')
    parser.add_argument('--Model', type=str, default=None, help = 'Model Architecure Name')
    parser.add_argument('--M_loc', type=str, default=None, help = 'Filename of pkl model file')
    args = parser.parse_args()

    ldict = locals()
    code = compile('net = ' + args.Model + '()', '<string>', 'single')
    exec('from models import ' + args.Model)
    exec(code, globals(), ldict)
    net = ldict['net']
    update_params(net, './models/' + args.M_loc + '.pkl')

    draw_params(net, args.M_loc)
