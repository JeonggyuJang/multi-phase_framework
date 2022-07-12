import torch
import torch.nn as nn
import torch.nn.functional as F
from pruning.layers import MaskedLinear, MaskedConv2d
import torch.nn.init as init
import numpy as np
#from lenet import *

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])


class ConvNet_prun_conv(nn.Module):
    def __init__(self,convnet_num_classes=10):
        super(ConvNet_prun_conv, self).__init__()
        self.conv1 = MaskedConv2d(3, 32, kernel_size=5, padding=0, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = MaskedConv2d(32, 64, kernel_size=5, padding=2, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = MaskedConv2d(64, 96, kernel_size=3, padding=0, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv4 = MaskedConv2d(96, 128, kernel_size=3, padding=0, stride=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        '''
        self.layer1 = nn.Sequential(
                self.conv1,
                nn.BatchNorm2d(32),
                nn.ReLU()
                #nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
                self.conv2,
                nn.BatchNorm2d(64),
                n.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
                self.conv3,
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
                self.conv4,
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''
        self.fc1 = nn.Linear(128*2*2, 2*2*26)
        self.fc2 = nn.Linear(26*2*2, convnet_num_classes)

    def forward(self, x):
        out = self.relu1(self.batchnorm1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.batchnorm2(self.conv2(out))))
        out = self.maxpool3(self.relu3(self.batchnorm3(self.conv3(out))))
        out = self.maxpool4(self.relu4(self.batchnorm4(self.conv4(out))))
        out = out.view(out.size(0),-1)
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        '''
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])

class ConvNet(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-2:]
                self.features = self._make_layers(cfg_kc[:-2])
                self.classifier = nn.Sequential(*[MaskedLinear(self.fc_shape[0][1], self.fc_shape[0][0]),\
                        MaskedLinear(self.fc_shape[1][1], self.fc_shape[1][0])])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = nn.Sequential(*[MaskedLinear(256, 64),\
                    MaskedLinear(64, self.num_classes)])
        else:
            self.features = nn.Sequential(*[MaskedConv2d(3, 48, kernel_size=5, padding=0, stride=1),\
                nn.BatchNorm2d(48), nn.ReLU(inplace=True),\
                MaskedConv2d(48, 96, kernel_size=5, padding=2, stride=1),\
                nn.BatchNorm2d(96), nn.ReLU(inplace=True),\
                nn.MaxPool2d(kernel_size=2,stride=2),\
                MaskedConv2d(96, 192, kernel_size=3, padding=0, stride=1),\
                nn.BatchNorm2d(192), nn.ReLU(inplace=True),\
                nn.MaxPool2d(kernel_size=2,stride=2),\
                MaskedConv2d(192, 256, kernel_size=3, padding=0, stride=1),\
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),\
                nn.MaxPool2d(kernel_size=2,stride=2),\
                nn.AvgPool2d(kernel_size=2, stride=2)])
            self.classifier = nn.Sequential(*[MaskedLinear(256, 64),\
                MaskedLinear(64, self.num_classes)])

    def forward(self, x):
        out = self.features(x)
        """
        out = self.features[0](x)
        for i in range(1, 16): # Relu 2, 5, 9, 13
            out = self.features[i](out)
            if i==9:
                data = out.cpu().detach().numpy()
                shape = data.shape
                total = shape[0]*shape[1]*shape[2]*shape[3]
                nonzeros = np.count_nonzero(data)
                print("Layer #{}, nonzeros = {}%".format(i,str(nonzeros/total*100)))
        """
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        conv_cnt = 0
        maxpool_cnt = 0
        for x in cfg:
            if x == 'M':
                if maxpool_cnt == 2:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.AvgPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                maxpool_cnt += 1
            else:
                if conv_cnt == 0:
                    layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=0, stride=1),\
				nn.BatchNorm2d(x),
				nn.ReLU(inplace=True)]
                elif conv_cnt == 1:
                    layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=2, stride=1),\
				nn.BatchNorm2d(x),
				nn.ReLU(inplace=True)]
                else:
                    layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=0, stride=1),\
				nn.BatchNorm2d(x),
				nn.ReLU(inplace=True)]
                in_channels = x
                conv_cnt += 1
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'gpu':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class ConvNet_WO_BN(nn.Module):
    def __init__(self,convnet_num_classes=10):
        super(ConvNet_WO_BN, self).__init__()
        self.conv1 = MaskedConv2d(3, 48, kernel_size=5, padding=0, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = MaskedConv2d(48, 96, kernel_size=5, padding=2, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv3 = MaskedConv2d(96, 192, kernel_size=3, padding=0, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv4 = MaskedConv2d(192, 256, kernel_size=3, padding=0, stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = MaskedLinear(256*2*2, 2*2*64)
        self.fc2 = MaskedLinear(64*2*2, convnet_num_classes)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.maxpool3(self.relu3(self.conv3(out)))
        out = self.maxpool4(self.relu4(self.conv4(out)))
        out = out.view(out.size(0),-1)
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        '''
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])
        self.fc1.set_mask(masks[4])
        self.fc2.set_mask(masks[5])



class Fire(nn.Module):
    def __init__(self, in_ch, sqz_ch, exp_1x1_ch, exp_3x3_ch):
        super(Fire, self).__init__()
        self.in_ch = in_ch
        self.sqz = MaskedConv2d(in_ch, sqz_ch, kernel_size = 1, stride=1, padding=0)
        self.bn_sqz = nn.BatchNorm2d(sqz_ch)
        self.sqz_act = nn.ReLU(inplace=True)
        self.exp_1x1 = MaskedConv2d(sqz_ch, exp_1x1_ch, kernel_size=1, stride=1, padding=0)
        self.bn_e1 = nn.BatchNorm2d(exp_1x1_ch)
        self.exp_1x1_act = nn.ReLU(inplace=True)
        self.exp_3x3 = MaskedConv2d(sqz_ch, exp_3x3_ch, kernel_size=3, stride=1, padding=1)
        self.bn_e2 = nn.BatchNorm2d(exp_3x3_ch)
        self.exp_3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        sqz_out = self.sqz_act(self.bn_sqz(self.sqz(x)))
        fire_out = torch.cat([
                                self.exp_1x1_act(self.bn_e1(self.exp_1x1(sqz_out))),
                                self.exp_3x3_act(self.bn_e2(self.exp_3x3(sqz_out)))
                                ], 1)
        return fire_out

class SqzNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SqzNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
                                        MaskedConv2d(3, 96, kernel_size=3, stride =1, padding=1),
                                        nn.BatchNorm2d(96),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True),
                                        Fire(96, 16, 64, 64),
                                        Fire(128, 16,64, 64),
                                        Fire(128, 32, 128, 128),
                                        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                        Fire(256, 32, 128, 128),
                                        Fire(256, 48, 192, 192),
                                        Fire(384, 48, 192, 192),
                                        Fire(384, 64, 256, 256),
                                        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True),
                                        Fire(512, 64, 256, 256)
                                        )
        self.final_conv = MaskedConv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
                                        nn.Dropout(p=0.5),
                                        self.final_conv,
                                        nn.ReLU(inplace=True),
                                        nn.AvgPool2d(4, stride=1)
                                        )
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                if m is self.final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.set_mask(masks[cnt])
                cnt+=1

cfg = {
        'VGG7' : [64, 'M', 128, 'M', 256, 256, 'M'],
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG11_48': [48, 'M', 96, 'M', 192, 192, 'M', 384, 384, 'M', 384, 384, 'M'],
        'VGG11_32': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
        'VGG11_16': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
        'VGG11_FAT': [64, 'M', 128, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        'ALEX' : [32, 'M', 32, 'A', 64, 'A'],
        'ALEX24' : [24, 'M', 24, 'A', 48, 'A'],
        'ALEX16' : [16, 'M', 16, 'A', 32, 'A'],
        'ALEX8' : [8, 'M', 8, 'A', 16, 'A']
        }

class ALEXnet(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ALEXnet, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(64, self.num_classes)
        else:
            self.features = self._make_layers(cfg['ALEX'])
            self.classifier = MaskedLinear(64, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=1),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class ALEXnet24(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ALEXnet24, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(48, self.num_classes)
        else:
            self.features = self._make_layers(cfg['ALEX24'])
            self.classifier = MaskedLinear(48, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=1),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class ALEXnet16(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ALEXnet16, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(32, self.num_classes)
        else:
            self.features = self._make_layers(cfg['ALEX16'])
            self.classifier = MaskedLinear(32, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=1),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError
class ALEXnet8(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ALEXnet8, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(16, self.num_classes)
        else:
            self.features = self._make_layers(cfg['ALEX8'])
            self.classifier = MaskedLinear(16, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=5, padding=1),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class VGG7(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG7, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(256, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG7'])
            self.classifier = MaskedLinear(256, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class VGG11(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11'])
            self.classifier = MaskedLinear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class VGG11_48(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_48, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(384, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11_48'])
            self.classifier = MaskedLinear(384, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError
class VGG11_32(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_32, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(256, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11_32'])
            self.classifier = MaskedLinear(256, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError
class VGG11_16(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_16, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(128, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11_16'])
            self.classifier = MaskedLinear(128, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError
class VGG11_CO(nn.Module):
    """
    Coarse-only Golden Model. Same with VGG11(), but for Management.
    """
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_CO, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11'])
            self.classifier = MaskedLinear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else:
            print("Error! No Pruning Method!")
            raise ValueError

class VGG11_WO_BN(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_WO_BN, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11'])
            self.classifier = MaskedLinear(512, self.num_classes)

        #if cfg_kc is not None:
        #    if with_fc:
        #        self.fc_out = cfg_kc.pop()
        #        self.features = self._make_layers(cfg_kc)
        #else:
        #    self.features = self._make_layers(cfg['VGG11'])

        #if with_fc:
        #    self.classifier = MaskedLinear(self.fc_out, self.num_classes)
        #    cfg_kc.append(self.fc_out)
        #else:
        #    self.classifier = MaskedLinear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc = False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd_scalpel':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG11_FC(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG11_FC, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-3:]
                self.features = self._make_layers(cfg_kc[:-3])
                self.classifier = nn.Sequential(*[MaskedLinear(self.fc_shape[0][1], self.fc_shape[0][0]),\
                                                MaskedLinear(self.fc_shape[1][1], self.fc_shape[1][0]),\
                                                MaskedLinear(self.fc_shape[2][1], self.fc_shape[2][0])
                                                ])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = nn.Sequential(*[MaskedLinear(512, 1024),\
                                                MaskedLinear(1024, 512),\
                                                MaskedLinear(512, self.num_classes)
                                                ])
        else:
            self.features = self._make_layers(cfg['VGG11'])
            self.classifier = nn.Sequential(*[MaskedLinear(512, 1024),\
                                            MaskedLinear(1024, 512),\
                                            MaskedLinear(512, self.num_classes)
                                            ])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, prune_fc = False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and prune_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG11_CIFAR_BIG(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=100, with_fc=False):
        super(VGG11_CIFAR_BIG, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG11'])
            self.classifier = MaskedLinear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG16(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG16'])
            self.classifier = MaskedLinear(512, self.num_classes)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.set_mask(masks[cnt], multi_phase_opt)
                print(cnt)
                cnt+=1
            elif isinstance(m, MaskedLinear):
                m.set_mask(masks[cnt], multi_phase_opt)
                print(cnt+10000)
                cnt+=1

class VGG16_CIFAR_BIG(nn.Module):
    def __init__(self, cfg_kc=None, num_classes = 100, with_fc=False):
        super(VGG16_CIFAR_BIG, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = MaskedLinear(self.fc_shape[1], self.fc_shape[0])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = MaskedLinear(512, self.num_classes)
        else:
            self.features = self._make_layers(cfg['VGG16'])
            self.classifier = MaskedLinear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG16L_CIFAR_BIG(nn.Module):
    def __init__(self, cfg_kc=None, num_classes = 100, with_fc=False):
        super(VGG16L_CIFAR_BIG, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-2:]
                self.features = self._make_layers(cfg_kc[:-2])
                self.classifier = nn.Sequential(*[MaskedLinear(self.fc_shape[0][1], self.fc_shape[0][0]),\
                        nn.ReLU(inplace=True),
                        MaskedLinear(self.fc_shape[1][1], self.fc_shape[1][0])
                        ])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = nn.Sequential(*[MaskedLinear(512, 512),\
                        MaskedLinear(512, self.num_classes)
                        ])
        else:
            self.features = self._make_layers(cfg['VGG16'])
            self.classifier = nn.Sequential(*[MaskedLinear(512, 512),\
                    nn.ReLU(inplace=True),
                    MaskedLinear(512, self.num_classes)
                    ])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for ind, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'gpu':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG16_BN_IMAGE(nn.Module):
    def __init__(self, cfg_kc = None, num_classes = 1000, with_fc=False):
        super(VGG16_BN_IMAGE, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-1:]
                self.features = self._make_layers(cfg_kc[:-1])
                self.classifier = nn.Sequential(*[MaskedLinear(self.fc_shape[0][1], self.fc_shape[0][0]),\
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, self.num_classes)
                        ])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = nn.Sequential(*[MaskedLinear(25088,4096),\
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, self.num_classes)
                        ])
        else:
            self.features = self._make_layers(cfg['VGG16'])
            self.classifier = nn.Sequential(*[MaskedLinear(25088, 4096),\
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    MaskedLinear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    MaskedLinear(4096, self.num_classes)
                    ])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for ind, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            ]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'gpu':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class VGG16_BN_CIFAR_BIG(nn.Module):
    def __init__(self, cfg_kc=None, num_classes = 100, with_fc=False):
        super(VGG16_BN_CIFAR_BIG, self).__init__()
        self.num_classes = num_classes
        if cfg_kc is not None:
            if with_fc:
                self.fc_shape = cfg_kc[-2:]
                self.features = self._make_layers(cfg_kc[:-2])
                self.classifier = nn.Sequential(*[nn.Dropout(0.5),\
                        MaskedLinear(self.fc_shape[0][1], self.fc_shape[0][0]),\
                        nn.ReLU(inplace=True), nn.Dropout(0.5),
                        MaskedLinear(self.fc_shape[1][1], self.fc_shape[1][0])
                        ])
            else:
                self.features = self._make_layers(cfg_kc)
                self.classifier = nn.Sequential(*[MaskedLinear(512, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        MaskedLinear(4096, self.num_classes)
                        ])
        else:
            self.features = self._make_layers(cfg['VGG16'])
            self.classifier = nn.Sequential(*[MaskedLinear(512, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    MaskedLinear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    MaskedLinear(4096, self.num_classes)
                    ])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for ind, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [MaskedConv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True),
                            ]
                in_channels = x
        return nn.Sequential(*layers)

    def set_masks(self, masks, pruning_method, with_fc=False):
        # Should be a less manual way to set masks
        # Leave it for the future
        cnt=0
        if pruning_method == 'layer':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear):
                    m.set_mask(masks[cnt])
                    cnt+=1
        elif pruning_method == 'simd':
            for m in self.modules():
                if isinstance(m, MaskedConv2d):
                    m.set_mask(masks[cnt])
                    cnt+=1
                elif isinstance(m, MaskedLinear) and with_fc == True:
                    m.set_mask(masks[cnt])
                    cnt+=1
        else: print("Error! No Pruning Method!")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cfg_kc=None):
        super(BasicBlock, self).__init__()
        if(cfg_kc != None):
            self.conv1 = MaskedConv2d(in_planes, cfg_kc[0], kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cfg_kc[0])
            self.conv2 = MaskedConv2d(cfg_kc[0], cfg_kc[1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cfg_kc[1])
        else:
            self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, cfg_kc=None, num_classes=10, with_fc=False):
        super(ResNet18, self).__init__()
        self.block = BasicBlock
        self.num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if cfg_kc != None:
            self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, cfg_kc=cfg_kc[:4])
            self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, cfg_kc=cfg_kc[4:8])
            self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, cfg_kc=cfg_kc[8:12])
            self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, cfg_kc=cfg_kc[12:16])
        else:
            self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
            self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
            self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
            self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = MaskedLinear(512*self.block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, cfg_kc=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        offset = 0
        for stride in strides:
            if cfg_kc == None: layers.append(block(self.in_planes, planes, stride, cfg_kc=cfg_kc))
            else:
                layers.append(block(self.in_planes, planes, stride, cfg_kc=cfg_kc[offset: offset+2]))
                offset += 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
