'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_activations import *

from torch.autograd import Variable

__all__ = ['Bottleneck_reswgt_both_act2','DPN_reswgt_both_act2','DPN26_reswgt_both_act2','DPN92_reswgt_both_act2','DPN_X1_reswgt_both_act2']

"""
This version of Dual Path Networks uses the `residual_weight` hyperparameter to 
weight BOTH the additive residual path, AND the concatenated one.
In this case we are keeping it simple by using the same weight for both,
but could theoretically expand this to use different weights for each.
"""


class Bottleneck_reswgt_both_act2(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, residual_weight=1.5, activation=F.relu):
        super(Bottleneck_reswgt_both_act2, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.residual_weight = residual_weight
        self.activation = activation

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )


    def forward(self, x):       
        
        # convolution 1
        out = self.activation(self.bn1(self.conv1(x)))

        x = self.shortcut(x)

        # convolution 2
        out = self.bn2(self.conv2(out))
        out = self.activation(out)

        d = self.out_planes

        # convolution 3
        out = self.bn3(self.conv3(out))
        
        # we multiply both the additive path AND the concatenated residual path
        # by our residual_weight hyperparameter
        out = torch.cat([(self.residual_weight*x[:,:d,:,:])+out[:,:d,:,:], self.residual_weight*x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = self.activation(out)

        return out


class DPN_reswgt_both_act2(nn.Module):
    def __init__(self, cfg, residual_weight=1.5, activation=F.relu):
        super(DPN_reswgt_both_act2, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.residual_weight = residual_weight
        self.activation = activation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64

        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck_reswgt_both_act2(self.last_planes, in_planes, out_planes, dense_depth, 
                stride, i==0, self.residual_weight, activation=self.activation))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DPN26_reswgt_both_act2(residual_weight=1.5,activation=F.relu):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN_reswgt_both_act2(cfg)

def DPN92_reswgt_both_act2(residual_weight=1.5,activation=F.relu):
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN_reswgt_both_act2(cfg)

def DPN_X1_reswgt_both_act2(residual_weight=1.5,activation=F.relu):
    cfg = {
        'in_planes': (96,192,384,768,768),
        'out_planes': (256,512,1024,2048,2048),
        'num_blocks': (3,4,20,3,3),
        'dense_depth': (16,32,24,128,128)
    }
    return DPN_reswgt_both_act2(cfg)


def test():
    net = DPN92_reswgt_both_act2()
    x = Variable(torch.randn(1,3,32,32))
    y = net(x)
    print(y)

# test()
