import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bridge(nn.Module):
    expansion = 1

    def __init__(self, depth, inchannels, outchannels, stride=1):
        super(Bridge, self).__init__()
        self.downsample = nn.Conv3d(inchannels, outchannels, kernel_size = 1) if inchannels != outchannels else None
        self.direct = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv3d(outchannels if i else inchannels, outchannels, kernel_size = 3, stride = stride, padding = 1),
                    nn.BatchNorm3d(outchannels),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                )
                for i in range(depth-1)
            ],
            nn.Conv3d(outchannels, outchannels, kernel_size = 3, stride = stride, padding = 1),
        )
        self.end = nn.Sequential(
            # nn.BatchNorm3d(outchannels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.direct(x)
        out = out + x if self.downsample == None else self.downsample(x)
        out = self.end(out)
        return out

class ResNet3d(nn.Module):
    def __init__(self, depths, channels):
        super(ResNet3d, self).__init__()
        assert(len(depths) == len(channels)-1)
        layers = []
        for depth, prev, curr in zip(depths, channels[:-1], channels[1:]):
            bridge = Bridge(depth, prev, curr)
            layers.append(bridge)
            prev = curr
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def resnet(depths, channels):
    return ResNet3d(depths, channels)


