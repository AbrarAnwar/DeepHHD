import torch.nn as nn
import torch
import numpy as np 

def conv2d(batchNorm, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None):
    if padding == None:
        padding = (kernel_size-1)//2
    if batchNorm:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

