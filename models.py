import torch.nn as nn
import torch
import numpy as np
from correlation_package.correlation import Correlation
from submodules import *

class SmallBlock(nn.Module):
    def __init__(self, num_conv, input_size, output_size):
        super(SmallBlock, self).__init__()
        modules = []
        size = input_size
        for _ in range(num_conv):
            modules.append(conv2d(False, size, output_size, kernel_size=3, stride=1))
            size = output_size
        self.sequential = nn.Sequential(*modules).cuda()
    def forward(self,x):
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, corr_size, height, width, repeat_num, batchNorm=True):
        super(Generator, self).__init__()

        self.x0_shape = [int(i/np.power(2, repeat_num-1)) for i in [height, width]] + [128]
        #print('x0 shape', self.x0_shape)

        #self.project = nn.Linear(corr_size, int(np.prod(self.x0_shape)))
        self.repeat_num = repeat_num

        self.first_conv = conv2d(False, 473, 128, kernel_size=3, stride=1)

        self.small_blocks = []
        for idx in range(self.repeat_num):
            self.small_blocks.append(SmallBlock(4, 128, 128))

        self.last_conv = conv2d(False, 128, 2, kernel_size=3, stride=1)
        self.width = width
        self.height = height

    def forward(self,x):

        #x = self.project(x)
        #print('project', x.shape)

        # we want to reshape h,w,d h/2^q, etc
        #x = x.reshape(-1, self.x0_shape[0], self.x0_shape[1], self.x0_shape[2])
        #x = x.reshape(-1, 3072, 55, 128)

        x = self.first_conv(x)
        x0 = x
        #print('first conv', x.shape)
        #print('repeat num', self.repeat_num)

        for idx in range(self.repeat_num):
            x = self.small_blocks[idx](x) 
            #print('sb', idx, x.shape)

            if idx < self.repeat_num - 1:
                x = x + x0
                x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
                x0 = x
            else:
                x = x + x0

        x = self.last_conv(x)
        x = nn.functional.interpolate(x, size=(self.width, self.height), mode='nearest')
        return x

        

        

class HHDFlow(nn.Module):
    def __init__(self, height, width, batchNorm=True):
        super(HHDFlow,self).__init__()

        self.batchNorm = batchNorm

        self.conv1 = conv2d(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv2d(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv2d(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv2d(self.batchNorm, 256, 32, kernel_size=1, stride=1)
        
        self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_act = nn.LeakyReLU(0.1,inplace=True)

        #repeat_num = int(np.log2(max(height, width))) - 2
        repeat_num = 4

        self.gen1 = Generator(441*55*128, height, width, repeat_num, False)
        self.gen2 = Generator(441*55*128, height, width, repeat_num, False)
        self.gen3 = Generator(441*55*128, height, width, repeat_num, False)

    def forward(self, x1, x2):
        # Send image through image towers to get reduced representations
        #print('x1,x2', x1.shape)
        conv1a = self.conv1(x1)
        conv2a = self.conv2(conv1a)
        conv3a = self.conv3(conv2a)


        conv1b = self.conv1(x2)
        conv2b = self.conv2(conv1b)
        conv3b = self.conv3(conv2b)

        out_conv_redir = self.conv_redir(conv3a)
        #print('1', conv1a.shape)
        #print('2', conv2a.shape)
        #print('3', conv3a.shape)
        #print('conv_redir', out_conv_redir.shape)

        # Merge the image representations together using a correlation layer
        out_corr = self.corr(conv3a, conv3b)
        out_corr = self.corr_act(out_corr)
        #print('corr',out_corr.shape)
        #out_corr = out_corr.reshape(-1, out_corr.shape[2], out_corr.shape[3], out_corr.shape[1])
        #print('corr',out_corr.shape)

        out_corr = torch.cat([out_conv_redir, out_corr], dim=1)
        #print('concat', out_corr.shape)




        # Now we want to do generative things
        r = self.gen1(out_corr)
        h = self.gen2(out_corr)
        d = self.gen3(out_corr)

        out = r + h + d

        return out, r, d, h



