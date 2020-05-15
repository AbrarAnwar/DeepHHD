import torch
import torch.nn as nn
import numpy as np

class HHDLoss(nn.Module):
    def __init__(self, loss):
        super(HHDLoss, self).__init__()
        self.loss = loss


    def forward(self, rhat, dhat, hhat, r, d, h):
        loss_val = self.loss(hhat, h)
        loss_val += self.loss(rhat, r)
        loss_val += self.loss(dhat, d)
        return loss_val

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

def EPE(input_flow, target_flow):
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()
