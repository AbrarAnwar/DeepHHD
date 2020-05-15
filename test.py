from util import *
from models import *
from losses import *
from datasets import *

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from pynhhd import nHHD



def test(save):

    model = HHDFlow(1024, 436).cuda()
    print('Loading Model')
    model.load_state_dict(torch.load('saves/' + save + '.pt'))
    model.eval()

    hhdloss = HHDLoss(nn.L1Loss().cuda()).cuda()
    l1loss = L1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = MPISintel('/home/abrar/flow/MPI-Sintel-complete/training/flow/', '/home/abrar/flow/MPI-Sintel-complete/training/clean/')
    # training time
    print('Dataset size:', len(dataset))
    epe_list = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            im1s = sample_batched[0][0].cuda()
            im2s = sample_batched[0][1].cuda()

            flos = sample_batched[1][0].cuda()

            #forward
            yhat, rhat, dhat, hhat = model(im1s,im2s)

            del im1s
            del im2s

            # backward and optmize
            rs = sample_batched[1][1].cuda()
            ds = sample_batched[1][2].cuda()
            hs = sample_batched[1][3].cuda()
            lossval = hhdloss(rhat,dhat,hhat, rs, ds, hs) + l1loss(yhat, flos.reshape(yhat.shape))

            epe = EPE(yhat, flos.reshape(yhat.shape)).item()
            epe_list.append(epe)
            #np.savez('images.npz', np.array(yhat.cpu()), np.array(rhat.cpu()), np.array(dhat.cpu()), np.array(hhat.cpu()))
            #exit(0)
            print('EPE: {:0>10.7f} \t\t loss: {:0>10.7f},\t AvgEPE: {:0>10.7f}\t Iteration: {}'.format(epe, lossval.item(), np.mean(epe_list), i_batch))

def main(args):
    test(args[1])

if __name__ == '__main__':
    main(sys.argv)
