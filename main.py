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



def train(save):
    model = HHDFlow(1024, 436).cuda()
    hhdloss = HHDLoss(nn.L1Loss().cuda()).cuda()
    l1loss = L1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)


    dataset = MPISintel('/home/abrar/flow/MPI-Sintel-complete/training/flow/', '/home/abrar/flow/MPI-Sintel-complete/training/clean/')
    # training time
    print('Dataset size:', len(dataset))
    for epoch in range(50):
        epe_list = []
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        for i_batch, sample_batched in enumerate(dataloader):
            im1s = sample_batched[0][0].cuda()
            im2s = sample_batched[0][1].cuda()

            flos = sample_batched[1][0].cuda()
            # zero parameter grads
            optimizer.zero_grad()

            #forward
            yhat, rhat, dhat, hhat = model(im1s,im2s)

            del im1s
            del im2s

            # backward and optmize
            rs = sample_batched[1][1].cuda()
            ds = sample_batched[1][2].cuda()
            hs = sample_batched[1][3].cuda()
            lossval = hhdloss(rhat,dhat,hhat, rs, ds, hs) + l1loss(yhat, flos.reshape(yhat.shape))
            lossval.backward()
            optimizer.step()


            epe = EPE(yhat, flos.reshape(yhat.shape)).item()
            epe_list.append(epe)
            print('Epoch {} \t\t EPE: {:0>10.7f} \t\t loss: {:0>10.7f},\t AvgEPE: {:0>10.7f}\t Iteration: {}'.format(epoch, epe, lossval.item(), np.mean(epe_list), i_batch))
        # Save a checkpoint
        torch.save(model.state_dict(), 'saves/' + save + '_checkpoint{}.pt'.format(epoch))

    print('Saving Model')
    torch.save(model.state_dict(), 'saves/' + save + '.pt')


def main(args):
    if args[1] == 'train':
        train(args[2])



if __name__ == '__main__':
    main(sys.argv)
