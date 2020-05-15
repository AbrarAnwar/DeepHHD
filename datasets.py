import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from glob import glob
from util import *
from PIL import Image
from pynhhd import nHHD

class MPISintel(data.Dataset):
    def __init__(self, flow_root, image_root):
        file_list = sorted(glob(flow_root + '*/*.flo'))
        self.flow_list = []
        self.image_list = []

        for file in file_list:
            fbase = file[len(flow_root):]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = image_root + fprefix + "%04d"%(fnum+0) + '.png'
            img2 = image_root + fprefix + "%04d"%(fnum+1) + '.png'
            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        # doing decomp here is smart cause multi core handling potentially
        vfield = np.array(Image.open(self.image_list[0][0]))
        self.dims = (vfield.shape[0], vfield.shape[1])        # Y, X
        dx = (0.001, 0.001)         # dy, dx                                                                                            
        self.nhhd = nHHD(grid=self.dims, spacings=dx)


    def __getitem__(self, index):

        index = index % self.size

        im1 = torch.from_numpy(np.array(Image.open(self.image_list[index][0])).reshape(3, self.dims[0], self.dims[1])).float()
        im2 = torch.from_numpy(np.array(Image.open(self.image_list[index][1])).reshape(3, self.dims[0], self.dims[1])).float()

        flo = read(self.flow_list[index])

        self.nhhd.decompose(flo, num_cores=4)

        r = torch.from_numpy(self.nhhd.r.reshape(2,self.dims[0],self.dims[1]))
        h = torch.from_numpy(self.nhhd.h.reshape(2,self.dims[0],self.dims[1]))
        d = torch.from_numpy(self.nhhd.d.reshape(2,self.dims[0],self.dims[1]))

        flo = torch.from_numpy(flo)

        return [im1, im2], [flo, r, d, h]

    def __len__(self):
        return self.size



"""
dataset = MPISintel('/home/abrar/flow/MPI-Sintel-complete/training/flow/', '/home/abrar/flow/MPI-Sintel-complete/training/clean/')

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
for i_batch, sample_batched in enumerate(dataloader):
    print(len(sample_batched[0]))
    print((sample_batched[0][0].shape))
    im1s = sample_batched[0][0]
    im2s = sample_batched[0][1]

    flos = sample_batched[1][0]
    rs = sample_batched[1][1]
    ds = sample_batched[1][2]
    hs = sample_batched[1][3]

    print(flos.shape)
    print(ds.shape)
"""

