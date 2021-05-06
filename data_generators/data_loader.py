import torch
from skimage.io import imread
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from torch.utils.data import DataLoader
from random import sample
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import as_strided
import pathlib
import argparse
from os import listdir
import numpy as np
import shutil
import pickle
import os
import torch
import tensorflow as tf
import gc


class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list, # list of input paths
                 targets: list, # list of mask paths
                 dims=[128, 128, 64],
                 overlaps=[15, 20, 20],
                 load=False,
                 root='./data/training/',
                 mode="train",
                 scale="loud",
                 arr_shape=(1800, 2400, 652)
                 ):
        self.list = []
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.dims = dims
        self.overlaps = overlaps
        self.mode = mode
        self.root = root
        self.scale = scale
        self.arr_shape = arr_shape
        self.save_name = self.root + 'hisource-list-' + self.mode + '-slidingwindowindices.txt'
        if load:
            ## load pre-generated data
            # self.list = [(inputs[i], targets[i]) for i in range(len(inputs))]
            with open(self.save_name, "rb") as fp:
                list_file = pickle.load(fp)
                self.list = list_file
            return

        for f_in, f_tar in zip(self.inputs, self.targets):
            self.list += save_sliding_window(self.arr_shape, self.dims, np.array(self.dims)-np.array(self.overlaps), f_in, f_tar)
        # Save list of subcubes
        with open(self.save_name, "wb") as fp:
            pickle.dump(self.list, fp)

    def __len__(self):
        return len(self.list)

    def __getitem__(self,
                    index: int):
        # Select the sample and prepare
        interval = ZScaleInterval()
        cube_files, x, y, z = self.list[index]
        subcube = np.moveaxis(fits.getdata(cube_files[0]), 0, 2)[x1:x2, y1:y2, z1:z2]
        # Get rid of nans in corners and Z scale normalise between 0 and 1 
        x = interval(np.nan_to_num(subcube, np.mean(subcube)))
        y = np.moveaxis(fits.getdata(cube_files[1]), 0, 2)[x1:x2, y1:y2, z1:z2]
        return torch.FloatTensor(x.astype(np.float32)).unsqueeze(0), torch.FloatTensor(y.astype(np.float32)).unsqueeze(0)


def save_sliding_window(arr_shape, window_shape, step, f_in, f_tar):
    x, y, z =(((np.array(arr_shape) - np.array(window_shape))
                      // np.array(step)) + 1)

    sliding_window_indices = []
    count = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                x1, x2 = dims[0]*i-overlaps[0]*i, dims[0]*(i+1)-overlaps[0]*i
                y1, y2 = dims[1]*j-overlaps[1]*j, dims[1]*(j+1)-overlaps[1]*j
                z1, z2 = dims[2]*k-overlaps[2]*k, dims[2]*(k+1)-overlaps[2]*k
                sliding_window_indices.append(([f_in, f_tar], [x1, x2], [y1, y2], [z1, z2]))
                count += 1
                print("\r", count*100/(x*y*z), end="")
    return sliding_window_indices
