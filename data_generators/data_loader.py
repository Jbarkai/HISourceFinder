import torch
from skimage.io import imread
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from torch.utils.data import DataLoader
from random import sample
from sklearn.model_selection import train_test_split
from skimage.util.shape import view_as_windows
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
                 root='../HISourceFinder/data/training/',
                 mode="train",
                 scale="loud",
                 train_size=0.6
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
        self.train_size = train_size
        self.save_name = self.root + 'hisource-list-' + self.mode + '-slidingwindow.txt'
        if load:
            ## load pre-generated data
            self.list = [(inputs[i], targets[i]) for i in range(len(inputs))]
            return

        subvol = '_vol_' + str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2]) + "_" + str(int(self.train_size*100))
        self.sub_vol_path = self.root + '/generated/' + self.scale +"/" + subvol + '/'
        if os.path.exists(self.sub_vol_path):
            shutil.rmtree(self.sub_vol_path)
            os.mkdir(self.sub_vol_path)
        else:
            os.makedirs(self.sub_vol_path)
        ################ SLIDING WINDOW ######################
        for index in range(len(self.inputs)):
            input_ID = self.inputs[index]
            target_ID = self.targets[index]
            filename = self.sub_vol_path + 'cube_' + str(index) +"_subcube_"
            list_saved_cubes = save_sliding_window(input_ID, dims, overlaps, filename, seg=False)
            print("saved %s cubes"%len(list_saved_cubes))
            list_saved_masks = save_sliding_window(target_ID, dims, overlaps, filename, seg=True)
            print("saved %s masks"%len(list_saved_masks))
            self.list += [(x, y) for x, y in zip(list_saved_cubes, list_saved_masks)]
        # Save list of subcubes
        with open(self.save_name, "wb") as fp:
            pickle.dump(self.list, fp)

    def __len__(self):
        return len(self.list)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_path, seg_path = self.list[index]
        x, y = np.load(input_path), np.load(seg_path)
        return torch.FloatTensor(x).unsqueeze(0), torch.FloatTensor(y).unsqueeze(0)


def save_sliding_window(input_ID, dims, overlaps, filename, seg=False):
    cube_data = np.moveaxis(fits.getdata(input_ID), 0, 2)
    arr_out = view_as_windows(cube_data, dims, np.array(dims)-np.array(overlaps))
    x,y,z = arr_out.shape[:3]
    count = 0
    filelist = []
    interval = ZScaleInterval()
    for i in range(x):
        for j in range(y):
            for k in range(z):
                subcube = arr_out[i, j, k, :, :, :]
                # Get rid of nans in corners
                no_nans = np.nan_to_num(subcube, np.mean(subcube))
                # Z scale normalise between 0 and 1  
                scaled = interval(no_nans)
                if seg:
                    filesave = filename + str(count) + 'seg.npy'
                else:
                    filesave = filename + str(count) + '.npy'
                np.save(filesave, scaled)
                filelist.append(filesave)
                count += 1
                print("\r", count*100/(x*y*z), end="")
    return filelist
