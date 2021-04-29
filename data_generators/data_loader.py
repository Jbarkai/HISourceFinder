import torch
from skimage.io import imread
from torch.utils.data import Dataset
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from torch.utils.data import DataLoader
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
                 scale="loud"
                 ):
        self.list = []
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.dims = dims
        self.overlaps = overlaps
        self.mode = mode
        self.save_name = root + 'hisource-list-' + self.mode + '-slidingwindow.txt'
        if load:
            ## load pre-generated data
            with open(self.save_name, "rb") as fp:
                list_file = pickle.load(fp)
            self.list = list_file
            return

        subvol = '_vol_' + str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2]) + "_" +self.mode
        self.sub_vol_path = root + '/generated/' + scale +"/" + subvol + '/'
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
            list_saved_masks = save_sliding_window(target_ID, dims, overlaps, filename, seg=False)
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
        # # Typecasting
        # input_x = torch.from_numpy(x.astype(np.float32)).type(self.inputs_dtype)
        # target_y = torch.from_numpy(y.astype(np.int64)).type(self.targets_dtype)
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
                no_nans = np.nan_to_num(subcube)
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


def main(batch_size, shuffle, num_workers, dims, overlaps, root, random_seed, train_size, scale):
    """Create training and validation datasets

    Args:
        batch_size (int): Batch size
        shuffle (bool): Whether or not to shuffle the train/val split
        num_workers (int): The number of workers to use
        dims (list): The dimensions of the subcubes
        overlaps (list): The dimensions of the overlap of subcubes
        root (str): The root directory of the data
        random_seed (int): Random Seed
        train_size (float): Ratio of training to validation split
        scale (str): Loud or soft - S-N ratio

    Returns:
        The training and validation data loaders
    """
    # input and target files
    inputs = [root+scale+'Input/' + x for x in listdir(root+scale+'Input') if ".fits" in x]
    targets = [root+'Target/' + x for x in listdir(root+'Target') if ".fits" in x]
    inputs_train, inputs_valid = train_test_split(
        inputs,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)

    targets_train, targets_valid = train_test_split(
        targets,
        random_state=random_seed,
        train_size=train_size,
        shuffle=True)
    # dataset training
    dataset_train = SegmentationDataSet(inputs=inputs_train,
                                        targets=targets_train,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=False,
                                        root=root,
                                        mode="train",
                                        scale=scale)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=False,
                                        root=root,
                                        mode="test",
                                        scale=scale)

    # dataloader training
    params = {'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers}
    dataloader_training = DataLoader(dataset=dataset_train, **params)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid, **params)
    return dataloader_training, dataloader_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training and validation datasets",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch_size', type=int, nargs='?', const='default', default=1,
        help='Batch size')
    parser.add_argument(
        '--shuffle', type=bool, nargs='?', const='default', default=True,
        help='Whether or not to shuffle the train/val split')
    parser.add_argument(
        '--num_workers', type=int, nargs='?', const='default', default=2,
        help='The number of workers to use')
    parser.add_argument(
        '--dims', type=list, nargs='?', const='default', default=[128, 128, 64],
        help='The dimensions of the subcubes')
    parser.add_argument(
        '--overlaps', type=list, nargs='?', const='default', default=[15, 20, 20],
        help='The dimensions of the overlap of subcubes')
    parser.add_argument(
        '--root', type=str, nargs='?', const='default', default='../HISourceFinder/data/training/',
        help='The root directory of the data')
    parser.add_argument(
        '--random_seed', type=int, nargs='?', const='default', default=42,
        help='Random Seed')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default="loud",
        help='The scale of inserted galaxies to noise')
    parser.add_argument(
        '--train_size', type=float, nargs='?', const='default', default=0.8,
        help='Ratio of training to validation split')
    args = parser.parse_args()

    main(
        args.batch_size, args.shuffle, args.num_workers, args.dims,
        args.overlaps, args.root, args.random_seed, args.train_size, args.scale)
