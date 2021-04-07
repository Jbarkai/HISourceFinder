import torch
from skimage.io import imread
from torch.utils.data import Dataset
from astropy.io import fits
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
import argparse
from os import listdir
import numpy as np
import shutil
import pickle
import os
import torch
import tensorflow as tf


class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list, # list of input paths
                 targets: list, # list of mask paths
                 dims=[10, 500, 500],
                 overlaps=[8, 400, 400],
                 load=False,
                 root='../HISourceFinder/data/training/'
                 ):
        self.list = []
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.dims = dims
        self.overlaps = overlaps
        self.save_name = root + 'hisource-list-slidingwindow.txt'
        if load:
            ## load pre-generated data
            with open(self.save_name, "rb") as fp:
                list_file = pickle.load(fp)
            self.list = list_file
            return

        subvol = '_vol_' + str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2])
        self.sub_vol_path = root + '/generated/' + subvol + '/'
        if os.path.exists(self.sub_vol_path):
            shutil.rmtree(self.sub_vol_path)
            os.mkdir(self.sub_vol_path)
        else:
            os.makedirs(self.sub_vol_path)
        ################ SLIDING WINDOW ######################
        for index in range(len(self.inputs)):
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load and slide over input
            cube_hdulist = fits.open(input_ID)
            x = cube_hdulist[0].data
            cube_hdulist.close()
            tensor_images = sliding_window(x, dims, overlaps)

            # Load and slide over target
            maskcube_hdulist = fits.open(target_ID)
            y = maskcube_hdulist[0].data
            maskcube_hdulist.close()
            tensor_segs = sliding_window(y, dims, overlaps)
            filename = self.sub_vol_path + 'cube_' + str(index) +"_subcube_"
            list_saved_paths = [(filename + str(j) + '.npy', filename + str(j) + 'seg.npy') for j in range(len(tensor_images))]
            ############### SAVE SUBCUBES ##########################
            for j in range(len(tensor_images)):
                np.save(list_saved_paths[j][0], tensor_images[j])
                np.save(list_saved_paths[j][1], tensor_segs[j])
            self.list += list_saved_paths
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
        # Typecasting
        input_x = torch.from_numpy(x.astype(np.float32)).type(self.inputs_dtype)
        target_y = torch.from_numpy(y.astype(np.int64)).type(self.targets_dtype)
        return input_x, target_y

def sliding_window(arr, dims, overlaps):
    kernel=(1, dims[0], dims[1], dims[2], 1)
    stride=(1, overlaps[0], overlaps[1], overlaps[2], 1) 
    _,sx,sy,sz,_ = kernel   
    in_patches=tf.extract_volume_patches(
        arr[None, ..., None],kernel,stride,'SAME',
    )
    _,x,y,z,n = in_patches.shape
    in_patches = tf.reshape(in_patches,[x*y*z,sx,sy,sz])
    return in_patches


def main(batch_size, shuffle, num_workers, dims, overlaps, root, random_seed, train_size):
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

    Returns:
        The training and validation data loaders
    """
    # input and target files
    inputs = [root+'Input/' + x for x in listdir(root+'Input') if ".fits" in x]
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
                                        root=root)

    # dataset validation
    dataset_valid = SegmentationDataSet(inputs=inputs_valid,
                                        targets=targets_valid,
                                        dims=dims,
                                        overlaps=overlaps,
                                        load=False,
                                        root=root)

    # dataloader training
    params = {'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers}
    dataloader_training = DataLoader(dataset=dataset_train, **params)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid, **params)
    return dataloader_training, dataloader_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training and validation datasets")
    parser.add_argument(
        '--batch_size', type=int, nargs='?', const='default', default=4,
        help='Batch size')
    parser.add_argument(
        '--shuffle', type=bool, nargs='?', const='default', default=True,
        help='Whether or not to shuffle the train/val split')
    parser.add_argument(
        '--num_workers', type=int, nargs='?', const='default', default=2,
        help='The number of workers to use')
    parser.add_argument(
        '--dims', type=list, nargs='?', const='default', default=[10, 500, 500],
        help='The dimensions of the subcubes')
    parser.add_argument(
        '--overlaps', type=list, nargs='?', const='default', default=[8, 400, 400],
        help='The dimensions of the overlap of subcubes')
    parser.add_argument(
        '--root', type=str, nargs='?', const='default', default='../HISourceFinder/data/training/',
        help='The root directory of the data')
    parser.add_argument(
        '--random_seed', type=int, nargs='?', const='default', default=42,
        help='Random Seed')
    parser.add_argument(
        '--train_size', type=float, nargs='?', const='default', default=0.8,
        help='Ratio of training to validation split')
    args = parser.parse_args()

    main(
        args.batch_size, args.shuffle, args.num_workers, args.dims,
        args.overlaps, args.root, args.random_seed, args.train_size)
