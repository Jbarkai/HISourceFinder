import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import pickle
import scipy.ndimage as ndimage
import random


class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label
        t = random.choice(self.transforms)
        if t == "nothing":
            return img_tensors, label
        for i in range(len(img_tensors)):

            if i == (len(img_tensors) - 1):
                ### do only once the augmentation to the label
                img_tensors[i], label = t(img_tensors[i], label)
            else:
                img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label


class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensors, label):
        augment = np.random.random(1) < self.p
        if not augment:
            return img_tensors, label

        for i in range(len(img_tensors)):

            for t in self.transforms:
                if i == (len(img_tensors) - 1):
                    ### do only once augmentation to the label
                    img_tensors[i], label = t(img_tensors[i], label)
                else:
                    img_tensors[i], _ = t(img_tensors[i], label)
        return img_tensors, label


def random_rotate3D(img_numpy, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    return ndimage.rotate(img_numpy, angle, axes=axes)


class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated
        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        img_numpy = random_rotate3D(img_numpy, self.min_angle, self.max_angle)
        if label.any() != None:
            label = random_rotate3D(label, self.min_angle, self.max_angle)
        return img_numpy, label

def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, max_percentage=0.2):
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


class RandomShift(object):
    def __init__(self, max_percentage=0.2):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy = random_shift(img_numpy, self.max_percentage)
        if label.any() != None:
            label = random_shift(label, self.max_percentage)
        return img_numpy, label

def random_flip(img_numpy, label=None, axis_for_flip=0):
    axes = [0, 1, 2]

    img_numpy = flip_axis(img_numpy, axes[axis_for_flip])
    img_numpy = np.squeeze(img_numpy)

    if label is None:
        return img_numpy, label
    else:
        y = flip_axis(label, axes[axis_for_flip])
        y = np.squeeze(y)
    return img_numpy, y


def flip_axis(img_numpy, axis):
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy


class RandomFlip(object):
    def __init__(self):
        self.axis_for_flip = np.random.randint(0, 3)

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped
        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """
        return random_flip(img_numpy, label, self.axis_for_flip)



class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list, # list of input paths
                 targets: list, # list of mask paths
                 dims=[128, 128, 64],
                 overlaps=[15, 20, 20],
                 root='./data/training/',
                 arr_shape=(1800, 2400, 652),
                 mode="train_val",
                 save_name="../saved_models/",
                 load=False,
                 augmentation=False
                 ):
        self.list = []
        self.inputs = inputs
        self.targets = targets
        self.inputs_dtype = np.float32
        self.targets_dtype = np.long
        self.dims = dims
        self.overlaps = overlaps
        self.root = root
        self.mode = mode
        self.arr_shape = arr_shape
        self.augmentation = augmentation
        if self.augmentation:
            self.transform = RandomChoice(
                transforms=[RandomRotation(), RandomFlip(),
                            RandomShift(), "nothing"], p=0.5)
        if load:
            self.save_name = save_name
            ## load pre-generated data
            with open(self.save_name, "rb") as fp:
                list_file = pickle.load(fp)
                self.list = list_file
            return
        self.save_name = save_name + self.mode + '-hisource-list-slidingwindowindices.txt'
        for f_in, f_tar in zip(self.inputs, self.targets):
            self.list += save_sliding_window(self.arr_shape, self.dims, self.overlaps, f_in, f_tar)
        # Save list of subcubes
        with open(self.save_name, "wb") as fp:
            pickle.dump(self.list, fp)

    def __len__(self):
        return len(self.list)

    def __getitem__(self,
                    index: int):
        # Select the sample and prepare
        cube_files, x, y, z = self.list[index]
        subcube = np.moveaxis(fits.getdata(cube_files[0]), 0, 2)[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        # Get rid of nans in corners and Z scale normalise between 0 and 1 
        # dat = interval(np.nan_to_num(subcube, np.mean(subcube)))
        seg_dat = np.moveaxis(fits.getdata(cube_files[1]), 0, 2)[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        if self.mode == 'train_val' and self.augmentation and (seg_dat > 0).any():
            print('augmentation')
            augmented_cube, augmented_seg = self.transform(subcube, seg_dat)
            return torch.FloatTensor(augmented_cube.astype(self.inputs_dtype)).unsqueeze(0), torch.FloatTensor(augmented_seg.astype(self.targets_dtype)).unsqueeze(0)
        else:
            return torch.FloatTensor(subcube.astype(self.inputs_dtype)).unsqueeze(0), torch.FloatTensor(seg_dat.astype(self.targets_dtype)).unsqueeze(0)


def save_sliding_window(arr_shape, dims, overlaps, f_in, f_tar):
    x, y, z =(((np.array(arr_shape) - np.array(dims))
                      // np.array(np.array(dims)-np.array(overlaps))) + 1)

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
    print()
    return sliding_window_indices
