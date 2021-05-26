import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
import sys
sys.path.insert(0,'..')
from medzoo_imports import create_model
import torch


def vnet_eval(cube_list, model):
    arr_shape = (652, 1800, 2400)
    empty_arr = np.zeros(arr_shape)*np.nan
    for index, window in enumerate(cube_list):
        cube_files, x, y, z = window
        subcube = np.moveaxis(fits.getdata(cube_files[0]), 0, 2)[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        input_tensor = torch.FloatTensor(subcube.astype(np.float32)).unsqueeze(0)[None, ...]
        with torch.no_grad():
            out_cube = model.inference(input_tensor)
            out_np = np.moveaxis(out_cube.squeeze()[0].numpy(), 2, 0)
            empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]] = np.nanmean(np.array([empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]], out_np]), axis=0)
            print("\r", index*100/len(cube_list), "%", end="")
    return empty_arr


def main(args, test_file):
    with open(test_file, "rb") as fp:
        test_list = pickle.load(fp)
    cubes = np.unique([i[0][0] for i in test_list])
    model, optimizer = create_model(args)
    model.restore_checkpoint(args.pretrained)
    model.eval()
    for cube in cubes:
        print(cube)
        cube_list = [i for i in test_list if cube in i[0][0]]
        empty_arr = vnet_eval(cube_list, model)
        binary_im = empty_arr > 0
        out_cube_file = "data/vnet_output/vnet_cubeout_" + cube.split("/")[-1]
        fits.writeto(out_cube_file, binary_im.astype(int))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INFERENCE VNET",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', type=str, nargs='?', const='default', default='VNET',
        help='The 3D segmentation model to use')
    parser.add_argument(
        '--opt', type=str, nargs='?', const='default', default='adam',
        help='The type of optimizer')
    parser.add_argument(
        '--lr', type=float, nargs='?', const='default', default=1e-3,
        help='The learning rate')
    parser.add_argument(
        '--inChannels', type=int, nargs='?', const='default', default=1,
        help='The desired modalities/channels that you want to use')
    parser.add_argument(
        '--classes', type=int, nargs='?', const='default', default=2,
        help='The number of classes')
    parser.add_argument(
        '--pretrained', type=str, nargs='?', const='default', default="saved_models/fold_0_checkpoints/VNET_/VNET__BEST.pth",
        help='The location of the pretrained model')
    parser.add_argument(
        '--test_file', type=str, nargs='?', const='default', default="../HISourceFinder/saved_models/test_list.txt",
        help='The file listing the test sliding window pieces')
    args = parser.parse_args()
    main(args, args.test_file)

