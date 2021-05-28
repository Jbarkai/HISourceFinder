import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
import sys
sys.path.insert(0,'..')
import skimage.measure as skmeas
from medzoo_imports import create_model
import torch
from datetime import datetime


def vnet_eval(cube_list, model):
    arr_shape = (652, 1800, 2400)
    empty_arr = np.zeros(arr_shape)*np.nan
    for index, window in enumerate(cube_list):
        cube_files, x, y, z = window
        subcube = fits.getdata(cube_files[0])[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
        input_tensor = torch.FloatTensor(np.moveaxis(subcube, 0, 2).astype(np.float32)).unsqueeze(0)[None, ...]
        with torch.no_grad():
            out_cube = model.inference(input_tensor)
        out_np = np.moveaxis(out_cube.squeeze()[1].numpy(), 2, 0)
        binary_im = out_np > 0
        empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]] = np.nanmax(np.array([empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]], binary_im.astype(int)]), axis=0)
        print("\r", index*100/len(cube_list), "%", end="")
    return empty_arr


def main(args, test_file):
    time_taken = {}
    with open(test_file, "rb") as fp:
        test_list = pickle.load(fp)
    cubes = np.unique([i[0][0] for i in test_list])
    model, optimizer = create_model(args)
    model.restore_checkpoint(args.pretrained)
    model.eval()
    for cube in cubes:
        before = datetime.now()
        print(cube)
        cube_list = [i for i in test_list if cube in i[0][0]]
        cube_list = [i for i in cube_list if (i[3] in [[176, 240], [220, 284]]) & 
               (i[2] in [[0, 128],[108, 236], [216, 344]]) &
              (i[1] in [[0, 128],[113, 241], [226, 354]])]
        empty_arr = vnet_eval(cube_list, model)
        nonbinary_im = skmeas.label(empty_arr)
        out_cube_file = "data/vnet_output/vnet_cubeout_" + cube.split("/")[-1]
        fits.writeto(out_cube_file, nonbinary_im, overwrite=True)
        after = datetime.now()
        difference = (after - before).total_seconds()
        time_taken.update({cube: difference})
    out_file = "vnet_performance_" + test_file.split("/")[0]
    with open(out_file, "wb") as fp:
        pickle.dump(time_taken, fp)
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
        '--pretrained', type=str, nargs='?', const='default', default="./VNET__last_epoch.pth",
        help='The location of the pretrained model')
    parser.add_argument(
        '--test_file', type=str, nargs='?', const='default', default="./notebooks/loud_1245mosC-slidingwindowindices.txt",
        help='The file listing the test sliding window pieces')
    args = parser.parse_args()
    main(args, args.test_file)

