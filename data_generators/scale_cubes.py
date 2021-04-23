from os import listdir
from astropy.io import fits
import numpy as np
import argparse

def main(filename, scale):
    cube_data = fits.getdata("../data/training/Input/" + filename)
    noise_file = filename.split("_")[-1].split(".fits")[0]+".derip.fits"
    noise_data = fits.getdata("../data/mosaics/" + noise_file)
    if scale == "soft":
        cube_data += noise_data*1e-1
    elif scale == "loud":
        cube_data += noise_data*4e-1
    fits.writeto("../data/training/"+scale+"Input/"+scale+"_"+filename.split("_")[-1], cube_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale cubes")
    parser.add_argument(
        '--filename', type=str, nargs='?', const='default', default='noisefree_1245mosB.fits',
        help='Filename')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default='noisefree_1245mosB.fits',
        help='Scaling amount')
    args = parser.parse_args()

    main(
        args.filename)
