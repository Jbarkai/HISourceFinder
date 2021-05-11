from os import listdir
from astropy.io import fits
import numpy as np
import argparse

def main(filename, scale):
    print(filename)
    cube_data = fits.getdata(filename)
    noise_file = filename.split("/")[-1].split("_")[-1].split(".fits")[0]+".derip.fits"
    noise_data = fits.getdata("./data/mosaics/" + noise_file)
    if scale == "soft":
        cube_data += noise_data*1e-1
    elif scale == "loud":
        cube_data += noise_data*4e-1
    fits.writeto("./data/training/"+scale+"Input/"+scale+"_"+filename.split("_")[-1], cube_data, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--filename', type=str, nargs='?', const='default', default='loud',
        help='Filename')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default='noisefree_1245mosB.fits',
        help='Scaling amount')
    args = parser.parse_args()

    main(args.filename, args.scale)
