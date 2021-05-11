import argparse
from os import listdir
import os
import pickle
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import numpy as np


def mto_eval(index, test_list, mto_dir):
    cube_files, z, x, y = test_list[index]
    interval = ZScaleInterval()
    subcube = fits.getdata(cube_files[0])[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    # Get rid of nans in corners and Z scale normalise between 0 and 1 
    dat = interval(np.nan_to_num(subcube, np.mean(subcube)))
    # Save as fits file
    subcube_file = (mto_dir + "/subcube_" + str(index))[0]
    fits.writeto(subcube_file, dat, overwrite=True)
    # Run MTO on subcube
    os.system('%s 1 16 "%s" 1 "" "" 32 1 test 1'%(mto_dir, subcube_file))
    # Load MTO output
    mto_ouput = fits.getdata(mto_dir+"segmcube.fits")
    # Convert it to binary
    mto_ouput[mto_ouput > 0] = 1
    # Delete outputted fits file
    os.remove(subcube_file)
    # Load ground truth
    seg_dat = fits.getdata(cube_files[1])[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    intersection = np.nansum(np.logical_and(seg_dat, mto_ouput).astype(int))
    if np.nansum(seg_dat) == np.nansum(mto_ouput) == 0:
        dice = 1
    else:
        union = np.nansum(seg_dat) + np.nansum(mto_ouput)
        dice = (2*intersection)/(union)
    return dice


def main(mto_dir, ):
    # Load test data
    with open(test_file, "rb") as fp:
        test_list = pickle.load(fp)
    dice_losses = []
    for index in range(len(test_list)):
        dice = mto_eval(index, test_list, mto_dir)
        dice_losses += [100.0*dice]
    with open("mto_dice.txt", "wb") as fp:
        pickle.dump(dice_losses, fp)
    print('Average: ', np.mean(dice_losses), "%")
    print('Standard Deviation: ', np.std(dice_losses), "%")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTO",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mto_dir', type=str, nargs='?', const='default', default="./mto-lvq",
        help='The directory where MTO lives')
    parser.add_argument(
        '--test_file', type=str, nargs='?', const='default', default="../HISourceFinder/saved_models/test_list.txt",
        help='The file listing the test sliding window pieces')
    args = parser.parse_args()

    main(args.mto_dir, args.test_file)
