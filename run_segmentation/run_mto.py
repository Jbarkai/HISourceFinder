import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi


def mto_eval(index, test_list, mto_dir, param_file):
    cube_files, x, y, z = test_list[index]
    subcube = fits.getdata("." + cube_files[0])[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
    smoothed_gal = ndi.gaussian_filter(subcube, sigma=3)
    # Save as fits file
    subcube_file = mto_dir + "/subcube_" + str(index) + ".fits"
    maskcube_file = mto_dir + "/masksubcube_" + str(index) + ".fits"
    output_file = mto_dir + "/outputcube_" + str(index) + ".fits"
    fits.writeto(subcube_file, subcube, overwrite=True)
    fits.writeto(subcube_file, smoothed_gal, overwrite=True)
    # Run MTO on subcube
    os.system('%s/mtobjects %s %s %s mto_%s'%(mto_dir, subcube_file, maskcube_file, param_file, output_file))
    # Load MTO output
    mto_ouput = fits.getdata("mto_%s"%output_file)
    # Convert it to binary
    mto_ouput[mto_ouput > 0] = 1
    # Delete outputted fits file
    os.remove(subcube_file)
    os.remove(maskcube_file)
    # Load ground truth
    seg_dat = fits.getdata("." + cube_files[1])[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
    gt = (seg_dat).flatten().tolist()
    pred = (mto_ouput).flatten().tolist()
    intersection = np.nansum(np.logical_and(gt, pred).astype(int))
    union_or = np.nansum(gt) + np.nansum(pred)
    return intersection, union_or


def main(mto_dir, test_file, param_file):
    # Load test data
    with open(test_file, "rb") as fp:
        test_list = pickle.load(fp)
    intersections = 0
    all_or = 0
    for index in range(len(test_list)):
        print("\r", index*100/len(test_list), end="")
        results = mto_eval(index, test_list, mto_dir, param_file)
        intersections += results[0]
        all_or += results[1]
    tot_dice_loss = 2*intersections/all_or
    output_file = test_file.split("/")[-2] + "/" + "mto_dice.txt"
    with open(output_file, "wb") as fp:
        pickle.dump(tot_dice_loss, fp)
    print('Total Dice Loss: ', 100*tot_dice_loss, "%")
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
    parser.add_argument(
        '--param_file', type=str, nargs='?', const='default', default="../HISourceFinder/saved_models/test_list.txt",
        help='The file listing the test sliding window pieces')
    args = parser.parse_args()

    main(args.mto_dir, args.test_file, args.param_file)
