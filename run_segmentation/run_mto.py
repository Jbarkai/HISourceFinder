import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi


def mto_eval(window, mto_dir, param_file, empty_arr, index):
    cube_files, x, y, z = window
    subcube = fits.getdata(cube_files[0])[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
    # Smooth and clip
    smoothed_gal = ndi.gaussian_filter(subcube, sigma=3)
    smoothed_gal[smoothed_gal < 0] = 0
    # Save as fits file
    subcube_file = mto_dir + "/subcube_" + str(index) + cube_files[0].split("/")[-1]
    maskcube_file = mto_dir + "/masksubcube_" + str(index) + cube_files[0].split("/")[-1]
    output_file = "data/mto_output/outputcube_" + str(index) + cube_files[0].split("/")[-1]
    fits.writeto(subcube_file, subcube, overwrite=True)
    fits.writeto(maskcube_file, smoothed_gal, overwrite=True)
    # Run MTO on subcube
    success = os.system('%s/mtobjects %s %s %s %s > data/mto_output/mto_output.log'%(mto_dir, subcube_file, maskcube_file, param_file, output_file))
    # Delete outputted fits file
    os.remove(subcube_file)
    os.remove(maskcube_file)
    if success != 0:
        return
    # Load MTO output
    mto_output = fits.getdata(output_file)
    os.remove(output_file)
    # Convert it to binary
    mto_output[mto_output > 0] = 1
    empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]] = np.nanmean(np.array(empty_arr[z[0]:z[1], x[0]:x[1], y[0]:y[1]], mto_output), axis=0)
    return


def main(mto_dir, test_file, param_file):
    # Load test data
    with open(test_file, "rb") as fp:
        test_list = pickle.load(fp)
    cubes = np.unique([i[0][0] for i in test_list])
    masks = np.unique([i[0][1] for i in test_list])
    tot_dice_loss = []
    for cube, mask in zip(cubes, masks):
        print(cube)
        cube_list = [i for i in test_list if cube in i[0][0]]
        empty_arr = np.zeros((652, 1800, 2400))
        for index, window in enumerate(cube_list):
            print("\r", index*100/len(cube_list), "%", end="")
            mto_eval(window, mto_dir, param_file, empty_arr, index)
        out_cube_file = "data/mto_output/mtocubeout_" + cube.split("/")[-1]
        fits.writeto(out_cube_file, empty_arr)
        # Load ground truth to evaluate
        gt = (fits.getdata(mask)).flatten().tolist()
        pred = (empty_arr).flatten().tolist()
        intersections = np.nansum(np.logical_and(gt, pred).astype(int))
        all_or = np.nansum(gt) + np.nansum(pred)
        tot_dice_loss += [2*intersections/all_or]
    output_file = test_file.split("/")[-2] + "/" + "mto_dice.txt"
    with open(output_file, "wb") as fp:
        pickle.dump(tot_dice_loss, fp)
    print('Total Dice Loss: ', 100*np.mean(tot_dice_loss), "%")
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
