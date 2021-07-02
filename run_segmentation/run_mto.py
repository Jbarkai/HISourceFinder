import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
import skimage.measure as skmeas
from datetime import datetime


def save_sliding_window(arr_shape, dims, overlaps, f_in):
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
                sliding_window_indices.append(([f_in], [x1, x2], [y1, y2], [z1, z2]))
                count += 1
                print("\r", count*100/(x*y*z), end="")
    print()
    return sliding_window_indices


def anisotropic_diffusion(image, num_iters=10, K=2,  
    stepsize_lambda=0.2):
    if stepsize_lambda > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    image = image.copy()
    # simplistic boundary conditions -- no diffusion at the boundary
    central = image[1:-1, 1:-1]
    n = image[:-2, 1:-1]
    s = image[2:, 1:-1]
    e = image[1:-1, :-2]
    w = image[1:-1, 2:]
    directions = [s,e,w]
    for i in range(num_iters):
        di = n - central
        accumulator = di/(1 + (np.absolute(di)/K)**2)
    for direction in directions:
        di = direction - central
        accumulator += di/(1 + (np.absolute(di)/K)**2)
    accumulator *= stepsize_lambda
    central += accumulator
    return image


def mto_eval(window, mto_dir, param_file, empty_arr, index):
    cube_files, x, y, z = window
    subcube = fits.getdata(cube_files[0])[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    # Normalise localised noise
    # noisecube = fits.getdata(sofia_noise_file)[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
    # subcube = subcube/noisecube
    # Save as fits file
    subcube_file = mto_dir + "/subcube_" + str(index) + cube_files[0].split("/")[-1]
    fits.writeto(subcube_file, subcube, overwrite=True)
    # Smooth and clip
    smoothed = ndi.gaussian_filter(subcube, sigma=0.85)
    diffused = anisotropic_diffusion(smoothed)
    diffused[diffused < 0] = 0
    maskcube_file = mto_dir + "/masksubcube_" + str(index) + cube_files[0].split("/")[-1]
    output_file = "data/mto_output/outputcube_" + str(index) + cube_files[0].split("/")[-1]
    fits.writeto(maskcube_file, diffused, overwrite=True)
    # Run MTO on subcube
    success = os.system('%s/mtobjects %s %s %s %s >> data/mto_output/mto_output.log'%(mto_dir, subcube_file, maskcube_file, param_file, output_file))
    # Delete outputted fits file
    os.remove(subcube_file)
    os.remove(maskcube_file)
    # os.remove(output_file.split(".")[0] + "_attributes.fits.gz")
    if success != 0:
        return
    # Load MTO output
    mto_output = fits.getdata(output_file)
    os.remove(output_file)
    # os.remove(output_file.replace(".fits", "_statistics.fits.gz"))
    # os.system("zip %s %s"%(output_file.replace(".fits", ".zip"), output_file))
    # Convert it to binary
    mto_output[mto_output > 0] = 1
    empty_arr[x[0]:x[1], y[0]:y[1], z[0]:z[1]] = np.nanmax(np.array([empty_arr[x[0]:x[1], y[0]:y[1], z[0]:z[1]], mto_output]), axis=0)
    return

def make_cube(f_in, mto_dir, param_file):
    arr_shape = (652, 1800, 2400)
    dims = [652, 200, 300]
    overlaps = [20, 15, 20]
    # sofia_noise_file = "data/sofia_output/sofia_" + f_in.split("/")[-1].replace(".fits", "_mask.fits")
    cube_list = save_sliding_window(arr_shape, dims, overlaps, f_in)
    empty_arr = np.zeros(arr_shape)
    for index, window in enumerate(cube_list):
        print("\r", index*100/len(cube_list), "%", end="")
        mto_eval(window, mto_dir, param_file, empty_arr, index)
        # mto_eval(window, mto_dir, param_file, empty_arr, index, sofia_noise_file)
    # os.remove(sofia_noise_file)
    out_cube_file = "data/mto_output/mtocubeout_" + f_in.split("/")[-1]
    # nonbinary_im = skmeas.label(empty_arr)
    fits.writeto(out_cube_file, empty_arr, overwrite=True)
    return


def main(mto_dir, param_file, input_dir):
    time_taken = {}
    # Load test data
    cubes = [input_dir + "/" + x for x in listdir(input_dir) if "1245mos" in x]
    for f_in in cubes:
        before = datetime.now()
        print(f_in)
        make_cube(f_in, mto_dir, param_file)
        after = datetime.now()
        difference = (after - before).total_seconds()
        time_taken.update({f_in: difference})
        # os.system("zip %s %s"%(out_cube_file.replace(".fits", ".zip"), out_cube_file))
    out_file = "mto_performance_" + input_dir.split("/")[-1] + ".txt"
    with open(out_file, "wb") as fp:
        pickle.dump(time_taken, fp)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTO",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mto_dir', type=str, nargs='?', const='default', default="../mtobjects",
        help='The directory of the MTO executable')
    parser.add_argument(
        '--param_file', type=str, nargs='?', const='default', default="../mtobjects/radio_smoothed-00_F.txt",
        help='The parameter file')
    parser.add_argument(
        '--input_dir', type=str, nargs='?', const='default', default="data/training/loudInput",
        help='The directory of the input data')
    args = parser.parse_args()

    main(args.mto_dir, args.param_file, args.input_dir)

