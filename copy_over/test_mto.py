import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
import skimage.measure as skmeas
from datetime import datetime
from astropy.visualization import ZScaleInterval
import pandas as pd
from scipy import stats


_REJECT_TILE = False
_ACCEPT_TILE = True

rejection_rate_1 = 0
rejection_rate_2 = 0


def estimate_bg(img, verbosity=1, rejection_rate=0.05):
    """Estimate the background mean and variance of an (image) array."""
    global rejection_rate_1, rejection_rate_2

    if verbosity:
        print("\n---Estimating background---")

    rejection_rate_1 = 1 - pow(1 - rejection_rate, 0.5)
    rejection_rate_2 = 1 - pow(1 - rejection_rate, 0.25)

    # Find a usable tile size
    tile_size = largest_flat_tile(img, rejection_rate)

    if tile_size == 0:
        raise ValueError("No usable background tiles")

    if verbosity > 0:
        print("Using a tile size of", tile_size, "in the background")

    # Return the background mean and variance
    return collect_info(img, tile_size, rejection_rate, verbosity)


def largest_flat_tile(img, sig_level, tile_size_start=6, tile_size_min=4, tile_size_max=7):
    """Find an image's largest flat tile.
       Tile_size values --> 2^tile_size - i.e. parameters should be exponents.
    """

    # Convert exponents to sizes
    current_size = 2**tile_size_start
    max_size = 2**tile_size_max
    min_size = 2**tile_size_min

    # If tiles available, double the size until a maximum is found
    if available_tiles(img, current_size, sig_level):
        while current_size < max_size:
            current_size *= 2
            if not available_tiles(img, current_size, sig_level):
                # Return the last level with flat tiles available
                return int(current_size/2)
        # Return the maximum tile size if no limit has been found
        return max_size
    else:
        # If no tiles available, halve size until flat tiles found
        while current_size > min_size:
            current_size = int(current_size / 2)
            if available_tiles(img, current_size, sig_level):
                # Return first size where flat tiles are found
                return min_size

    # Return 0 if no flat tiles can be found
    return 0


def available_tiles(img, tile_length, sig_level):
    """Check if at least one background tile is available at this scale"""

    # Iterate over tiles
    for y in range(0, img.shape[0] - tile_length, tile_length):
        for x in range(0,img.shape[1]-tile_length, tile_length):
            # Test each tile for flatness
            if check_tile_is_flat(img[y:y+tile_length,x:x+tile_length], sig_level):
                return True
    return False


def collect_info(img, tile_length, rejection_rate, verbosity=1):
    """Find all flat tiles of the largest usable tile size"""

    flat_tiles = []

    for y in range(0, img.shape[0]-tile_length, tile_length):
        for x in range(0,img.shape[1]-tile_length, tile_length):
            # Test each tile for flatness
            if check_tile_is_flat(img[y:y+tile_length, x:x+tile_length], rejection_rate):
                # If flat, add to list
                flat_tiles.append([x,y])

    if verbosity:
        print("Number of usable tiles:", len(flat_tiles))

    # Estimate mean and variance over usable tiles
    return est_mean_and_variance(img, tile_length, flat_tiles)


def check_tile_is_flat(tile, rejection_rate):
    """Test if tile is flat - check normality and equal means"""

    # Discard tiles which are entirely zeros
    # Prevents breaking where the image has e.g. borders removed
    # May result in slightly lower background estimates as partial zero tiles are not removed
    if np.all(tile == 0):
        return _REJECT_TILE

    # Discard tiles which are entirely NANs
    if np.count_nonzero(~np.isnan(tile)) == 0:
        return _REJECT_TILE

    # If tile fails to be normal, reject it
    if test_normality(tile, rejection_rate_1) is False:
        return _REJECT_TILE

    # If half tile means are not equal, reject the tile
    if check_tile_means(tile, rejection_rate_2) is False:
        return _REJECT_TILE

    return _ACCEPT_TILE


def check_tile_means(tile, sig_level):
    """Check if tile halves have equal means"""

    half_height = int(tile.shape[0] / 2)
    half_width = int(tile.shape[1] / 2)

    # Top and bottom - if means unequal, reject tile
    if not test_mean_equality(tile[:half_height,:], tile[half_height:,:], sig_level):
        return _REJECT_TILE

    # Left and right - if means unequal reject tile
    if not test_mean_equality(
            tile[:,:half_width], tile[:,half_width:], sig_level):
        return _REJECT_TILE

    return _ACCEPT_TILE


def test_normality(array, test_statistic):
    """Test the hypothesis that the values in an array come from a normal distribution"""
    k2, p = stats.normaltest(array.ravel(), nan_policy='omit')

    # If p < test_statistic -> reject null hypothesis -> values are not from a normal distribution
    if p < test_statistic:
        return _REJECT_TILE
    else:
        return _ACCEPT_TILE


def test_mean_equality(array_a, array_b, test_statistic):
    """Test the hypothesis that two arrays have an equal mean"""

    # T-test assuming equal variance
    s, p = stats.ttest_ind(array_a.ravel(), array_b.ravel(), nan_policy='omit')

    # If p < test_statistic -> reject null hypothesis -> means are not equal
    if p < test_statistic:
        return _REJECT_TILE
    else:
        return _ACCEPT_TILE


def est_mean_and_variance(img, tile_length, usable):
    """Calculate a mean and variance from a list of array indices"""

    total_bg = np.vstack([img[u[1]:u[1]+tile_length,
                                  u[0]:u[0]+tile_length] for u in usable])

    return np.nanmean(total_bg, axis=None), np.nanvar(total_bg,axis=None)

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




mto_dir = "../mtobjects"
param_file = "../mtobjects/radio_smoothed-00_F.txt"
input_dir = "data/training/loudInput"
index = 0
cube_file = "data/training/loudInput/loud_1245mosC.fits"
subcube = fits.getdata(cube_file)[:110, 200:500, 1100:1500]
subcube_file = mto_dir + "/subcube_" + str(index) + cube_file.split("/")[-1]
interval = ZScaleInterval()
fits.writeto(subcube_file, subcube, overwrite=True)

bg_mean, bg_std = estimate_bg(subcube)
fin = open(param_file, "rt")
new_param = subcube_file.replace("fits", "txt")
fout = open(new_param, "wt")
for line in fin:
    fout.write(line.replace(
        'MeanBackground = 0','MeanBackground = %s'%bg_mean
        ).replace('StdDev = 0', 'StdDev = %s'%bg_std))
fin.close()
fout.close()
# Smooth and clip
smoothed = ndi.gaussian_filter(subcube, sigma=0.85)
diffused = anisotropic_diffusion(smoothed)
diffused[diffused < 0] = 0
maskcube_file = mto_dir + "/masksubcube_" + str(index) + cube_file.split("/")[-1]
output_file = "data/mto_output/outputcube_" + str(index) + cube_file.split("/")[-1]
fits.writeto(maskcube_file, diffused, overwrite=True)
# Run MTO on subcube
success = os.system('%s/mtobjects %s %s %s %s >> data/mto_output/mto_output.log'%(mto_dir, subcube_file, maskcube_file, new_param, output_file))
os.remove(new_param)
print(subcube_file)
print(maskcube_file)
print(output_file)
mto_output = fits.getdata(output_file)
mask_data = fits.getdata("data/training/Target/mask_1245mosC.fits")[:110, 200:500, 1100:1500]
mask_seg = skmeas.label(mask_data)
print(len(skmeas.regionprops(mto_output)), len(skmeas.regionprops(mask_seg)))
mto_output[mto_output > 0] = 1
gt = (mask_data).flatten().tolist()
pred = (mto_output).flatten().tolist()
intersection = np.nansum(np.logical_and(gt, pred).astype(int))
union_or = np.nansum(mask_data) + np.nansum(mto_output)
tot_dice_loss = 2*intersection/union_or
print(tot_dice_loss)
