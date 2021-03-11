
from os import listdir
from random import sample, uniform
import numpy as np
from astropy.io import fits
from astropy import units as u
from cube import Cube
from random import uniform
import gc


def create_fake_cube(noise_file, gal_dir='data/mock_gals', set_wcs=None, reproject=True, scale=1, ctype=False):
    # Load noise cube
    noise_cube = Cube(noise_file, set_wcs)
    noise_cube.load_cube(ctype=ctype)
    # Take subset for resources for now
    # subcube = noise_cube.cube_data[:, 800:1312, 800:1312]
    cube_header = noise_cube.header
    cube_data = noise_cube.cube_data
    # Delete noise cube to free up memory
    del(noise_cube)
    gc.collect()
    dim = (512, 512)
    empty_cube = np.zeros(cube_data.shape)
    # # Remove existing sources
    # noise_cube.create_mask()
    # noise_cube.cube_data = noise_cube.cube_data - noise_cube.masked
    # Choose a random sample of mock galaxies
    no_gals = int(uniform(200, 500))
    gals = sample(listdir(gal_dir), no_gals)
    smoothed_gals = [prep_gal(
        i, no_gals, gal_dir + "/"+f, dim, cube_header
        ) for i, f in enumerate(gals)]
    [insert_gal(i, no_gals, g, cube_data, empty_cube, dim) for i, g in enumerate(smoothed_gals)]
    return cube_data, empty_cube

def prep_gal(i, no_gals, filename, dim, cube_header, reproject=True):
    gal_cube = Cube(filename)
    gal_cube.load_cube(ctype=True, scale=True) # Convert Westerbork Units (W.U) to Jy/Beam
    # gal_cube.cube_data = gal_cube.cube_data[90:110, :, :]
    if reproject:
        gal_cube.rescale_cube(dim=dim)
        gal_cube.smooth_cube(cube_header)
    # gal_cube.crop_cube(scale=scale)
    gal_cube.create_mask()
    gal_data = gal_cube.cube_data
    # gal_mask = gal_cube.masked

    print("\r" + str(int(i*100/no_gals)) + "% smoothed", end="")
    return gal_data #, gal_mask


def insert_gal(i, no_gals, gal_data, cube_data, empty_cube, dim):
    # Randomly place galaxy in x and y direction and fill whole z
    si = int(uniform(0, cube_data.shape[1]-dim[0]))
    sj = int(uniform(0, cube_data.shape[2]-dim[1]))
    sk = int(uniform(0, cube_data.shape[0]-gal_data.shape[0]))
    cube_data[sk:sk+gal_data.shape[0], si:dim[0]+si, sj:dim[1]+sj] += gal_data*1e1
    empty_cube[sk:sk+gal_data.shape[0], si:dim[0]+si, sj:dim[1]+sj] += gal_data*1e1

    print("\r" + str(int(i*100/no_gals)), end="%")


def create_training_set(mos_dir="data/mosaics", gal_dir='data/mock_gals', out_dir="data/training"):
    # Decide number of galaxies per cube
    no_cubes = len(listdir(mos_dir))
    for i, mosaic in enumerate(listdir(mos_dir)):
        print("Making cube %s "%i, "out of %s..."%no_cubes)
        new_cube, mask_cube = create_fake_cube(mos_dir + "/" + mosaic, gal_dir, out_dir)
        # new_cube, mask_cube = create_fake_cube(noise_file, set_wcs='B1950')
        # new_cube.plot_slice(slice_i=si, sliced=False)
        # output new cubes
        hdu = fits.PrimaryHDU(new_cube)
        hdu.writeto(out_dir + '/mockcube_%s.fits'%i)
        # output the masks of the new cubes
        hdu = fits.PrimaryHDU(mask_cube)
        hdu.writeto(out_dir + '/maskcube_%s.fits'%i)
        # Free up memory
        del(hdu)
        del(new_cube)
        del(mask_cube)
        gc.collect()
        print("Cube %s Done!"%i)
    print("Done!")