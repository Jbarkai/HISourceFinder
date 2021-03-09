
from os import listdir
from random import sample
import numpy as np
from astropy import units as u
from cube import Cube
from random import uniform


def create_fake_cube(noise_file, set_wcs=None, reproject=False, scale=1, ctype=False):
    # Load noise cube
    noise_cube = Cube(noise_file, set_wcs)
    noise_cube.load_cube(ctype=ctype)
    # Take subset for resources for now
    noise_cube.cube_data = noise_cube.cube_data[:, 800:1312, 800:1312]
    dim = noise_cube.cube_data.shape[1:]
    empty_cube = np.zeros(noise_cube.cube_data.shape)
    # # Remove existing sources
    # noise_cube.create_mask()
    # noise_cube.cube_data = noise_cube.cube_data - noise_cube.masked
    # Choose a random sample of mock galaxies
    # print("Adding Galaxy %s "%filename, files.index(filename)+1, "out of %s"%len(files))
    [insert_gal(f, noise_cube.cube_data, empty_cube, dim, reproject, scale) for f in listdir('data/mock_gals')]
    return noise_cube, empty_cube

def insert_gal(filename, cube_data, empty_cube, dim, reproject=True, scale=1):
    gal_cube = Cube("data/mock_gals/"+filename)
    gal_cube.load_cube(ctype=True)
    # gal_cube.cube_data = gal_cube.cube_data[90:110, :, :]
    if reproject:
        gal_cube.rescale_cube(dim=dim)
        gal_cube.smooth_cube()
    gal_cube.crop_cube(scale=scale)
    # Randomly place galaxy in x and y direction and fill whole z
    mk, mi, mj = gal_cube.cube_data.shape
    si = int(uniform(0, cube_data.shape[1]-mi))
    sj = int(uniform(0, cube_data.shape[2]-mj))
    sk = int(uniform(0, cube_data.shape[0]-mk))
    in_loc = cube_data[sk:sk+mk, si:si+mi, sj:sj+mj]
    cube_data[sk:sk+mk, si:si+mi, sj:sj+mj] = gal_cube.cube_data + in_loc
    empt_loc = empty_cube[sk:sk+mk, si:si+mi, sj:sj+mj]
    empty_cube[sk:sk+mk, si:si+mi, sj:sj+mj] = gal_cube.cube_data + empt_loc