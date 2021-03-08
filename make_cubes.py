
from os import listdir
from random import sample
import numpy as np
from astropy import units as u
from cube import Cube
from random import uniform


def create_fake_cube(noise_file, set_wcs=None):
    # Load noise cube
    print("Loading %s..."%noise_file)
    noise_cube = Cube(noise_file, set_wcs)
    noise_cube.load_cube()
    # # Remove existing sources
    # noise_cube.create_mask()
    # noise_cube.cube_data = noise_cube.cube_data - noise_cube.masked
    # Choose a random sample of mock galaxies
    print("Adding Galaxies...")
    files = [filename for filename in listdir('data') if "model" in filename]
    gal_locs =[]
    for filename in files:
        print("Adding Galaxy %s "%filename, files.index(filename)+1, "out of %s"%len(files))
        gal_cube = Cube("data/"+filename)
        gal_cube.load_cube(ctype=True)
        # Subcube (need to make this dependent on galaxy size)
        gal_cube.cube_data = gal_cube.cube_data[90:110, 150:250, 150:250]
        gal_cube.smooth_cube()
        gal_cube.create_mask(scale=1e2)
        # Randomly place galaxy in x and y direction and fill whole z
        mk, mi, mj = gal_cube.masked.shape
        si = int(uniform(0, noise_cube.cube_data.shape[1]-mi))
        sj = int(uniform(0, noise_cube.cube_data.shape[2]-mj))
        sk = int(uniform(0, noise_cube.cube_data.shape[0]-mk))
        in_loc = noise_cube.cube_data[sk:sk+mk, si:si+mi, sj:sj+mj]
        noise_cube.cube_data[sk:sk+mk, si:si+mi, sj:sj+mj] = gal_cube.masked + in_loc
        gal_locs.append([[sk, si, sj], [mk, mi, mj]])
    return gal_locs, noise_cube