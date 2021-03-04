
from os import listdir
from random import sample
import numpy as np
from astropy import units as u
from cube import Cube


def create_fake_cube(noise_file, gal_dir, no_gal, freq_slice=471):
    # Load noise cube
    noise_cube = Cube(noise_file)
    noise_cube.load_cube()
    # Remove existing sources
    noise_cube.create_mask()
    noise_cube.cube_data = noise_cube.cube_data - noise_cube.masked
    noise_slice = noise_cube.cube_data[freq_slice, :, :]
    # Choose a random sample of mock galaxies
    filenames = sample(listdir(gal_dir), no_gal)
    for filename in filenames:
        # Load galaxy cube
        gal_cube = Cube(filename)
        gal_cube.load_cube()
        # Smooth each galaxy
        gal_cube.smooth_cube()
        # Insert in random location
        im = gal_cube.convolved_image
        # Create random location to insert
        insert_loc = noise_slice[100:100+im.shape[0], 100:100+im.shape[1]]
        noise_slice[100:100+im.shape[0], 100:100+im.shape[1]] = im + insert_loc
    return noise_slice