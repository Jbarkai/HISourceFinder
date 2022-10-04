from os import listdir
import numpy as np
from astropy.io import fits

noise_files = listdir("data/mosaics")
for noise_file in noise_files:
    print(noise_file)
    noise_cube_hdulist = fits.open("data/mosaics/" + noise_file)
    noise_cube = noise_cube_hdulist[0].data[:, 400:-400, 400:-400]
    noise_header = noise_cube_hdulist[0].header
    noise_cube_hdulist.close()
    noise_data = np.zeros(noise_cube.shape) # To create noise free cube
    i = noise_file.split(".")[0]
    fits.writeto(f'data/empty_cubes/mask_{i}.fits', noise_data, noise_header, overwrite=True)

