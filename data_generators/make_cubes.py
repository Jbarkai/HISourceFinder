
"""
Create the simulated cubes by inserting 200-500 random snoothed
mock galaxies randomly into a random mosaiced cube.
"""
from os import listdir
from random import sample, uniform
import argparse
import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube
from cube_functions import add_to_cube
import pandas as pd
import gc


def remove_real_sources(noise_data):
    return


def create_fake_cube(noise_file, gal_dir, out_dir):
    """Create fake noise cube and outputs fits files

    Args:
        noise_file (str): The file of the noise cube
        gal_dir (str): The directory of the galaxy cubes
        out_dir (str): Output directory of created cube
        min_gal (int): Minimum number of galaxies to insert
        max_gal (int): Maximum number of galaxies to insert

    Returns:
        The return value. True for success, False otherwise.
    """
    try:
        # print("Making cube %s "%i, "out of %s..."%no_cubes)
        # Load noise cube
        print(noise_file)
        noise_cube_hdulist = fits.open(noise_file)
        noise_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
        noise_cube_hdulist[0].header['CUNIT3'] = 'Hz'
        noise_cube = SpectralCube.read(noise_cube_hdulist)
        # slice corners
        noise_cube = noise_cube[:, 400:-400, 400:-400]
        # noise_data = noise_cube.unmasked_data[:, :, :].value
        noise_header = noise_cube.header
        noise_spectral = noise_cube.spectral_axis
        noise_cube_hdulist.close()
        noise_data = np.zeros(noise_cube.shape) # To create noise free cube
        del noise_cube
        gc.collect()
        # Choose a random sample of mock galaxies and insert them
        no_gals = 300
        print("Inserting %s galaxies"%no_gals)
        gals = sample([f for f in listdir(gal_dir) if ".fits" in f], no_gals)
        inserted_gals_df = pd.DataFrame(columns=["gal_file", "z_pos", "x_pos", "y_pos", "orig_mass", "new_mass"])
        for j, gal in enumerate(gals):
            success = False
            while not success:
                inserted_gals_df, success = add_to_cube(
                    j, no_gals, gal_dir + "/" + gal, noise_header, noise_spectral, noise_data, inserted_gals_df
                )
        mos_name = noise_file.split("/")[-1].split(".")[0]
        inserted_gals_df["mos_name"] = mos_name
        inserted_gals_df.to_csv(mos_name + "_inserted.csv", index=False)
        print("Successfully inserted galaxies")
        # output new cube and its mask file
        i = noise_file.split(".")[0].split("/")[-1]
        empty_cube = (noise_data > 0).astype(int)
        fits.writeto(out_dir + '/Target/mask_%s.fits'%i, empty_cube, noise_header, overwrite=True)
        print("Mask Cube Done!")
        del empty_cube
        gc.collect()
        fits.writeto(out_dir + '/Input/noisefree_%s.fits'%i, noise_data, noise_header, overwrite=True)
        print("Mock Cube Done!")
        print("Cube %s Done!"%i)
        return True
    except ValueError as e:
        print("Noise Cube %s was unable to be created"%noise_file)
        print(e)
        return False


def main(cube_file, gal_dir, out_dir):
    """Run creation of simulated cubes

    Args:
        no_cubes (int): Total number of cubes to create
        mos_dir (str): The directory of the mosaics
        gal_dir (str): The directory of the mock galaxies
        out_dir (str): Output directory of created cube
        min_gal (int): Minimum number of galaxies to insert
        max_gal (int): Maximum number of galaxies to insert

    Returns:
        The return value. True for success, False otherwise.
    """
    # cubes = sample([mos_dir + "/" + k for k in listdir(mos_dir) if ".fits" in k], no_cubes)
    success = create_fake_cube(cube_file, gal_dir, out_dir)
    if success:
        print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert mock galaxies into HI cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--gal_dir', type=str, nargs='?', const='default', default='data/mock_gals',
        help='The directory of the mock galaxy cubes')
    parser.add_argument(
        '--out_dir', type=str, nargs='?', const='default', default="data/training",
        help='The output directory of the synthetic cubes')
    parser.add_argument(
        '--cube_file', type=str, nargs='?', const='default', default="data/mosaics/1245mosC.derip.norm.fits",
        help='The noise cube to insert into')
    args = parser.parse_args()

    main(args.cube_file, args.gal_dir, args.out_dir)
