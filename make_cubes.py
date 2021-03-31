
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
import gc



def create_fake_cube(noise_file, gal_dir, out_dir, min_gal=200, max_gal=500):
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
        noise_data = noise_cube.unmasked_data[:, :, :].value
        noise_header = noise_cube.header
        noise_spectral = noise_cube.spectral_axis
        noise_cube_hdulist.close()
        del noise_cube
        gc.collect()
        empty_cube = np.zeros(noise_data.shape)
        # Choose a random sample of mock galaxies and insert them
        no_gals = int(uniform(min_gal, max_gal))
        print("Inserting %s galaxies"%no_gals)
        gals = sample([f for f in listdir(gal_dir) if ".fits" in f], no_gals)
        success = [add_to_cube(
            j, no_gals, gal_dir + "/" + g, noise_header, noise_spectral, noise_data, empty_cube
            ) for j, g in enumerate(gals)]
        if all(success):
            print("Successfully inserted galaxies")
        # output new cube and its mask file
        i = noise_file.split(".")[0].split("/")[-1]
        fits.writeto(out_dir + '/maskcube_%s.fits'%i, empty_cube, noise_header, overwrite=True)
        print("Mask Cube Done!")
        fits.writeto(out_dir + '/mockcube_%s.fits'%i, noise_data, noise_header, overwrite=True)
        print("Mock Cube Done!")
        # print("Cube %s Done!"%i)
        return True
    except ValueError as e:
        print("Noise Cube %s was unable to be created"%noise_file)
        print(e)
        return False


# def main(no_cubes, mos_dir, gal_dir, out_dir, min_gal, max_gal):
#     """Run creation of simulated cubes

#     Args:
#         no_cubes (int): Total number of cubes to create
#         mos_dir (str): The directory of the mosaics
#         gal_dir (str): The directory of the mock galaxies
#         out_dir (str): Output directory of created cube
#         min_gal (int): Minimum number of galaxies to insert
#         max_gal (int): Maximum number of galaxies to insert

#     Returns:
#         The return value. True for success, False otherwise.
#     """
#     cubes = sample([mos_dir + "/" + k for k in listdir(mos_dir) if ".fits" in k], no_cubes)
#     success = [create_fake_cube(
#         k, 1, f, gal_dir, out_dir, min_gal, max_gal
#         ) for k, f in enumerate(cubes)]
#     if all(success):
#         print("Success!")




def main(cube_file, mos_dir, gal_dir, out_dir, min_gal, max_gal):
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
    success = create_fake_cube(cube_file, gal_dir, out_dir, min_gal, max_gal)
    if all(success):
        print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert mock galaxies into HI cubes")
    parser.add_argument(
        '--mos_dir', type=str, nargs='?', const='default', default="data/mosaics",
        help='The directory of the noise cubes to insert the mock galaxies into')
    parser.add_argument(
        '--gal_dir', type=str, nargs='?', const='default', default='data/mock_gals',
        help='The directory of the mock galaxy cubes')
    parser.add_argument(
        '--out_dir', type=str, nargs='?', const='default', default="data/training",
        help='The output directory of the synthetic cubes')
    parser.add_argument(
        '--cube_file', type=str, nargs='?', const='default', default="data/mosaics/1245mosC.derip.fits",
        help='The noise cube to insert into')
    parser.add_argument(
        '--min_gal', type=int, nargs='?', const='default', default=200,
        help='The minimum number of galaxies to insert')
    parser.add_argument(
        '--max_gal', type=int, nargs='?', const='default', default=500,
        help='The maximum number of galaxies to insert')
    args = parser.parse_args()

    main(args.cube_file, args.mos_dir, args.gal_dir, args.out_dir, args.min_gal, args.max_gal)
