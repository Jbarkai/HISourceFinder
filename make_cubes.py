
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
from gal_cube import GalCube
import gc


def add_to_cube(i, no_gals, filename, noise_cube, noise_data, empty_cube):
    """Load, smooth, regrid and insert mock galaxies

    Args:
        i (int): Cube index
        no_cubes (int): Total number of cubes
        filename (str): The file name of the mock galaxy
        noise_cube (SpectralCube): Noise cube to insert galaxy into
        noise_data (numpy.array): 3D array of noise cube to insert it to
        empty_cube (numpy.array): Empty 3D array the shape of cube_data

    Returns:
        The return value. True for success, False otherwise.
    """
    print("\r" + str(int(i*100/no_gals)) + "%", end="")
    gal_cube = GalCube(filename)
    # Load Galaxy
    gal_cube.load_cube()
    # Choose channel
    gal_cube.choose_freq(noise_cube)
    # Smooth cube
    gal_cube.smooth_cube()
    # Regrid Cube
    gal_cube.regrid_cube(noise_cube)
    # Rescale flux
    gal_cube.rescale_cube(noise_cube)
    # Insert galaxy
    gal_cube.insert_gal(noise_data, empty_cube)
    print("\r" + str(int(i*100/no_gals)) + "% inserted", end="")
    return True



def create_fake_cube(i, no_cubes, noise_file, gal_dir, out_dir, min_gal=200, max_gal=500):
    """Create fake noise cube and outputs fits files

    Args:
        i (int): Cube index
        no_cubes (int): Total number of cubes
        noise_file (str): The file of the noise cube
        gal_dir (str): The directory of the galaxy cubes
        out_dir (str): Output directory of created cube
        min_gal (int): Minimum number of galaxies to insert
        max_gal (int): Maximum number of galaxies to insert

    Returns:
        The return value. True for success, False otherwise.
    """
    print("Making cube %s "%i, "out of %s..."%no_cubes)
    # Load noise cube
    noise_cube_hdulist = fits.open(noise_file)
    noise_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
    noise_cube_hdulist[0].header['CUNIT3'] = 'Hz'
    noise_cube = SpectralCube.read(noise_cube_hdulist)
    noise_data = noise_cube.unmasked_data[:, :, :].value
    noise_cube_hdulist.close()
    empty_cube = np.zeros(noise_cube.shape)
    # Choose a random sample of mock galaxies and insert them
    no_gals = int(uniform(min_gal, max_gal))
    gals = sample([f for f in listdir(gal_dir) if ".fits" in f], no_gals)
    success = [add_to_cube(
        j, no_gals, gal_dir + "/" + g, noise_cube, noise_data, empty_cube
        ) for j, g in enumerate(gals)]
    if all(success):
        print("Successfully inserted galaxies")
    # output new cube and its mask file
    hdu1 = fits.PrimaryHDU(noise_data, noise_cube.header)
    hdu1.writeto(out_dir + '/mockcube_%s.fits'%i, overwrite=True)
    print("Mock Cube Done!")
    del hdu1
    del noise_data
    del noise_cube
    gc.collect()
    hdu2 = fits.PrimaryHDU(empty_cube, noise_cube.header)
    hdu2.writeto(out_dir + '/maskcube_%s.fits'%i, overwrite=True)
    print("Cube %s Done!"%i)
    return True


def main(no_cubes, mos_dir, gal_dir, out_dir, min_gal, max_gal):
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
    cubes = sample([mos_dir + "/" + k for k in listdir(mos_dir) if ".fits" in k], no_cubes)
    success = [create_fake_cube(
        k, no_cubes, f, gal_dir, out_dir, min_gal, max_gal
        ) for k, f in enumerate(cubes)]
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
        '--no_cubes', type=int, nargs='?', const='default', default=2,
        help='The number of synthetic training cubes to produce')
    parser.add_argument(
        '--min_gal', type=int, nargs='?', const='default', default=200,
        help='The minimum number of galaxies to insert')
    parser.add_argument(
        '--max_gal', type=int, nargs='?', const='default', default=500,
        help='The maximum number of galaxies to insert')
    args = parser.parse_args()

    main(args.no_cubes, args.mos_dir, args.gal_dir, args.out_dir, args.min_gal, args.max_gal)