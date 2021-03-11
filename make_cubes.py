
"""
Create the simulated cubes by inserting 200-500 random snoothed mock galaxies randomly into a random mosaiced cube.
"""
from os import listdir
from random import sample, uniform
import argparse
import numpy as np
from astropy.io import fits


def insert_gal(i, no_gals, gal_data, cube_data, empty_cube, dim):
    """Inserts galaxy randomly into given cube

    Args:
        i (int): Galaxy index
        no_gals (int): Total number of galaxies
        gal_data (numpy.array):3D array of galaxy cube data
        cube_data (numpy.array): 3D array of noise cube to insert it to
        empty_cube (numpy.array): Empty 3D array the shape of cube_data
        dim (tuple): Dimensions of insertion location

    Returns:
        The return value. True for success, False otherwise.
    """
    # Randomly place galaxy in x and y direction and fill whole z
    x_pos = int(uniform(0, cube_data.shape[1]-dim[0]))
    y_pos = int(uniform(0, cube_data.shape[2]-dim[1]))
    z_pos = int(uniform(0, cube_data.shape[0]-gal_data.shape[0]))
    cube_data[
        z_pos:z_pos+gal_data.shape[0], x_pos:dim[0]+x_pos, y_pos:dim[1]+y_pos
        ] += gal_data*1e1
    empty_cube[
        z_pos:z_pos+gal_data.shape[0], x_pos:dim[0]+x_pos, y_pos:dim[1]+y_pos
        ] += gal_data*1e1
    print("\r" + str(int(i*100/no_gals)), end="%")
    return True


def create_fake_cube(i, no_cubes, noise_file, smoothed_gals, dim, out_dir):
    """Create fake noise cube and outputs fits files

    Args:
        i (int): Cube index
        no_cubes (int): Total number of cubes
        smoothed_gals (list): A list of 3D array of smoothed galaxy cube data
        dim (tuple): Dimensions of insertion location
        out_dir (str): Output directory of created cube

    Returns:
        The return value. True for success, False otherwise.
    """
    print("Making cube %s "%i, "out of %s..."%no_cubes)
    # Load noise cube
    cube_data = fits.getdata(noise_file)
    empty_cube = np.zeros(cube_data.shape)
    # Choose a random sample of mock galaxies and insert them
    no_gals = int(uniform(200, 500))
    gals = sample(smoothed_gals, no_gals)
    success = [insert_gal(i, no_gals, g, cube_data, empty_cube, dim) for i, g in enumerate(gals)]
    print("Success: ", success)
    # output new cube and its mask file
    hdu1 = fits.PrimaryHDU(cube_data)
    hdu2 = fits.PrimaryHDU(empty_cube)
    hdu1.writeto(out_dir + '/mockcube_%s.fits'%i)
    hdu2.writeto(out_dir + '/maskcube_%s.fits'%i)
    print("Cube %s Done!"%i)
    return True


def main(no_cubes, mos_dir, gal_dir, out_dir, dim):
    """Run creation function

    Args:
        no_cubes (int): Total number of cubes
        gal_dir (str): The directory of the mock galaxies
        out_dir (str): Output directory of created cube
        dim (tuple): Dimensions of insertion location

    Returns:
        The return value. True for success, False otherwise.
    """
    cubes = sample(listdir(mos_dir), no_cubes)
    print("Loading in galaxies...")
    smoothed_gals = [fits.getdata(f) for f in listdir(gal_dir)]
    print("Creating new cubes...")
    success = [create_fake_cube(
        i, no_cubes, f, smoothed_gals, dim, out_dir
        ) for i, f in enumerate(cubes)]
    print("Success! ", success)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert mock galaxies into HI cubes")
    parser.add_argument('mos_dir', type=str, nargs='?', default="data/mosaics",
     help='The directory of the noise cubes to insert the mock galaxies into')
    parser.add_argument('gal_dir', type=str, nargs='?', default='data/mock_gals/smoothed',
     help='The directory of the mock galaxy cubes')
    parser.add_argument('out_dir', type=str, nargs='?', default="data/training",
     help='The output directory of the synthetic cubes')
    parser.add_argument('dim', type=tuple, nargs='?', default=(512, 512),
     help='The dimensions to rescale the galaxies to')
    parser.add_argument('no_cubes', type=int, nargs='?', default=100,
     help='The number of synthetic training cubes to produce')
    args = parser.parse_args()

    main(args.no_cubes, args.mos_dir, args.gal_dir, args.out_dir, args.dim)
