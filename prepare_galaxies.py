
"""
Prepare the mock galaxies for insertion into noise cubes
"""
from os import listdir
import argparse
from astropy.io import fits
from cube import Cube


def prep_gal(i, no_gals, filename, dim, cdelts, reproject=True, out_dir='mock_gals/smoothed'):
    """Load, rescalem smooth and output mock galaxies

    Args:
        i (int): Galaxy index
        no_gals (int): Total number of galaxies
        filename (str): The file name of the mock galaxy
        dim (tuple): Dimensions to rescale galaxies to
        cdelts (tuple): The 3 directional pixel scales to smooth to
        reproject (bool): Whether or not to reproject galax
        out_dir (str): Output directory of smoothed galaxies

    Returns:
        The return value. True for success, False otherwise.
    """
    gal_cube = Cube(filename)
    gal_cube.load_cube(ctype=True, scale=True) # Convert Westerbork Units (W.U) to Jy/Beam
    if reproject:
        gal_cube.rescale_cube(dim=dim)
        gal_cube.smooth_cube(cdelts)
    # gal_cube.crop_cube(scale=scale)
    # gal_cube.create_mask()
    # output new cubes
    hdu = fits.PrimaryHDU(gal_cube.cube_data)
    hdu.writeto(out_dir + '/smoothed_%s'%filename)
    print("\r" + str(int(i*100/no_gals)) + "% smoothed", end="")
    # return gal_data #, gal_mask
    return True

def main(gal_dir, out_dir, dim):
    """Run galaxy smoothing

    Args:
        gal_dir (str): The directory of the mock galaxies
        out_dir (str): Output directory of smoothed galaxies
        dim (tuple): Dimensions to rescale galaxies to

    Returns:
        The return value. True for success, False otherwise.
    """
    cdelt1 = -1.666666707000E-03
    cdelt2 = 1.666666707000E-03
    # CDELT3 = 1.220703125000E+04
    cdelt3 = 3.662109375000E+04
    # subcube = noise_cube.cube_data[:, 800:1312, 800:1312]
    # Prepare the galaxy cubes
    print("Smoothing all the galaxy cubes...")
    success = [prep_gal(
        i, len(listdir(gal_dir)), gal_dir + "/"+f, dim,(cdelt1, cdelt2, cdelt3), out_dir
        ) for i, f in enumerate(listdir(gal_dir))]
    print("Success! ", success)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale and smooth mock galaxies")
    parser.add_argument('gal_dir', type=str, const=1, default='data/mock_gals',
     help='The directory of the mock galaxy cubes')
    parser.add_argument('out_dir', type=str, const=1, default="data/mock_gals/smoothed",
     help='The output directory of the smoothed synthetic cubes')
    parser.add_argument('dim', type=tuple, const=1, default=(512, 512),
     help='The dimensions to rescale the galaxies to')
    args = parser.parse_args()

    main(args.gal_dir, args.out_dir, args.dim)
