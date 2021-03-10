from make_cubes import create_fake_cube
from astropy.io import fits
from os import listdir


def main():
    # Decide number of galaxies per cube
    no_cubes = len(listdir("data/mosaics"))
    for i, mosaic in enumerate(listdir("data/mosaics")):
        print("Making cube %s "%i, "out of %s..."%no_cubes)
        new_cube, mask_cube = create_fake_cube("data/mosaics/" + mosaic)
        # new_cube, mask_cube = create_fake_cube(noise_file, set_wcs='B1950')
        # new_cube.plot_slice(slice_i=si, sliced=False)
        # output new cubes
        hdu = fits.PrimaryHDU(new_cube.cube_data)
        hdu.writeto('data/training/mockcube_%s.fits'%i)
        # output the masks of the new cubes
        hdu = fits.PrimaryHDU(mask_cube)
        hdu.writeto('data/training/maskcube_%s.fits'%i)
        print("Cube %s Done!"%i)
    print("Done!")
    return 0

if __name__ == "__main__":
    main()