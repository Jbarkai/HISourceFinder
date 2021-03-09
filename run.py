from make_cubes import create_fake_cube
from astropy.io import fits


def main():
    noise_file = 'data/1245mosC.derip.fits'
    no_cubes = 10
    for i in range(no_cubes):
        print("Making cube %s "%i, "out of %s..."%no_cubes)
        new_cube, mask_cube = create_fake_cube(noise_file)
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