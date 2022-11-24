from os import listdir
from astropy.io import fits
import numpy as np
import argparse
import astropy.units as u


def main(filename, orig_file, noise_file, out_dir):
    print(filename)
    cube_data = fits.getdata(filename)
    orig_data = fits.getdata(orig_file)[:, 400:-400, 400:-400]
    
    hdul = fits.open(noise_file)
    hdr = hdul[0].header
    norm_noise = hdul[0].data[:, 400:-400, 400:-400]
    rms = np.sqrt(np.nanmean(orig_data**2, axis=0))
    # Normalise galaxies like noise (divide by rms at each pixel)
    cube_data_scaled = np.array([cube_data[i]/rms for i in range(cube_data.shape[0])])
    # Add galaxies to cube
    norm_noise += cube_data_scaled
    fits.writeto(out_dir+filename.split("_")[-1], norm_noise, header=hdr, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--noise_free_file', type=str, nargs='?', const='default', default='./data/training/Input/noisefree_1245mosC.fits',
        help='The file name of the noise free cube with inserted galaxies')
    parser.add_argument(
        '--orig_file', type=str, nargs='?', const='default', default='./data/orig_mosaics/1245mosC.derip.fits',
        help='The file name of the original, un-normalised HI emission cube')
    parser.add_argument(
        '--noise_file', type=str, nargs='?', const='default', default='./data/mosaics/1245mosC.derip.norm.fits',
        help='The file name of the normalised HI emission cube')
    parser.add_argument(
        '--out_dir', type=str, nargs='?', const='default', default="data/training/",
        help='The output directory of the created cubes')
    args = parser.parse_args()

    main(args.noise_free_file, args.orig_file, args.noise_file, args.out_dir)
