from os import listdir
from astropy.io import fits
import numpy as np
import argparse
import astropy.units as u


def main(filename):
    print(filename)
    cube_data = fits.getdata(filename)
    mos_name = filename.split("/")[-1].split("_")[-1].split(".fits")[0]
    noise_file = mos_name + ".derip.norm.fits"
    print(noise_file)
    orig_data = fits.getdata("./data/orig_mosaics/%s.derip.fits"%mos_name)[:, 400:-400, 400:-400]
    
    hdul = fits.open("./data/mosaics/" + noise_file)
    hdr = hdul[0].header
    norm_noise = hdul[0].data[:, 400:-400, 400:-400]
    rms = np.sqrt(np.nanmean(orig_data**2, axis=0))
    # Normalise galaxies like noise (divide by rms at each pixel)
    cube_data_scaled = np.array([cube_data[i]/rms for i in range(cube_data.shape[0])])
    # Add galaxies to cube
    norm_noise += cube_data_scaled
    fits.writeto("./data/training/InputBoth/"+filename.split("_")[-1], norm_noise, header=hdr, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--filename', type=str, nargs='?', const='default', default='./data/training/Input/noisefree_1245mosC.fits',
        help='Filename')
    args = parser.parse_args()

    main(args.filename)
