from os import listdir
from astropy.io import fits
import numpy as np
import argparse
import astropy.units as u


def main(filename, scale):
    print(filename)
    cube_data = fits.getdata(filename)
    noise_file = filename.split("/")[-1].split("_")[-1].split(".fits")[0]+".derip.norm.fits"
    print(noise_file)
    hdul = fits.open("./data/mosaics/" + noise_file)
    hdr = hdul[0].header
    noise_data = hdul[0].data
    hdul.close()
    dx = np.abs(hdr["CDELT1"]*u.deg)
    sigma_x = (dx/np.sqrt(8*np.log(2))).to(u.deg).value
    # if scale == "soft":
    #     cube_data += noise_data*1e-1
    # elif scale == "loud":
    cube_data += noise_data*4e-1
    noise_corner = np.random.normal(scale=sigma_x, size=cube_data.shape)
    cube_data[np.isnan(cube_data)] = noise_corner[np.isnan(cube_data)]
    fits.writeto("./data/training/"+scale+"Input/"+scale+"_"+filename.split("_")[-1], cube_data, header=hdr, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--filename', type=str, nargs='?', const='default', default='noisefree_1245mosB.fits',
        help='Filename')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default='loud',
        help='Scaling amount')
    args = parser.parse_args()

    main(args.filename, args.scale)
