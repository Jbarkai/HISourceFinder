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
    norm_noise = hdul[0].data
    hdul.close()
    dx = np.abs(hdr["CDELT1"]*u.deg)
    sigma_x = (dx/np.sqrt(8*np.log(2))).to(u.deg).value
    orig_noise = fits.getdata("./data/orig_mosaics/" + noise_file.replace(".norm", ""))
    scale_fac = np.nanmean(norm_noise[:, 400:-400, 400:-400]/orig_noise[:, 400:-400, 400:-400], axis=(1,2))
    # Create Guassian noise in corners
    noise_corner = np.random.normal(scale=sigma_x, size=cube_data.shape)
    # Scale corner noise
    noise_corner_scaled = np.array([noise_corner[i]*scale_fac[i] for i in range(noise_corner.shape[0])])
    # Add corners to noise cube
    norm_noise[np.isnan(norm_noise)] = noise_corner_scaled[np.isnan(noise_corner_scaled)]
    # Scale galaxies
    cube_data_scaled = np.array([cube_data[i]*scale_fac[i] for i in range(cube_data.shape[0])])
    # Add galaxies to cube
    norm_noise += cube_data_scaled*1e1
    fits.writeto("./data/training/"+scale+"Input/"+scale+"_"+filename.split("_")[-1], norm_noise, header=hdr, overwrite=True)

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
