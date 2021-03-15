
"""
Prepare and insert a mock galaxy into a noise cube
"""
from random import sample, randint
import numpy as np
from scipy.ndimage import zoom
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from spectral_cube import SpectralCube


def load_cube(filename):
    """Load fits Cube

    Args:
        filename (str): The filename of the cube
    Return:
        Spectral cube of the fits file
    """
    gal_cube_hdulist = fits.open(filename)
    gal_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
    gal_cube_hdulist[0].header['CUNIT3'] = 'Hz'
    gal_cube = SpectralCube.read(gal_cube_hdulist)
    gal_cube_hdulist.close()
    return gal_cube


def smooth_cube(gal_cube, fwhm=15*u.arcsec):
    """Spatially smooth cube using 2D Gaussian kernel

    Args:
        gal_cube (SpectralCube): The spectral cube to smooth
        fwhm (Quantity): Telescope FWHM
    Return:
        The smoothed cube
    """
    # Find sigma from FWHM
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    # Convert to pixels by dividing by pixel size
    sigma_pix = sigma.to('deg').value/gal_cube.header['CDELT2']
    gauss_kernel = Gaussian2DKernel(sigma_pix)
    gal_data = gal_cube.unmasked_data[:, :, :].value
    smoothed_gal = np.zeros(gal_data.shape)
    for idx in range(len(gal_data)):
        smoothed_gal[idx, :, :] = convolve(gal_data[idx, :, :], gauss_kernel)
    return smoothed_gal


def regrid_cube(gal_cube, noise_cube, gal_data, orig_d=50*u.Mpc, h_0=70*u.km/(u.Mpc*u.s)):
    """Choose random frequency for insertion and regrid cube accrodingly

    Args:
        gal_cube (SpectralCube): The cube to regrid
        noise_cube (SpectralCube): The noise cube to choose the
        gal_data (array): The cube values to regrid
        orig_d (Quantity): The original cube distance
        h_0 (Quantity): The Hubble constant at t=0
    Return:
        The regrided/resampled cube data
    """
    rest_freq = gal_cube.header['FREQ0']*u.Hz
    # Find the width of the current galaxy cube
    gal_redshift = (rest_freq/gal_cube.spectral_axis)-1
    gal_d_pos = (gal_redshift*const.c/h_0).to(u.Mpc)
    gal_width = max(gal_d_pos)- min(gal_d_pos)
    # Find max frequency
    max_pos = noise_cube.shape[0]-gal_cube.shape[0]
    max_freq = (rest_freq/(1+orig_d*h_0/const.c)).to(u.Hz)
    idx_range = np.where(np.where((noise_cube.spectral_axis < max_freq))[0] < max_pos)[0]
    np.where(noise_cube.spectral_axis < max_freq)[0]
    # Find insert channel below max freq
    freq_pos = noise_cube.spectral_axis[idx_range]
    redshift = (rest_freq/freq_pos)-1
    d_pos = (redshift*const.c/h_0).to(u.Mpc)
    # Randomly pick channel within this subset which fits in noise cube
    z_pos = randint(0, idx_range[-1])
    print(z_pos)
    new_width = max(d_pos[z_pos:z_pos+gal_data.shape[0]]) - min(d_pos[z_pos:z_pos+gal_data.shape[0]])
    width_frac = new_width/gal_width
    # Get pixel size ratio
    x_fac = noise_cube.header['CDELT1']/gal_cube.header['CDELT1']
    y_fac = noise_cube.header['CDELT2']/gal_cube.header['CDELT2']
    # Regrid cube to new distance and pixel sizes
    resampled = zoom(gal_data, (float(width_frac), x_fac, y_fac))
    return z_pos, resampled


def insert_gal(gal_data, noise_data, empty_cube, z_pos, verbose=False):
    """Inserts galaxy randomly into given cube

    Args:
        gal_data (numpy.array):3D array of galaxy cube data
        noise_data (numpy.array): 3D array of noise cube to insert it to
        empty_cube (numpy.array): Empty 3D array the shape of cube_data
        z_pos (int): Channel to insert galaxy to

    Returns:
        The return value. True for success, False otherwise.
    """
    # Randomly place galaxy in x and y direction and fill whole z
    x_pos = randint(0, noise_data.shape[1]-gal_data.shape[1])
    y_pos = randint(0, noise_data.shape[2]-gal_data.shape[2])
    if verbose:
        print(x_pos, y_pos)
    noise_data[
        z_pos:gal_data.shape[0]+z_pos,
        x_pos:gal_data.shape[1]+x_pos,
        y_pos:gal_data.shape[2]+y_pos
        ] += gal_data*3e-2
    empty_cube[
        z_pos:gal_data.shape[0]+z_pos,
        x_pos:gal_data.shape[1]+x_pos,
        y_pos:gal_data.shape[2]+y_pos
        ] += gal_data*3e-2
    return True


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
    orig_d = 50*u.Mpc
    fwhm = 15*u.arcsec
    h_0 = 70*u.km/(u.Mpc*u.s)
    # Load the cube
    gal_cube = load_cube(filename)
    # Smooth spatially
    smoothed_gal = smooth_cube(gal_cube, fwhm)
    # Regrid cube
    z_pos, resampled = regrid_cube(gal_cube, noise_cube, smoothed_gal, orig_d, h_0)
    success = insert_gal(resampled, noise_data, empty_cube, z_pos)
    print("\r" + str(int(i*100/no_gals)) + "% inserted", end="")
    return success
