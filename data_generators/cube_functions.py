
"""
Prepare and insert a mock galaxy into a noise cube
"""
from random import randint
import numpy as np
from scipy.ndimage import zoom
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import warnings
import gc
import pandas as pd
from datetime import datetime
import math


# Ignore warning about header
warnings.filterwarnings("ignore", message="Could not parse unit W.U.")



def add_to_cube(i, no_gals, filename, noise_header, noise_spectral, noise_data, inserted_gals_df, verbose=False):
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
    try:
        now = datetime.now()
        print("\r Making galaxy %s out of %s"%((i+1), no_gals), end="")
        orig_d = 50*u.Mpc
        h_0 = 70*u.km/(u.Mpc*u.s)
        noise_res = [15*u.arcsec, 25*u.arcsec]
        # Load Galaxy
        print("load")
        mass_df = pd.read_csv("original_masses.csv")
        orig_mass = mass_df.loc[mass_df["filename"] == filename.split("/")[-1].split(".")[0], "mass"].values[0]
        dx, dy, dF, rest_freq, orig_scale, gal_data = load_cube(filename, orig_d, h_0)
        # Choose channel
        print("choose freq")
        chosen_f, new_z, new_dist, z_pos = choose_freq(
            noise_spectral, noise_data.shape, gal_data.shape, rest_freq, h_0, orig_d)
        # Smooth cube
        print("smooth")
        smoothed_gal = smooth_cube(noise_res, new_z, new_dist, dx, dy, gal_data, orig_scale)
        del gal_data
        gc.collect()
        # Regrid Cube
        print("resample")
        resampled, new_dF = regrid_cube(smoothed_gal, noise_header, new_dist, dx, dy, dF, orig_scale, chosen_f, rest_freq)
        del smoothed_gal
        gc.collect()
        # Randomly place galaxy in x and y direction and fill whole z
        x_pos = randint(0, noise_data.shape[1]-resampled.shape[1])
        y_pos = randint(0, noise_data.shape[2]-resampled.shape[2])
        # Rescale flux to distance
        print("scale")
        scaled_flux = rescale_cube(resampled, orig_d, new_dist)
        del resampled
        gc.collect()
        new_mass = hi_mass(scaled_flux.value, chosen_f, new_dist.value, new_dF, new_z.value)
        # Insert galaxy
        print("insert")
        insert_gal(scaled_flux, x_pos, y_pos, z_pos, noise_data)
        inserted_gals_df = inserted_gals_df.append({
            "gal_file": filename,
            "z_pos": z_pos,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "orig_mass": orig_mass,
            "new_mass": np.log10(new_mass)
        }, ignore_index=True)
        if verbose:
            print(z_pos, x_pos, y_pos)
        print("\r Inserted galaxy ", i, "out of ", no_gals, end="")
        print(datetime.now() - now)
        return inserted_gals_df, True
    except ValueError as e:
        print("Galaxy %s was unable to be inserted"%filename)
        print(e)
        return inserted_gals_df, False

def hi_mass(scaled_flux, chosen_f, new_dist, dF, new_z):
    Iv = np.sum(scaled_flux)*u.Jy*const.c.to(u.km/u.s)*dF/chosen_f
    sigma_beam = (np.pi*5*25/(4*np.log(2)))*(1+new_z)**2
    tot_flux = (6**2)*Iv/sigma_beam
    mass = 2.35e5*tot_flux*(float(new_dist)*u.Mpc)**2/((1+new_z)**2)
    return mass.value

def load_cube(filename, orig_d, h_0):
    """Load fits Cube

    Args:
        filename (str): The filename of the cube
    """
    gal_cube_hdulist = fits.open(filename)
    gal_header = gal_cube_hdulist[0].header
    rest_freq = gal_header['FREQ0']*u.Hz
    redshift = h_0*orig_d/const.c
    # Convert from W.U. to JY/BEAM
    gal_data = gal_cube_hdulist[0].data*5e-3
    gal_cube_hdulist.close()
    # Get spatial pixel sizes
    dx = np.abs(gal_header["CDELT1"]*u.deg)
    dy = gal_header["CDELT2"]*u.deg
    dF = gal_header["CDELT3"]*u.Hz
    orig_scale = orig_d*np.tan(np.deg2rad(dx.to(u.deg)))
    return dx, dy, dF, rest_freq, orig_scale, gal_data


def choose_freq(noise_spectral, noise_shape, gal_shape, rest_freq, h_0, orig_d):
    """Choose frequency channel for insertions

    Args:
        noise_cube (SpectralCube): The noise cube to insert into
    """
    # Find max frequency
    # max_pos = noise_cube.shape[0]-self.gal_data.shape[0]
    max_freq = (rest_freq/(1+orig_d*h_0/const.c)).to(u.Hz)
    # idx_range = np.where(np.where((noise_spectral < max_freq))[0] < max_pos)[0]
    idx_range = np.where(noise_spectral < max_freq)[0]
    # Randomly pick channel within this subset which fits in noise cube
    z_pos = randint(0, idx_range[-1])
    chosen_f = noise_spectral[z_pos]
    # Calculate redshift and distance of channel
    new_z = (rest_freq/chosen_f)-1
    new_dist = (const.c*new_z/h_0).to(u.Mpc)
    return chosen_f, new_z, new_dist, z_pos

def smooth_cube(noise_res, new_z, new_dist, dx, dy, gal_data, orig_scale):
    """Spatially smooth cube using 2D Gaussian kernel
    """
    # Calculate new spatial resolution
    new_beam_x = (1+new_z)*noise_res[0]
    new_beam_y = (1+new_z)*noise_res[1]
    spat_res_x = new_dist*np.arctan(np.deg2rad(new_beam_x.to(u.deg).value))
    spat_res_y = new_dist*np.arctan(np.deg2rad(new_beam_y.to(u.deg).value))
    # Use the original resolution to find FWHM to smooth to
    fwhm_x = ((spat_res_x/orig_scale)*dx).to(u.arcsec)
    fwhm_y = ((spat_res_y/orig_scale)*dy).to(u.arcsec)
    # Find sigma from FWHM
    sigma_x = fwhm_x/np.sqrt(8*np.log(2))
    sigma_y = fwhm_y/np.sqrt(8*np.log(2))
    # Smooth to new channel
    gauss_kernel = Gaussian2DKernel((sigma_x.to(u.deg)/dx).value, (sigma_y.to(u.deg)/dy).value)
    gauss_kernel.normalize(mode="peak")
    # smoothed_gal = np.array([convolve(sliced, gauss_kernel, normalize_kernel=False) for sliced in gal_data])
    smoothed_gal = np.apply_along_axis(
        lambda x: convolve(x.reshape(gal_data.shape[1],gal_data.shape[2]),
         gauss_kernel, normalize_kernel=False), 1, gal_data.reshape(gal_data.shape[0],-1)
    )
    return smoothed_gal

def regrid_cube(smoothed_gal, noise_header, new_dist, dx, dy, dF, orig_scale, chosen_f, rest_freq):
    """Resample cube spatially and in the frequency domain

    Args:
        noise_cube (SpectralCube): The noise cube to resample to
    """
    noise_dF = noise_header["CDELT3"]*u.Hz
    new_pix_size_x = new_dist*np.tan(np.deg2rad(dx.to(u.deg)))
    new_pix_size_y = new_dist*np.tan(np.deg2rad(dy.to(u.deg)))
    pix_scale_x = dx*(new_pix_size_x/orig_scale)
    pix_scale_y = dy*(new_pix_size_y/orig_scale)
    noise_rest_vel = (const.c*(noise_dF/chosen_f)).to(u.km/u.s)
    rest_vel = (const.c*(dF/rest_freq)).to(u.km/u.s)
    dF_scale = float(rest_vel/noise_rest_vel)
    dx_scale = float(dx/pix_scale_x)
    dy_scale = float(dy/pix_scale_y)
    resampled = zoom(smoothed_gal, (dF_scale, dx_scale, dy_scale))
    return resampled, dF*dF_scale

def rescale_cube(resampled, orig_d, new_dist):
    """Rescale flux of galaxy cube to primary beam

    Args:
        noise_cube (SpectralCube): The noise cube to insert into
    """
    flux_scale = (orig_d/new_dist)**2
    scaled_flux = flux_scale*resampled
    return scaled_flux
    

def insert_gal(scaled_flux, x_pos, y_pos, z_pos, noise_data):
    """Inserts galaxy randomly into given cube

    Args:
        noise_data (numpy.array): 3D array of noise cube to insert it to
    """
    masked_bin = (scaled_flux > np.mean(scaled_flux) + np.std(scaled_flux)).astype(int)
    masked = masked_bin*scaled_flux
    z1, z2, z3, z4 = find_bounds(z_pos, scaled_flux.shape[0], noise_data.shape[0])
    x1, x2, x3, x4 = find_bounds(x_pos, scaled_flux.shape[1], noise_data.shape[1])
    y1, y2, y3, y4 = find_bounds(y_pos, scaled_flux.shape[2], noise_data.shape[2])
    noise_data[z1:z2, x1:x2, y1:y2] += masked[z3:z4, x3:x4, y3:y4].value

def find_bounds(pos, gal_shape, max_shape):
    a1 = pos-math.floor(gal_shape/2) if pos-math.floor(gal_shape/2) > 0 else 0
    a2 = pos+math.ceil(gal_shape/2) if pos+math.ceil(gal_shape/2) < max_shape else max_shape
    b1 = 0 if pos-math.floor(gal_shape/2) > 0 else math.floor(gal_shape/2) - pos
    b2 = gal_shape if pos+math.ceil(gal_shape/2) < max_shape else -(pos+math.ceil(gal_shape/2)-max_shape)
    return a1, a2, b1, b2