import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from astropy.wcs import WCS
from scipy import ndimage as ndi
from astropy.visualization import ImageNormalize, ZScaleInterval
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import tensorflow as tf
# import skimage.measure as skmeas
from os import listdir
# import paramiko
# from skimage.util.shape import view_as_windows
from spectral_cube import SpectralCube
import os
import gc
import pickle
import sys
sys.path.insert(0,'..')
from data_generators.cube_functions import *
from random import sample, uniform
import pandas as pd

orig_d = 50*u.Mpc
h_0 = 70*u.km/(u.Mpc*u.s)
noise_res = [15*u.arcsec, 25*u.arcsec]
gals = sample([i for i in listdir("data/mock_gals")], 300)
noise_cube_hdulist = fits.open("data/mosaics/1245mosC.derip.norm.fits")
orig_data = fits.getdata("data/orig_mosaics/1245mosC.derip.fits")
noise_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
noise_cube_hdulist[0].header['CUNIT3'] = 'Hz'
noise_cube = SpectralCube.read(noise_cube_hdulist)
noise_data = noise_cube.unmasked_data[:, :, :].value
noise_header = noise_cube.header
noise_spectral = noise_cube.spectral_axis
noise_cube_hdulist.close()

def hi_mass(scaled_flux, new_dF, chosen_f, new_dist, new_z, z_pos):
    masked_bin = (scaled_flux > np.mean(scaled_flux) + np.std(scaled_flux)).astype(int)
    masked = masked_bin*scaled_flux
    Sv = (u.km/u.s)*u.Jy*np.sum([masked[i]*np.sum(masked_bin[i])*u.Jy*const.c.to(u.km/u.s)*new_dF/noise_spectral[z_pos+i] for i in range(scaled_flux.shape[0])], axis=0)
    M = np.sum(2.36e5*Sv*(new_dist)**2)/((1+new_z)**2)
    hi_mass = M/((np.pi*15*25/(4*np.log(2)))*((1+new_z)**2))
    return hi_mass.value

gal_df = pd.DataFrame()
for gal in ["g1_model1000.fits"]:
    if True:
        print(gal)
        filename = "data/mock_gals/" + gal
        gal_cube_hdulist = fits.open(filename)
        gal_header = gal_cube_hdulist[0].header
        rest_freq = gal_header['FREQ0']*u.Hz
        redshift = h_0*orig_d/const.c
        # Convert from W.U. to JY/BEAM
        gal_data = gal_cube_hdulist[0].data #*5e-3
        gal_cube_hdulist.close()
        # Get spatial pixel sizes
        dF = gal_header["CDELT3"]*u.Hz
        max_freq = (rest_freq/(1+orig_d*h_0/const.c)).to(u.Hz)
        idx_range = np.where(noise_spectral < max_freq)[0]
        # print(hi_mass(gal_data, dF, max_freq, orig_d, redshift, idx_range[0]))
        orig_mass, dx, dy, dF, rest_freq, orig_scale, gal_data = load_cube(filename, orig_d, h_0)
        print(orig_mass)
        chosen_f, new_z, new_dist, z_pos = choose_freq(
            noise_spectral, noise_data.shape, gal_data.shape, rest_freq, h_0, orig_d)
        smoothed_gal = smooth_cube(noise_res, new_z, new_dist, dx, dy, gal_data, orig_scale)
        resampled, new_dF = regrid_cube(smoothed_gal, noise_header, new_dist, dx, dy, dF, orig_scale, chosen_f, rest_freq)
        scaled_flux = rescale_cube(resampled, noise_header, orig_d, rest_freq, new_dist, h_0, new_z, orig_mass, new_dF)
        scale_fac = np.nanmean(noise_data[z_pos:z_pos+scaled_flux.shape[0], 400:-400, 400:-400]/orig_data[z_pos:z_pos+scaled_flux.shape[0], 400:-400, 400:-400], axis=(1,2))
        print(scale_fac[0])
        # rescaled = np.array([scaled_flux[i]*scale_fac[i] for i in range(scaled_flux.shape[0])])*1e1
        # mass = pd.DataFrame([hi_mass(rescaled, new_dF, chosen_f, new_dist, new_z, z_pos)], columns=['mass'])
        # mass['filename'] = gal
        deltaV = (new_dF*const.c/rest_freq).to(u.km/u.s)
        S_v = np.sum(scaled_flux, axis=0)*u.Jy*deltaV
        new_mass = np.sum(2.36e5*S_v*new_dist**2)/(1+new_z)
        print(new_mass)
        print(hi_mass(scaled_flux, new_dF, chosen_f, new_dist, new_z, z_pos))
        # gal_df = gal_df.append(mass)
    else:
        print("fail")

# gal_df.to_csv("mass_test.csv")
