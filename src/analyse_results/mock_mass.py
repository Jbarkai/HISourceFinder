import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
import astropy.units as u
from astropy.wcs import WCS
from os import listdir
import sys
sys.path.insert(0,'..')
from data_generators.cube_functions import *
import pandas as pd
from spectral_cube import SpectralCube
from random import randint


def hi_mass(scaled_flux, new_dF, chosen_f, new_dist, new_z, x_pos, y_pos, z_pos, scale_fac, noise_data, noise_header, rms):
    masked_bin = (scaled_flux > np.mean(scaled_flux) + np.std(scaled_flux)).astype(int)
    masked = masked_bin*scaled_flux*5e-3 # convert from W.U. Jy/Beam
    scaled = np.array([masked[i]*scale_fac[i]*1e1 for i in range(masked.shape[0])])
    descaled = np.sum(scaled, axis=0)*rms[x_pos:scaled.shape[1]+x_pos, y_pos:scaled.shape[2]+y_pos]
    Iv = np.sum(descaled)*u.Jy*const.c.to(u.km/u.s)*new_dF/chosen_f
    sigma_beam = (np.pi*noise_header['BMAJ']*noise_header['BMIN']/(4*np.log(2)))*(1+new_z)**2
    Sv = (noise_header['CDELT1']**2)*Iv/sigma_beam
    mass = 2.35e5*Sv*(new_dist)**2/((1+new_z)**2)
    return mass.value

orig_d = 50*u.Mpc
h_0 = 70*u.km/(u.Mpc*u.s)
noise_res = [15*u.arcsec, 25*u.arcsec]

noise_cube_hdulist = fits.open("data/mosaics/1245mosC.derip.norm.fits")
noise_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
noise_cube_hdulist[0].header['CUNIT3'] = 'Hz'
noise_cube = SpectralCube.read(noise_cube_hdulist)
noise_data = noise_cube.unmasked_data[:, :, :].value
noise_header = noise_cube.header
noise_spectral = noise_cube.spectral_axis
noise_cube_hdulist.close()
orig_data = fits.getdata("data/orig_mosaics/1245mosC.derip.fits")
scale_fac = np.nanmean(noise_data[:, 400:-400, 400:-400]/orig_data[:, 400:-400, 400:-400], axis=(1,2))
rms = np.sqrt(np.nanmean(orig_data**2, axis=0))
gal_df = pd.DataFrame()
gals =[i for i in listdir("data/mock_gals") if ".fits" in i]
k = 0
for gal in gals:
    subcube = fits.getdata("data/mock_gals/" + gal)
    orig_mass, dx, dy, dF, rest_freq, orig_scale, gal_data = load_cube("data/mock_gals/" + gal, orig_d, h_0)
    chosen_f, new_z, new_dist, z_pos = choose_freq(
    noise_spectral, noise_data.shape, gal_data.shape, rest_freq, h_0, orig_d)
    smoothed_gal = smooth_cube(noise_res, new_z, new_dist, dx, dy, gal_data, orig_scale)
    resampled, new_dF = regrid_cube(smoothed_gal, noise_header, new_dist, dx, dy, dF, orig_scale, chosen_f, rest_freq)
    scaled_flux = rescale_cube(resampled, noise_header, orig_d, rest_freq, new_dist, h_0, new_z, orig_mass, new_dF)
    x_pos = randint(0, noise_data.shape[1]-resampled.shape[1])
    y_pos = randint(0, noise_data.shape[2]-resampled.shape[2])
    mass = hi_mass(scaled_flux, new_dF, chosen_f, new_dist, new_z, x_pos, y_pos, z_pos, scale_fac, noise_data, noise_header, rms)
    gal_df = gal_df.append(pd.DataFrame([{"file":gal, "mass": mass}]))
    k += 1
    print("\r", k*100/len(gals), "%", end="")
gal_df.to_csv("gal_masses.txt")


