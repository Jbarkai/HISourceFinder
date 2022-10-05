

from astropy import units as u
import astropy.constants as const
from astropy.io import fits
import numpy as np
import pandas as pd
from os import listdir

def flux(row, hi_data, cat_data, rms):
    subcube = hi_data[int(row['bbox-0']):int(row['bbox-3']), int(row['bbox-1']):int(row['bbox-4']), int(row['bbox-2']):int(row['bbox-5'])]
    det_subcube = cat_data[int(row['bbox-0']):int(row['bbox-3']), int(row['bbox-1']):int(row['bbox-4']), int(row['bbox-2']):int(row['bbox-5'])]
    # print(np.unique(det_subcube))
    det_subcube[det_subcube != row.label] = 0
    det_subcube[det_subcube == row.label] = 1
    descaled = np.sum(subcube*det_subcube, axis=0)*rms[int(row['bbox-1']):int(row['bbox-4']), int(row['bbox-2']):int(row['bbox-5'])]
    h_0 = 70*u.km/(u.Mpc*u.s)
    rest_freq = 1.4204058e9*u.Hz
    z = float(row.dist)*u.Mpc*h_0/const.c.to(u.km/u.s)
    freq = rest_freq/(1+z)
    dF = 36621.09375*u.Hz
    Iv = np.sum(descaled)*u.Jy*const.c.to(u.km/u.s)*dF/freq
    sigma_beam = (np.pi*5*25/(4*np.log(2)))*(1+z)**2
    tot_flux = (6**2)*Iv/sigma_beam
    peak_flux = np.max(descaled*u.Jy*const.c.to(u.km/u.s)*dF*(6**2)/(freq*sigma_beam))
    return tot_flux, peak_flux

def hi_mass(row, tot_flux):
    h_0 = 70*u.km/(u.Mpc*u.s)
    z = float(row.dist)*u.Mpc*h_0/const.c.to(u.km/u.s)
    mass = 2.35e5*tot_flux*(float(row.dist)*u.Mpc)**2/((1+z)**2)
    return mass.value


catalogs = ["VNET_catalog.txt"]

for catalog in catalogs:
    print(catalog)
    cat_df = pd.read_csv("results/" + catalog)
    cat_df["hi_mass"] = np.nan
    cat_df["peak_flux"] = np.nan
    cat_df["tot_flux"] = np.nan
    for mos_name in cat_df.mos_name.unique():
        print(mos_name)
        hi_data = fits.getdata("./data/training/InputBoth/loud_%s.fits"%mos_name)  
        orig_data = fits.getdata("./data/orig_mosaics/%s.derip.fits"%mos_name)  
        rms = np.sqrt(np.nanmean(orig_data**2, axis=0))     
        cat_data = fits.getdata(cat_df[cat_df.file.str.contains(mos_name)].file.unique()[0])
        # cat_data = fits.getdata("./data/training/TargetBoth/mask_%s.fits"%mos_name)
        for i, row in cat_df[cat_df.mos_name==mos_name].iterrows():
            try:
                tot_flux, peak_flux = flux(row, hi_data, cat_data, rms)
                mass = hi_mass(row, tot_flux)
            except:
                mass = np.nan
                tot_flux = np.nan
                peak_flux = np.nan
            cat_df.loc[i, "hi_mass"] = mass
            cat_df.loc[i, "tot_flux"] = tot_flux.value
            cat_df.loc[i, "peak_flux"] = peak_flux.value
            print("\r", i*100/len(cat_df[cat_df.mos_name==mos_name]), "%", end="")
    cat_df.to_csv("./results/%s"%catalog)
