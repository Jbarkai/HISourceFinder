from spectral_cube import SpectralCube
from astropy import units as u
import astropy.constants as const
from astropy.io import fits
import numpy as np
import pandas as pd
from photutils.centroids import centroid_com


mto_cat_df = pd.read_csv("./results/loud_MTOexp3_catalog.txt", index_col=0)
sofia_cat_df = pd.read_csv("./results/loud_SOFIAexp3_catalog.txt", index_col=0)
sofia_cat_df["mos_name"] = sofia_cat_df.file.str.split("_", expand=True)[3]
mto_cat_df["mos_name"] = mto_cat_df.file.str.split("_", expand=True)[3].str.replace(".fits", "")
vnet_cat_df = pd.read_csv("./results/loud_VNETexp3_catalog.txt", index_col=0)
vnet_cat_df["mos_name"] = vnet_cat_df.file.str.split("_", expand=True)[4].str.replace(".fits", "")
mask_cat_df = pd.read_csv("./results/loud_MASKexp3_catalog.txt", index_col=0)
mask_cat_df["mos_name"] = mask_cat_df.file.str.split("_", expand=True)[1].str.replace(".fits", "")
def hi_mass(scaled_flux, new_dF, chosen_f, new_dist, new_z, noise_spectral, z_pos):
    masked_bin = (scaled_flux > np.mean(scaled_flux) + np.std(scaled_flux)).astype(int)
    masked = masked_bin*scaled_flux
    Sv = (u.km/u.s)*u.Jy*np.sum([masked[i]*np.sum(masked_bin[i])*const.c.to(u.km/u.s)*new_dF/noise_spectral[int(z_pos)+i] for i in range(scaled_flux.shape[0])], axis=0)
    M = np.sum(2.36e5*Sv*(new_dist)**2)/((1+new_z)**2)
    hi_mass = M/((np.pi*15*25/(4*np.log(2)))*((1+new_z)**2))
    return hi_mass.value

for cat_df, method in zip([sofia_cat_df, mto_cat_df, vnet_cat_df, mask_cat_df], ["SOFIA", "MTO", "VNET", "MASK"]):
    cat_df["hi_mass"] = np.nan
    # cat_df["asymmetry"] = 0
    for mos_name in cat_df.mos_name.unique():
        noise_cube_hdulist = fits.open("./data/training/InputBoth/loud_%s.fits"%mos_name)
        noise_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
        noise_cube_hdulist[0].header['CUNIT3'] = 'Hz'
        noise_cube = SpectralCube.read(noise_cube_hdulist)
        hi_data = noise_cube.unmasked_data[:, :, :].value
        noise_spectral = noise_cube.spectral_axis
        cat_data = fits.getdata(cat_df[cat_df.file.str.contains(mos_name)].file.unique()[0])
        for i, row in cat_df[cat_df.mos_name==mos_name].iterrows():
            subcube = hi_data[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
            det_subcube = cat_data[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
            det_subcube[det_subcube != row.label] = 0
            det_subcube[det_subcube == row.label] = 1
            dF = 36621.09375*u.Hz
            gal_dF = 24208*u.Hz
            dx = 0.001666666707*u.deg
            h_0 = 70*u.km/(u.Mpc*u.s)
            rest_freq = 1.420405758000E+09*u.Hz
            dist = (row.dist*u.Mpc)
            redshift = h_0*dist/const.c
            chosen_f = rest_freq/(redshift + 1)
            noise_rest_vel = (const.c*(dF/chosen_f)).to(u.km/u.s)
            rest_vel = (const.c*(gal_dF/rest_freq)).to(u.km/u.s)
            dF_scale = float(rest_vel/noise_rest_vel)
            try:
                mass = hi_mass(subcube*det_subcube, gal_dF*dF_scale, chosen_f, dist, redshift, noise_spectral, row['centroid-0'])
            except:
                mass = np.nan
            cat_df.loc[i, "hi_mass"] = mass
            print("\r", i*100/len(cat_df[cat_df.mos_name==mos_name]), "%", end="")
            '''
            y, x = (np.where(moment_0 == np.max(moment_0)))
            if (len(x) == 1) & (len(y) == 1):
                mpx, mpy = float((centroid_com(moment_0)[0] + x)/2), float((centroid_com(moment_0)[1] + y)/2)
                dx = mpx - (moment_0.shape[1]/2)
                dy = mpy - (moment_0.shape[0]/2)
                z1, z2 = row['bbox-0'], row['bbox-3']
                if (dx > 0) & (dy < 0):
                    x1, x2 = int(row['bbox-1'] + dy), int(row['bbox-4'])
                    y1, y2 = row['bbox-2'], int(row['bbox-2'] + 2*mpx)
                if (dx > 0) & (dy > 0):
                    x1, x2 = int(row['bbox-1']), int(row['bbox-1'] + 2*mpy)
                    y1, y2 = row['bbox-2'], int(row['bbox-2'] + 2*mpx)
                if (dx < 0) & (dy < 0):
                    x1, x2 = int(row['bbox-1'] + dy), int(row['bbox-4'])
                    y1, y2 = int(row['bbox-2']+dx), row['bbox-5']
                if (dx < 0) & (dy > 0):
                    x1, x2 = int(row['bbox-1']), int(row['bbox-1'] + 2*mpy)
                    y1, y2 = int(row['bbox-2']+dx), row['bbox-5']
                new_cube = hi_data[z1:z2, x1:x2, y1:y2]*cat_data[z1:z2, x1:x2, y1:y2]
                new_cube[cat_data[z1:z2, x1:x2, y1:y2] != int(row.label)] = 0
                moment_0_pad = np.sum(new_cube, axis=0)
                new_cube[new_cube > 0] = 1
                new_cube[new_cube != 1] = 0
                noise_map = np.sqrt(np.sum(new_cube, axis=0))
                noise_map[noise_map == 0] = np.nan
                SNR_map = moment_0_pad/noise_map
                rotated = np.rot90(SNR_map, 2)
                subtracted = np.abs(SNR_map - rotated)
                asymmetry = np.nansum(subtracted)/(np.nansum(SNR_map)+ np.nansum(rotated))
                cat_df.loc[i, "asymmetry"] = asymmetry
            dF = 36621.09375*u.Hz
            dx = 0.001666666707*u.deg
            h_0 = 70*u.km/(u.Mpc*u.s)
            rest_freq = 1.420405758000E+09*u.Hz
            dist = (row.dist*u.Mpc)
            scale = dist*np.tan(np.deg2rad(dx))
            deltaV = (dF*const.c/rest_freq).to(u.km/u.s)
            redshift = h_0*dist/const.c
            S_v = moment_0*deltaV
            mass = np.sum(2.36e5*S_v*dist**2)/(1+redshift)
            cat_df.loc[i, "hi_mass"] = mass.value
            print("\r", i*100/len(cat_df[cat_df.mos_name==mos_name]), "%", end="")
            '''
    cat_df.to_csv("./results/loud_%s_catalog.txt"%method)

