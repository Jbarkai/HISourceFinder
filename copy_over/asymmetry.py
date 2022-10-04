from astropy import units as u
import astropy.constants as const
from astropy.io import fits
import numpy as np
import pandas as pd
from photutils.centroids import centroid_com
from reproject import reproject_interp
import socket
import argparse
import os
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.wcs import utils
from astropy.utils.data import clear_download_cache
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def getimages(ra,dec,size=240,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    try:
        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
            "&filters={filters}").format(**locals())
        table = Table.read(url, format='ascii')
        return table
    except FileNotFoundError:
        return False

def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
    
    """Get URL for images in the table
    
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    print(size)
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters)
    if type(table) == bool:
        return False
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def get_opt(new_wcs, ra_pix=1030, dec_pix=1030, size_pix=100, d_width=0.001666666707*u.deg):
    try:
        ex_co_ords = utils.pixel_to_skycoord(ra_pix, dec_pix, new_wcs).to_string().split(" ")
        pix_size = int((size_pix*d_width.to(u.arcsec))/(0.25*u.arcsec))
        fitsurl = geturl(float(ex_co_ords[0]), float(ex_co_ords[1]), size=pix_size, filters="i", format="fits")
        if type(fitsurl) == bool:
            return False, 0
        fh = fits.open(fitsurl[0])
    
        fim = fh[0].data
        # replace NaN values with zero for display
        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)
        fh.close()
        clear_download_cache()
        return bfim, fh[0].header
    except socket.timeout:
        print("The read operation timed out")
        return False, 0

def SNR_map(moment, subcube):
    binary_cube = subcube.unmasked_data[:].value
    binary_cube[binary_cube > 0] = 1
    binary_cube[binary_cube != 1] = 0
    noise_map = np.sqrt(np.sum(binary_cube, axis=0))
    noise_map[noise_map == 0] = np.nan
    return moment.value/noise_map

def add_pad(file, label, spec_cube, optical_reprojected, im_dims):
    z1, z2, x1, x2, y1, y2 = im_dims
    cube = fits.getdata(file)[z1:z2, x1:x2, y1:y2]
    cube[cube == label] = 1
    cube[cube != 1] = 0
    subcube = SpectralCube(spec_cube.unmasked_data[z1:z2, x1:x2, y1:y2]*cube, wcs=spec_cube.wcs)
    moment = subcube.with_spectral_unit(u.Hz).moment(order=0)
    SNR = SNR_map(moment, subcube)
    bound = np.nanstd(SNR)
    masked = (SNR > 2*bound)*optical_reprojected
    opt_center = centroid_com(masked)
    # ADD ALL CASES
    dx1 = opt_center[0]
    dy1 = opt_center[1]
    dx2 = masked.shape[1] - opt_center[0]
    dy2 = masked.shape[0] - opt_center[1]
    top, bottom, left, right = 0, 0, 0, 0
    if (dx1 < dx2) & (dy1 < dy2): # pad left & bottom
        left = dx2 - dx1
        bottom = dy2 - dy1
    if (dx1 > dx2) & (dy1 > dy2): # pad right & top
        right = dx1 - dx2
        top = dy1 - dy2
    if (dx1 > dx2) & (dy1 < dy2): # pad right & bottom
        right = dx1 - dx2
        bottom = dy2 - dy1
    if (dx1 < dx2) & (dy1 > dy2): # pad left & top
        left = dx2 - dx1
        top = dy1 - dy2
    return [(int(top), int(bottom)), (int(left), int(right))]

def gal_asym(row, spec_cube, suffix="_sofia"):
    z1 = int(row['bbox-0%s'%suffix])
    z2 = int(row['bbox-3%s'%suffix])
    x1 = int(row['bbox-1%s'%suffix])
    x2 = int(row['bbox-4%s'%suffix])
    y1 = int(row['bbox-2%s'%suffix])
    y2 = int(row['bbox-5%s'%suffix])
    cube = fits.getdata(row['file%s'%suffix])[z1:z2, x1:x2, y1:y2]
    cube[cube == row['label%s'%suffix]] = 1
    cube[cube != 1] = 0
    subcube = SpectralCube(spec_cube.unmasked_data[z1:z2, x1:x2, y1:y2]*cube, wcs=spec_cube.wcs)
    moment = subcube.with_spectral_unit(u.Hz).moment(order=0)
    SNR = SNR_map(moment, subcube)
    bound = np.nanstd(SNR)
    SNR[(SNR > 0) & (SNR <= bound)] = 1/3
    SNR[(SNR > bound) & (SNR <= 3*bound)] = 2/3
    SNR[(SNR != 1/3) & (SNR != 2/3) & ~np.isnan(SNR)] = 1
    # or np.nan_to_num(vnet_SNR*vnet_moment).value
    if suffix != "_mask":
        if type(row['type%s'%suffix]) == str:
            # unmasked_moment = spec_cube[z1:z2, x1:x2, y1:y2].with_spectral_unit(u.Hz).moment(order=0)
            # gal = get_opt(unmasked_moment.wcs, ra_pix=unmasked_moment.shape[0]/2, dec_pix=unmasked_moment.shape[1]/2, size_pix=np.max(unmasked_moment.shape))
            # optical_reprojected = reproject_interp(fits.PrimaryHDU(data=gal[0], header=WCS(gal[1]).to_header()), unmasked_moment.header)[0]
            # pad_dims = add_pad(row['file%s'%suffix], row['label%s'%suffix], spec_cube, optical_reprojected, [z1, z2, x1, x2, y1, y2])
            # moment_padded = np.pad(np.nan_to_num(SNR), pad_dims, mode='constant')
            return np.nan
        else:
            moment_padded = np.nan_to_num(SNR)
    else:
        moment_padded = np.nan_to_num(SNR)
    rotated = np.rot90(moment_padded, 2)
    subtracted = np.abs(moment_padded - rotated)
    asymmetry = np.sum(subtracted)/(np.sum(moment_padded)+ np.sum(rotated))
    return asymmetry

def assymetry(row):
    hi_data = fits.open("data/training/InputBoth/loud_" + row.mos_name + ".fits")
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = SpectralCube.read(hi_data)
    hi_data.close()
    if ~np.isnan(row.label_mto):
        mto_asym = gal_asym(row, spec_cube, suffix="_mto")
    else:
        mto_asym = np.nan
    if ~np.isnan(row.label_sofia):
        sof_asym = gal_asym(row, spec_cube, suffix="_sofia")
    else:
        sof_asym = np.nan
    if ~np.isnan(row.label_vnet):
        vnet_asym = gal_asym(row, spec_cube, suffix="_vnet")
    else:
        vnet_asym = np.nan
    if ~np.isnan(row.label_mask):
        mask_asym = gal_asym(row, spec_cube, suffix="_mask")
    else:
        mask_asym = np.nan

    return sof_asym, mto_asym, vnet_asym, mask_asym

    
real_gals = pd.read_csv("real_gals.csv")
asymmetrys = pd.DataFrame()
for i, row in real_gals[real_gals.true_positive_mocks.fillna(False).astype(bool) | real_gals.true_positive_mocks_sofia.fillna(False).astype(bool) | real_gals.true_positive_mocks_vnet.fillna(False).astype(bool)].iterrows():
    print(i*100/len(real_gals))
    sof_asym, mto_asym, vnet_asym, mask_asym = assymetry(row)
    asymmetrys = asymmetrys.append(pd.DataFrame([[row.mos_name, sof_asym, mto_asym, vnet_asym, mask_asym, row.label_sofia, row.label_mto, row.label_vnet, row.label_mask]],
    columns=["mos_name", "sof_asym", "mto_asym", "vnet_asym", "mask_asym", "label_sofia", "label_mto", "label_vnet", "label_mask"]))
asymmetrys.to_csv("mock_asymmetry.csv", index=False)
