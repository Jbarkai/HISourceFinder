import numpy as np
import socket
import argparse
import pandas as pd
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


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def two_gaussians(x, h1, c1, w1, h2, c2, w2):
    return Gauss(x, h1, c1, w1) + Gauss(x, h2, c2, w2)

def fit_gauss(subcube, ax):
    x = np.arange(subcube.shape[0])
    y = [np.nansum(subcube[k]) for k in range(subcube.shape[0])]
    ax.plot(x, y, 'b+:', label='data')
    try:
        # weighted arithmetic mean (corrected - check the section below)
        peaks, properties = find_peaks(y, distance=25)
        peak_x = [x[y == sorted(np.array(y)[peaks])[-2:][0]][0], x[y == sorted(np.array(y)[peaks])[-2:][1]][0]]
        peak_y = sorted(np.array(y)[peaks])[-2:]
        sigma1 = np.sqrt(sum(y * (x - peak_x[0])**2) / sum(y))
        sigma2 = np.sqrt(sum(y * (x - peak_x[1])**2) / sum(y))
        popt, pcov = curve_fit(two_gaussians, x, y, p0=[
            peak_y[0], peak_x[0], sigma1,
            peak_y[1], peak_x[1], sigma2])
        residuals = y- two_gaussians(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        ax.plot(x, two_gaussians(x, *popt), 'g-', label='fit')
    except:
        r_squared = np.nan
    return


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

def overlay_hi(row, method, spec_cube, output_file="./optical_catalogs/", d_width=0.001666666707*u.deg):
    max_size = np.max([row.nx, row.ny])
    dx = int(max_size*0.5)
    det_data = fits.getdata(row.file)
    subcube = spec_cube[row['bbox-0']:row['bbox-3'], row['bbox-1']-dx:row['bbox-4']+dx, row['bbox-2']-dx:row['bbox-5']+dx]
    det_subcube = det_data[row['bbox-0']:row['bbox-3'], row['bbox-1']-dx:row['bbox-4']+dx, row['bbox-2']-dx:row['bbox-5']+dx]
    det_subcube[:, :dx] = 0
    det_subcube[:, :, :dx] = 0
    det_subcube[:, -dx:] = 0
    det_subcube[:, :, -dx:] = 0
    masked = SpectralCube(subcube.unmasked_data[:]*det_subcube, wcs=subcube.wcs)
    try:
        masked_moment_0 = masked.with_spectral_unit(u.Hz).moment(order=0)
        masked_moment_1 = masked.with_spectral_unit(u.Hz).moment(order=1)
        unmasked_moment_0 = subcube.with_spectral_unit(u.Hz).moment(order=0)
    except IndexError:
        print("Index error")
        return

    gal, gal_header = get_opt(unmasked_moment_0.wcs, ra_pix=unmasked_moment_0.shape[0]/2, dec_pix=unmasked_moment_0.shape[1]/2, size_pix=np.max(unmasked_moment_0.shape), d_width=d_width)
    if type(gal) != bool:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(2, 2, 1, projection=masked_moment_0.wcs)
        ax.contour(masked_moment_0, zorder=2, origin='lower', cmap="Reds")
        ax.imshow(gal.data, transform=ax.get_transform(WCS(gal_header)), zorder=0, origin='lower')
        ax.axis('off')
        ax.set_xlim((0,masked_moment_0.shape[0]))
        ax.set_ylim((0,masked_moment_0.shape[1]))

        ax1 = fig.add_subplot(2, 2, 4, projection=unmasked_moment_0.wcs)
        ax1.axis('off')
        ax1.contour(unmasked_moment_0, zorder=1, origin='lower', cmap="Reds")
        ax1.imshow(gal.data, transform=ax1.get_transform(WCS(gal_header)), zorder=0, origin='lower')
        ax1.set_xlim((0,unmasked_moment_0.shape[0]))
        ax1.set_ylim((0,unmasked_moment_0.shape[1]))

        ax2 = fig.add_subplot(2, 2, 2, projection=masked_moment_0.wcs)
        ax2.axis('off')
        ax2.contour(masked_moment_0, zorder=1, origin='lower', cmap="Reds")
        ax2.imshow(masked_moment_1.value, zorder=0, origin='lower', alpha=0.6)
        ax2.set_xlim((0,unmasked_moment_0.shape[0]))
        ax2.set_ylim((0,unmasked_moment_0.shape[1]))

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis('off')
        fit_gauss(unmasked_moment_0.hdu.data, ax3)

        fig.tight_layout(pad=0)
        plt.savefig(output_file + method + "_" + row.mos_name + "_" + str(row.label) + ".png")

def main(method, output_file):
    cat_df = pd.read_csv("./results/loud_%sexp5_2_catalog.txt"%method, index_col=0)
    noise_res = [(15*u.arcsec).to(u.deg), (25*u.arcsec).to(u.deg)]
    kpc_lim = [0, 300]
    n_vel_lim = [7, 750]
    d_width = 0.001666666707*u.deg
    cat_df["nx_kpc"] = cat_df.dist*np.tan(np.deg2rad(d_width*cat_df.nx))*1e3
    cat_df["ny_kpc"] = cat_df.dist*np.tan(np.deg2rad(d_width*cat_df.ny))*1e3
    cond = (
        (cat_df.nx*d_width < noise_res[0]) | (cat_df.ny*d_width < noise_res[1]) | 
        (cat_df.ny_kpc > kpc_lim[1]) | (cat_df.nx_kpc > kpc_lim[1]) |
        (cat_df.n_vel > n_vel_lim[1]) | (cat_df.n_vel < n_vel_lim[0])
    )
    cat_df = cat_df[~cond]
    for mos_name in np.sort(cat_df.mos_name.unique()):
        print(mos_name)
        hi_data = fits.open("./data/orig_mosaics/%s.derip.fits"%mos_name)
        hi_data[0].header['CTYPE3'] = 'FREQ'
        hi_data[0].header['CUNIT3'] = 'Hz'
        spec_cube = SpectralCube.read(hi_data)
        hi_data.close()

        subset = cat_df[~cat_df.true_positive_mocks & (cat_df.mos_name==mos_name)]
        for i, row in subset.iterrows():
            file_name = output_file + method + "_" + row.mos_name + "_" + str(row.label) + ".png"
            if os.path.isfile(file_name):
                pass
            else:
                overlay_hi(row, method, spec_cube, output_file, d_width)
                print("\r", i*100/len(subset), "%", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay HI moment 0 map on optical cross-matched catalog",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--method', type=str, nargs='?', const='default', default="VNET",
        help='The method to extract catalogs from')
    parser.add_argument(
        '--output_file', type=str, nargs='?', const='default', default='./',
        help='The output file for the images')
    args = parser.parse_args()

    main(args.method, args.output_file)
