import numpy as np
import socket
import argparse
import pandas as pd
import os
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.wcs import utils
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube

def getimages(ra,dec,size=240,filters="grizy"):
    
    """Query ps1filenames.py service to get a list of images
    
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


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
    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters)
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


def get_opt(new_wcs, ra_pix=1030, dec_pix=1030, size_pix=100):
    try:
        d_width = 0.001666666707*u.deg
        ex_co_ords = utils.pixel_to_skycoord(ra_pix, dec_pix, new_wcs).to_string().split(" ")
        pix_size = int((size_pix*d_width.to(u.arcsec))/(0.25*u.arcsec))
        fitsurl = geturl(float(ex_co_ords[0]), float(ex_co_ords[1]), size=pix_size, filters="i", format="fits")
        fh = fits.open(fitsurl[0])
        fim = fh[0].data
        # replace NaN values with zero for display
        fim[np.isnan(fim)] = 0.0
        # set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        bfim = transform(fim)
        fh[0].data = bfim
        return fh[0]
    except socket.timeout:
        print("The read operation timed out")
        return False

def overlay_hi(row, method, spec_cube, output_file="./optical_catalogs/"):
    subcube = spec_cube[row['bbox-0']:row['bbox-3'], row['bbox-1']-int(row.nx*0.5):row['bbox-4']+int(row.nx*0.5), row['bbox-2']-int(row.ny*0.5):row['bbox-5']+int(row.ny*0.5)]
    sof_data = fits.getdata(row.file)
    masked = SpectralCube(subcube.unmasked_data[:]*sof_data[row['bbox-0']:row['bbox-3'], row['bbox-1']-int(row.nx*0.5):row['bbox-4']+int(row.nx*0.5), row['bbox-2']-int(row.ny*0.5):row['bbox-5']+int(row.ny*0.5)], wcs=subcube.wcs)
    moment_0 = masked.with_spectral_unit(u.Hz).moment(order=0)
    gal = get_opt(moment_0.wcs, ra_pix=moment_0.shape[0]/2, dec_pix=moment_0.shape[1]/2, size_pix=np.max(moment_0.shape))

    if type(gal) == fits.hdu.image.PrimaryHDU:
        ax = plt.subplot(projection=moment_0.wcs)
        ax.contour(moment_0, zorder=1, origin='lower')
        ax.imshow(gal.data, transform=ax.get_transform(WCS(gal.header)), zorder=0, origin='lower')
        ax.set_xlim((0,moment_0.shape[0]))
        ax.set_ylim((0,moment_0.shape[1]))
        plt.savefig(output_file + method + "_" + row.mos_name + "_" + str(row.label) + ".png")

def main(method, output_file):
    cat_df = pd.read_csv("./results/loud_%s_catalog.txt"%method, index_col=0)
    for mos_name in cat_df.mos_name.unique():
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
                overlay_hi(row, method, spec_cube, output_file)
                print("\r", i*100/len(subset), "%", end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay HI moment 0 map on optical cross-matched catalog",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--method', type=str, nargs='?', const='default', default="SOFIA",
        help='The method to extract catalogs from')
    parser.add_argument(
        '--output_file', type=str, nargs='?', const='default', default='./optical_catalogs/',
        help='The output file for the images')
    args = parser.parse_args()

    main(args.method, args.output_file)
