from astropy.io import fits
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy import units as u
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import cv2
from scipy import ndimage as ndi
from matplotlib.cm import register_cmap


class Cube:
    def __init__(self, filename, set_wcs=None):
        self.filename = filename
        self.set_wcs = set_wcs
    def load_cube(self, ctype=False):
        hi_data = fits.open(self.filename)
        if ctype:
            hi_data[0].header['CTYPE3'] = 'FREQ'
        if self.set_wcs:
            self.wcs = WCS(key=self.set_wcs)
        else:
            self.wcs = WCS(hi_data[0].header)
        self.header = hi_data[0].header
        self.cube_data = hi_data[0].data
        hi_data.close()

    def smooth_cube(self, telescope_resolution=15*u.arcsecond, shrink=False, shrink_xy=[50, 50]):
        # calculate the sigma in pixels.
        # Smooth using Gaussian with sigma = tel_res/pixel_size in each direction
        sigma1 = telescope_resolution.to('deg').value/self.header['CDELT1']
        sigma2 = telescope_resolution.to('deg').value/self.header['CDELT2']
        sigma3 = telescope_resolution.to('deg').value/self.header['CDELT3']
        convolved_image = ndi.gaussian_filter(self.cube_data, sigma=(sigma1, sigma2, sigma3), order=0)
        if shrink:
            convolved_image = cv2.resize(convolved_image, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        self.cube_data = convolved_image

    def create_mask(self, scale=1):
        # Find Sources
        self.masked = (self.cube_data > np.mean(self.cube_data) + np.std(self.cube_data)).astype(int)*scale

    def plot_slice(self, slice_i=10, sliced=True):
        # Get ds9 map
        ds9aips0 = {'red': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                                v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                                [0.196, 0.475, 0, 0.373, 0, 0, 1, 1, 1]), 
                    'green': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                                v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                                [0.196, 0, 0, 0.655, 0.596, 0.965, 1, 0.694, 0]),
                    'blue': lambda v : np.select([v < 1/9., v < 2/9., v < 3/9., v < 4/9., v < 5/9.,
                                                v < 6/9., v < 7/9., v < 8/9., v <= 1],
                                                [0.196, 0.608, 0.785, 0.925, 0, 0, 0, 0, 0])}
        register_cmap('ds9aips0', data=ds9aips0)
        fig = plt.figure()
        if sliced:
            ax = fig.add_subplot(projection=self.wcs, slices=('y', 'x', slice_i))
        else:
            ax = fig.add_subplot(projection=self.wcs)
        norm = ImageNormalize(self.cube_data[slice_i, :, :], interval=ZScaleInterval())
        im = ax.imshow(self.cube_data[slice_i, :, :], origin='lower', cmap='ds9aips0', norm=norm)
        cbar = plt.colorbar(im)
        cbar.set_label('Frequency (Hz)', size=16)
        plt.xlabel('Right Ascension (J2000)', fontsize=16)
        plt.ylabel('Declination (J2000)', fontsize=16)
        plt.show()
