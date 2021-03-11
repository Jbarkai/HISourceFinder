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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class Cube:
    def __init__(self, filename, set_wcs=None):
        self.filename = filename
        self.set_wcs = set_wcs
    def load_cube(self, ctype=False, scale=False):
        hi_data = fits.open(self.filename)
        if ctype:
            hi_data[0].header['CTYPE3'] = 'FREQ'
        if self.set_wcs:
            self.wcs = WCS(key=self.set_wcs)
        else:
            self.wcs = WCS(hi_data[0].header)
        self.header = hi_data[0].header
        self.cube_data = hi_data[0].data
        if scale:
            self.cube_data = self.cube_data*5e-3
        hi_data.close()

    def rescale_cube(self, dim=(512, 512)):
        # Resize image
        img_stack_sm = np.zeros((self.cube_data.shape[0], dim[0], dim[1]))
        for idx in range(len(self.cube_data)):
            img = self.cube_data[idx, :, :]
            img_sm = cv2.resize(img, (dim[1], dim[0]), interpolation=cv2.INTER_CUBIC)
            img_stack_sm[idx, :, :] = img_sm
        self.cube_data = img_stack_sm

    def smooth_cube(self, old_header, telescope_resolution=15*u.arcsecond):
        # calculate the sigma in pixels.
        # Smooth using Gaussian with sigma = old_pixel_size/new_pixel_size in each direction
        sigma1 = old_header['CDELT1']/self.header['CDELT1']
        sigma2 = old_header['CDELT1']/self.header['CDELT2']
        sigma3 = old_header['CDELT1']/self.header['CDELT3']
        convolved_image = ndi.gaussian_filter(self.cube_data, sigma=(sigma1, sigma2, sigma3), order=0)
        self.cube_data = convolved_image

    def crop_cube(self, scale=1):
        # Crop around galaxy
        true_points = np.argwhere(self.cube_data > np.nanmean(self.cube_data))
        c1 = true_points.min(axis=0)
        c2 = true_points.max(axis=0)
        cropped = self.cube_data[:, c1[1]:c2[1]+1, c1[2]:c2[2]+1]
        self.cube_data = cropped

    def create_mask(self):
        # Find Sources
        self.masked = (self.cube_data > np.nanmean(self.cube_data) + np.std(self.cube_data)).astype(int)*self.cube_data

    def plot_slice(self, slice_i=10, sliced=True, norm=True):
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
        
        newcmp = LinearSegmentedColormap('ds9aips0', segmentdata=ds9aips0)
        fig = plt.figure()
        if sliced:
            ax = fig.add_subplot(projection=self.wcs, slices=('y', 'x', slice_i))
        else:
            ax = fig.add_subplot(projection=self.wcs)
        if norm:
            norm = ImageNormalize(self.cube_data[slice_i, :, :], interval=ZScaleInterval())
            im = ax.imshow(self.cube_data[slice_i, :, :], origin='lower', cmap=newcmp, norm=norm)
        else:
            im = ax.imshow(self.cube_data[slice_i, :, :], origin='lower', cmap=newcmp)
        cbar = plt.colorbar(im)
        cbar.set_label('Frequency (Hz)', size=16)
        plt.xlabel('Right Ascension (J2000)', fontsize=16)
        plt.ylabel('Declination (J2000)', fontsize=16)
        plt.show()
