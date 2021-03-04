from astropy.io import fits
import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import cv2


class Cube:
    def __init__(self, filename):
        self.filename = filename

    def load_cube(self):
        hi_data = fits.open(self.filename)
        if cube_data[0].header['CTYPE3'] == 'FREQ-OHEL':
            cube_data[0].header['CTYPE3'] = 'FREQ'
        self.header = hi_data[0].header
        self.cube_data = hi_data[0].data
        hi_data.close()

    def smooth_cube(self, freq_slice=471):
        telescope_resolution = 15*u.arcsecond
        # calculate the sigma in pixels.
        sigma = telescope_resolution.to('deg').value/self.header['CDELT2']
        # By default, the Gaussian kernel will go to 4 sigma in each direction
        psf = Gaussian2DKernel(sigma)
        sliced = self.cube_data[freq_slice, :, :]
        convolved_image = convolve_fft(sliced, psf, boundary='wrap')
        # Shrink
        self.convolved_image = cv2.resize(convolved_image, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)

    def create_mask(self):
        # Find Sources
        self.masked = (self.cube_data > np.mean(self.cube_data) + np.std(self.cube_data)).astype(int)

    def plot_slice(self, freq_slice=471, mask=0):
        wcs = WCS(self.header)
        sliced = self.cube_data[freq_slice, :, :]

        fig = plt.figure()
        ax = fig.add_subplot(projection=wcs, slices=('y', 'x', freq_slice))
        norm = ImageNormalize(sliced, interval=ZScaleInterval())
        im = ax.imshow(sliced, origin='lower', cmap='ds9aips0', norm=norm)
        if type(mask) != int:
            ax.contour(mask, cmap='Greys_r')
        cbar = plt.colorbar(im)
        cbar.set_label('Frequency (Hz)', size=16)
        plt.xlabel('Right Ascension (J2000)', fontsize=16)
        plt.ylabel('Declination (J2000)', fontsize=16)
        plt.show()
