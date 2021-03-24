
"""
Prepare and insert a mock galaxy into a noise cube
"""
from random import randint
import numpy as np
from scipy.ndimage import zoom
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from spectral_cube import SpectralCube
import warnings


# Ignore warning about header
warnings.filterwarnings("ignore", message="Could not parse unit W.U.")


class GalCube:
    def __init__(self, filename):
        """Initialize fits Cube

        Args:
            filename (str): The filename of the cube
        """
        self.filename = filename
        self.orig_d = 50*u.Mpc
        self.h_0 = 70*u.km/(u.Mpc*u.s)
        self.noise_res = [15*u.arcsec, 25*u.arcsec]
    def load_cube(self):
        """Load fits Cube
        """
        gal_cube_hdulist = fits.open(self.filename)
        gal_cube_hdulist[0].header['CTYPE3'] = 'FREQ'
        gal_cube_hdulist[0].header['CUNIT3'] = 'Hz'
        self.gal_cube = SpectralCube.read(gal_cube_hdulist)
        # Convert from W.U. to JY/BEAM
        self.gal_data = self.gal_cube.unmasked_data[:, :, :].value #*5e-3
        gal_cube_hdulist.close()
        # Get spatial pixel sizes
        self.dx = np.abs(self.gal_cube.header["CDELT1"]*u.deg)
        self.dy = self.gal_cube.header["CDELT2"]*u.deg
        self.dF = self.gal_cube.header["CDELT3"]*u.Hz
        self.orig_scale = self.orig_d*np.tan(np.deg2rad(self.dx.to(u.deg)))
        self.rest_freq = self.gal_cube.header['FREQ0']*u.Hz

    def choose_freq(self, noise_cube):
        """Choose frequency channel for insertions

        Args:
            noise_cube (SpectralCube): The noise cube to insert into
        """
        # Find max frequency
        max_pos = noise_cube.shape[0]-self.gal_cube.shape[0]
        max_freq = (self.rest_freq/(1+self.orig_d*self.h_0/const.c)).to(u.Hz)
        idx_range = np.where(np.where((noise_cube.spectral_axis < max_freq))[0] < max_pos)[0]
        # Find insert channel below max freq
        freq_pos = noise_cube.spectral_axis[idx_range]
        # Randomly pick channel within this subset which fits in noise cube
        self.z_pos = randint(0, idx_range[-1])
        self.chosen_f = noise_cube.spectral_axis[self.z_pos]
        # Calculate redshift and distance of channel
        self.new_z = (self.rest_freq/self.chosen_f)-1
        self.new_dist = (const.c*self.new_z/self.h_0).to(u.Mpc)

    def smooth_cube(self):
        """Spatially smooth cube using 2D Gaussian kernel
        """
        # Calculate new spatial resolution
        new_beam_x = (1+self.new_z)*self.noise_res[0]
        new_beam_y = (1+self.new_z)*self.noise_res[1]
        spat_res_x = self.new_dist*np.arctan(np.deg2rad(new_beam_x.to(u.deg).value))
        spat_res_y = self.new_dist*np.arctan(np.deg2rad(new_beam_y.to(u.deg).value))
        # Use the original resolution to find FWHM to smooth to
        fwhm_x = ((spat_res_x/self.orig_scale)*self.dx).to(u.arcsec)
        fwhm_y = ((spat_res_y/self.orig_scale)*self.dy).to(u.arcsec)
        # Find sigma from FWHM
        sigma_x = fwhm_x/np.sqrt(8*np.log(2))
        sigma_y = fwhm_y/np.sqrt(8*np.log(2))
        # Smooth to new channel
        self.smoothed_gal = np.zeros(self.gal_data.shape)
        gauss_kernel = Gaussian2DKernel((sigma_x.to(u.deg)/self.dx).value, (sigma_y.to(u.deg)/self.dy).value)
        gauss_kernel.normalize(mode="peak")
        for idx in range(len(self.gal_data)):
            self.smoothed_gal[idx, :, :] = convolve(self.gal_data[idx, :, :], gauss_kernel, normalize_kernel=False)

    def regrid_cube(self, noise_cube):
        """Resample cube spatially and in the frequency domain

        Args:
            noise_cube (SpectralCube): The noise cube to resample to
        """
        noise_dF = noise_cube.header["CDELT3"]*u.Hz
        new_pix_size_x = self.new_dist*np.tan(np.deg2rad(self.dx.to(u.deg)))
        new_pix_size_y = self.new_dist*np.tan(np.deg2rad(self.dy.to(u.deg)))
        pix_scale_x = self.dx*(new_pix_size_x/self.orig_scale)
        pix_scale_y = self.dy*(new_pix_size_y/self.orig_scale)
        noise_rest_vel = (const.c*(noise_dF/self.chosen_f)).to(u.km/u.s)
        rest_vel = (const.c*(self.dF/self.rest_freq)).to(u.km/u.s)
        dF_scale = noise_rest_vel/rest_vel
        # Regrid cube to new distance and pixel sizes
        self.resampled = zoom(self.smoothed_gal, (float(dF_scale), float(pix_scale_x/self.dx), float(pix_scale_y/self.dy)))

    def rescale_cube(self, noise_cube):
        """Rescale flux of galaxy cube

        Args:
            noise_cube (SpectralCube): The noise cube to insert into
        """
        flux_scale = (self.orig_d/self.new_dist)**2
        self.scaled_flux = flux_scale*self.resampled

    def insert_gal(self, noise_data, empty_cube):
        """Inserts galaxy randomly into given cube

        Args:
            noise_data (numpy.array): 3D array of noise cube to insert it to
            empty_cube (numpy.array): Empty 3D array the shape of cube_data
        """
        # Randomly place galaxy in x and y direction and fill whole z
        x_pos = randint(0, noise_data.shape[1]-self.gal_data.shape[1])
        y_pos = randint(0, noise_data.shape[2]-self.gal_data.shape[2])
        noise_data[
            self.z_pos:self.gal_data.shape[0]+self.z_pos,
            x_pos:self.gal_data.shape[1]+x_pos,
            y_pos:self.gal_data.shape[2]+y_pos
            ] += self.gal_data
        masked = (self.gal_data > np.mean(self.gal_data) + np.std(self.gal_data)).astype(int)
        empty_cube[
            self.z_pos:self.gal_data.shape[0]+self.z_pos,
            x_pos:self.gal_data.shape[1]+x_pos,
            y_pos:self.gal_data.shape[2]+y_pos
            ] += masked
