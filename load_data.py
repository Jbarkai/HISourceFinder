from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from spectral_cube import SpectralCube
from astropy.utils.data import download_file
from astropy import units as u
from astropy import wcs


def load_data():
    # Get Noise
    noise_data = fits.open('data/U6805.lmap.fits')
    noise_data[0].header['CTYPE3'] = 'FREQ'
    noise_data[0].header['CUNIT3'] = 'Hz'
    rest_freq = noise_data[0].header['FREQR']*u.Hz
    noise_cube = SpectralCube.read(noise_data)
    noise_data.close()
    # Get HI data
    hi_datafile = download_file(
    'http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits',
    cache=True, show_progress=True)
    hi_data = fits.open(hi_datafile)  # Open the FITS file for reading
    hi_cube = SpectralCube.read(hi_data)  # Initiate a SpectralCube
    hi_data.close()  # Close the FITS file - we already read it in and don't need it anymore!
    return noise_cube, hi_cube, rest_freq
    
# If you want to take a subcube
# _, b, _ = cube.world[0, :, 0]  #extract latitude world coordinates from cube
# _, _, l = cube.world[0, 0, :]  #extract longitude world coordinates from cube
# # Define desired latitude and longitude range
# lat_range = [-46, -40] * u.deg
# lon_range = [306, 295] * u.deg

# Create a sub_cube cut to these coordinates
# sub_cube = cube.subcube(xlo=lon_range[0], xhi=lon_range[1], ylo=lat_range[0], yhi=lat_range[1])
# sub_cube_slab = sub_cube.spectral_slab(-300. *u.km / u.s, 300. *u.km / u.s)