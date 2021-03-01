from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from spectral_cube import SpectralCube
from astropy.utils.data import download_file
from astropy import units as u


def load_data():
    noise_data = fits.getdata('data/U6805.lmap.fits')
    hi_datafile = download_file(
        'http://data.astropy.org/tutorials/FITS-cubes/reduced_TAN_C14.fits',
        cache=True, show_progress=True)
    hi_data = fits.open(hi_datafile)  # Open the FITS file for reading
    cube = SpectralCube.read(hi_data)  # Initiate a SpectralCube
    hi_data.close()  # Close the FITS file - we already read it in and don't need it anymore!

    moment_0 = cube.with_spectral_unit(u.km/u.s).moment(order=0)  # Zero-th moment
    moment_1 = cube.with_spectral_unit(u.km/u.s).moment(order=1)  # First moment
    hi_column_density = moment_0 * 1.82 * 10**18 / (u.cm * u.cm) * u.s / u.K / u.km
    
    return noise_data, moment_0, moment_1, hi_column_density
    
# If you want to take a subcube
# _, b, _ = cube.world[0, :, 0]  #extract latitude world coordinates from cube
# _, _, l = cube.world[0, 0, :]  #extract longitude world coordinates from cube
# # Define desired latitude and longitude range
# lat_range = [-46, -40] * u.deg
# lon_range = [306, 295] * u.deg

# Create a sub_cube cut to these coordinates
# sub_cube = cube.subcube(xlo=lon_range[0], xhi=lon_range[1], ylo=lat_range[0], yhi=lat_range[1])
# sub_cube_slab = sub_cube.spectral_slab(-300. *u.km / u.s, 300. *u.km / u.s)