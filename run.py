#%%
from load_data import load_data
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import astropy_mpl_style


def plot_cube(title, cube_data, mask=0, vmin=0, vmax=200, rest_value=None, velocity_convention=None):
    spec_cube = cube_data.with_spectral_unit(u.km/u.s, rest_value=rest_value, velocity_convention=velocity_convention)
    moment_0 = spec_cube.moment(order=0)  # Zero-th moment
    moment_1 = spec_cube.moment(order=1)  # First moment
    hi_column_density = moment_0 * 1.82 * 10**18 / (u.cm * u.cm) * u.s / u.K / u.km
    # Initiate a figure and axis object with WCS projection information
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=moment_1.wcs)

    # Display the moment map image
    im = ax.imshow(moment_1.hdu.data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.invert_yaxis()  # Flips the Y axis

    # Add axes labels
    ax.set_xlabel("Galactic Longitude (degrees)", fontsize=16)
    ax.set_ylabel("Galactic Latitude (degrees)", fontsize=16)

    # Add a colorbar
    cbar = plt.colorbar(im, pad=.07)
    cbar.set_label('Velocity (km/s)', size=16)

    # Overlay set of RA/Dec Axes
    overlay = ax.get_coords_overlay('fk5')
    overlay.grid(color='white', ls='dotted', lw=2)
    overlay[0].set_axislabel('Right Ascension (J2000)', fontsize=16)
    overlay[1].set_axislabel('Declination (J2000)', fontsize=16)
    # Overplot column density contours
    if mask == 0:
        ax.contour(hi_column_density, cmap='Greys_r')
    else:
        ax.contour(mask, cmap='Greys_r')
    fig.tight_layout()
    fig.savefig('Plots/%s.png'%title)

def main():
    noise_cube, hi_cube, rest_freq = load_data()
    plot_cube("noise", noise_cube, vmin=-2, vmax=2, rest_value=rest_freq, velocity_convention=u.doppler_radio)
    plot_cube("HI", hi_cube)
    return 0

if __name__ == "__main__":
    main()