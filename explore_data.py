import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import skimage.measure as skmeas
from os import listdir
import gc


def get_mask_data(mask, eccentricity, flatness, vol, galdim):
    print(mask)
    maskcube_hdulist = fits.open("./data/training/Target/" + mask)
    maskcube_data = maskcube_hdulist[0].data
    maskcube_hdulist.close()
    shape_needed = maskcube_data.shape
    new_mask = maskcube_data > 0
    del maskcube_data
    gc.collect()
    object_labels = skmeas.label(new_mask)
    del new_mask
    gc.collect()
    some_props = skmeas.regionprops(object_labels)
    eigen_vals = [gal.inertia_tensor_eigvals for gal in some_props]
    eccentricities = [e[0]/e[1] for e in eigen_vals]
    flatnesses = [e[1]/e[2] for e in eigen_vals]
    galdims = [gal.image.shape for gal in some_props]
    vols = [np.prod(gal) for gal in galdims]
    pixel_percent = np.sum(vols)/np.prod(shape_needed)
    eccentricity.append(eccentricities)
    flatness.append(flatnesses)
    vol.append(vols)
    galdim.append(galdims)
    pixel_percents.append(pixel_percent)
    return some_props

def get_cube_data(mask, tot_flux, peak_flux, some_props):
    print(mask.split("_")[-1])
    cube_hdulist = fits.open("./data/training/Input/noisefree_" + mask.split("_")[-1])
    cube_data = cube_hdulist[0].data
    cube_hdulist.close()
    bbs = [gal.bbox for gal in some_props]
    tot_fluxes = [np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]])
     for i in range(len(some_props))]
    peak_fluxes = [np.max(np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]], axis=0))
     for i in range(len(some_props))]
    del cube_data
    gc.collect()
    tot_flux.append(tot_fluxes)
    peak_flux.append(peak_fluxes)


tot_flux = []
peak_flux = []
eccentricity = []
flatness = []
vol = []
galdim = []
pixel_percents = []
masks = [i for i in listdir("./data/training/Target")if ".fits" in i]


for mask in masks:
    some_props = get_mask_data(mask, eccentricity, flatness, vol, galdim)
    get_cube_data(mask, tot_flux, peak_flux, some_props)
    break
plt.boxplot(tot_flux)
plt.xlabel("Synthetic Noise-free Cube")
plt.ylabel("Total Flux")
plt.show()
plt.savefig('./plots/tot_flux.png')
plt.boxplot(peak_flux)
plt.xlabel("Synthetic Noise-free Cube")
plt.ylabel("Peak Flux")
plt.show()
plt.savefig('./plots/peak_flux.png')
plt.boxplot(eccentricity)
plt.xlabel("Synthetic Noise-free Cube")
plt.ylabel("Eccentricity")
plt.show()
plt.savefig('./plots/eccentricity.png')
plt.boxplot(flatness)
plt.xlabel("Synthetic Noise-free Cube")
plt.ylabel("Flatness")
plt.show()
plt.savefig('./plots/flatness.png')
plt.boxplot(vol)
plt.xlabel("Synthetic Noise-free Cube")
plt.ylabel("Volume")
plt.show()
plt.savefig('./plots/vol.png')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,15))
ax1.boxplot([i[0] for i in galdim])
ax2.boxplot([i[1] for i in galdim])
ax3.boxplot([i[2] for i in galdim])
ax1.set_ylabel("Number of Channels")
ax2.set_ylabel("Width")
ax3.set_ylabel("Height")
ax3.set_xlabel("Synthetic Noise-free Cube")
fig.tight_layout()
plt.show()
plt.savefig('./plots/galdim.png')
plt.hist(pixel_percents)
plt.ylabel("Number of Cubes")
plt.ylabel("Galaxy Pixel Percentage")
plt.show()
plt.savefig('./plots/pixel_percents.png')
