import numpy as np
from astropy.io import fits
import skimage.measure as skmeas
from os import listdir
import pickle

def get_vals(noise_data, tot_flux, peak_flux, eccentricity, flatness):
    x = np.random.randint(0, noise_data.shape[0]-15)
    y = np.random.randint(0, noise_data.shape[1]-20)
    z = np.random.randint(0, noise_data.shape[2]-20)
    subcube = noise_data[x:x+15, y:y+20, z:z+20]
    if not np.isnan(subcube).all():
        subcube[np.where(subcube > np.mean(subcube)+3*np.std(subcube))] = 0
        tot_flux += [np.nansum(subcube)]
        peak_flux += [np.nanmax(np.nansum(subcube, axis=0))]
        some_props = skmeas.regionprops((subcube*0+1).astype(int))[0]
        eigen_vals = some_props.inertia_tensor_eigvals
        eccentricity += [eigen_vals[0]/eigen_vals[1]]
        flatness += [eigen_vals[1]/eigen_vals[2]]
    return


def get_means(mosaic, root, mean_tot_flux, mean_peak_flux, mean_eccentricity, mean_flatness):
    tot_flux = []
    peak_flux = []
    eccentricity = []
    flatness = []
    noise_data = np.moveaxis(fits.getdata(root+mosaic), 0, 2)
    for i in range(350):
        get_vals(noise_data, tot_flux, peak_flux, eccentricity, flatness)
    mean_tot_flux += [(mosaic, np.nanmean(tot_flux))]
    mean_peak_flux += [(mosaic, np.nanmean(peak_flux))]
    mean_eccentricity += [(mosaic, np.nanmean(eccentricity))]
    mean_flatness += [(mosaic, np.nanmean(flatness))]
    return

root = "./data/mosaics/"
mosaics = [x for x in listdir(root) if ".fits" in x]
mean_tot_flux = []
mean_peak_flux = []
mean_eccentricity = []
mean_flatness = []
for mosaic in mosaics:
    print(mosaic)
    get_means(mosaic, root, mean_tot_flux, mean_peak_flux, mean_eccentricity, mean_flatness)
with open("./noise_eccentricity.txt", "wb") as fp:
    pickle.dump(mean_eccentricity, fp)
with open("./noise_flatness.txt", "wb") as fp:
    pickle.dump(mean_flatness, fp)
with open("./noise_tot_flux.txt", "wb") as fp:
    pickle.dump(mean_tot_flux, fp)
with open("./noise_peak_flux.txt", "wb") as fp:
    pickle.dump(mean_peak_flux, fp)
