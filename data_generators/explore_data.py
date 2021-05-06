import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from os import listdir
import argparse
import pickle
import gc

freq_dict = {"1245mosB.derip.fits": [1401252439.970,1425092772.001],
"1245mosC.derip.fits": [1381037599.970,1404877932.001],
"1245mosD.derip.fits": [1360822751.970,1384663084.001],
"1245mosE.derip.fits": [1340607911.970,1364448244.001],
"1245mosF.derip.fits": [1320393067.970,1344233400.001],
"1245mosG.derip.fits": [1300178221.970,1324018554.001],
"1245mosH.derip.fits": [1279963378.910,1303803710.941],
"1353mosB.derip.fits": [1401179199.970,1425019532.001],
"1353mosC.derip.fits": [1380964351.970,1404804684.001],
"1353mosD.derip.fits": [1360749511.970,1384589844.001],
"1353mosE.derip.fits": [1340534667.970,1364375000.001],
"1353mosF.derip.fits": [1320319823.970,1344160156.001],
"1353mosG.derip.fits": [1300104979.970,1323945312.001],
"1353mosH.derip.fits": [1279890136.720,1303730468.751]}

def get_plot_data(mask, root, eccentricity, flatness, vol, galdim, tot_flux, peak_flux, pixel_percents):
    print(mask)
    cube_data = fits.getdata(root + "Target/" + mask)
    shape_needed = cube_data.shape
    object_labels = label(cube_data)
    # del new_mask
    # gc.collect()
    print(len(np.unique(object_labels)))
    some_props = regionprops(object_labels)
    del object_labels
    gc.collect()
    eigen_vals = [gal.inertia_tensor_eigvals for gal in some_props]
    galdims = [gal.image.shape for gal in some_props]
    bbs = [gal.bbox for gal in some_props]
    del some_props
    gc.collect()

    tot_fluxes = [mask] + [np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]])
     for i in range(len(bbs))]
    peak_fluxes = [mask] + [np.max(np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]], axis=0))
     for i in range(len(bbs))]
    del cube_data
    gc.collect()

    eccentricities = [mask] + [e[0]/e[1] for e in eigen_vals]
    flatnesses = [mask] + [e[1]/e[2] for e in eigen_vals]
    vols = [np.prod(gal) for gal in galdims]
    gal_dims = [mask] + galdims
    pixel_percent = [mask] + [np.sum(vols)/np.prod(shape_needed)]
    vols = [mask] + vols

    tot_flux.append(tot_fluxes)
    peak_flux.append(peak_fluxes)
    eccentricity.append(eccentricities)
    flatness.append(flatnesses)
    vol.append(vols)
    galdim.append(galdims)
    pixel_percents.append(pixel_percent)


def main(root, output_dir):
    tot_flux = []
    peak_flux = []
    eccentricity = []
    flatness = []
    vol = []
    galdim = []
    pixel_percents = []
    masks = [i for i in listdir(root+"Target")if ".fits" in i]


    for mask in masks:
        get_plot_data(mask, root, eccentricity, flatness, vol, galdim, tot_flux, peak_flux, pixel_percents)
    with open("../eccentricity.txt", "wb") as fp:
        pickle.dump(eccentricity, fp)
    with open("../flatness.txt", "wb") as fp:
        pickle.dump(flatness, fp)
    with open("../vol.txt", "wb") as fp:
        pickle.dump(vol, fp)
    with open("../galdim.txt", "wb") as fp:
        pickle.dump(galdim, fp)
    with open("../tot_flux.txt", "wb") as fp:
        pickle.dump(tot_flux, fp)
    with open("../peak_flux.txt", "wb") as fp:
        pickle.dump(peak_flux, fp)
    with open("../voxel_percents.txt", "wb") as fp:
        pickle.dump(pixel_percents, fp)
        # some_props = get_mask_data(mask, eccentricity, flatness, vol, galdim)
        # get_cube_data(mask, tot_flux, peak_flux, some_props)
        # del some_props
        # gc.collect()

    # fig3 = plt.gcf()
    # plt.hist(pixel_percents)
    # plt.ylabel("Number of Cubes")
    # plt.xlabel("Galaxy Pixel Percentage")
    # plt.show()
    # fig3.savefig(output_dir+'pixel_percents.png')

    # fig1 = plt.gcf()
    # plt.boxplot(tot_flux)
    # plt.xlabel("Synthetic Noise-free Cube")
    # plt.ylabel("Total Flux")
    # plt.show()
    # fig1.savefig(output_dir+'tot_flux.png')

    # fig1 = plt.gcf()
    # plt.boxplot(peak_flux)
    # plt.xlabel("Synthetic Noise-free Cube")
    # plt.ylabel("Peak Flux")
    # plt.show()
    # fig1.savefig(output_dir+'peak_flux.png')

    # fig1 = plt.gcf()
    # plt.boxplot(eccentricity)
    # plt.xlabel("Synthetic Noise-free Cube")
    # plt.ylabel("Eccentricity")
    # plt.show()
    # fig1.savefig(output_dir+'eccentricity.png')

    # fig1 = plt.gcf()
    # plt.boxplot(flatness)
    # plt.xlabel("Synthetic Noise-free Cube")
    # plt.ylabel("Flatness")
    # plt.show()
    # fig1.savefig(output_dir+'flatness.png')

    # fig4 = plt.gcf()
    # plt.boxplot(vol)
    # plt.xlabel("Synthetic Noise-free Cube")
    # plt.ylabel("Volume")
    # plt.show()
    # fig4.savefig(output_dir+'vol.png')

    # fig2 = plt.gcf()
    # fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,15))
    # ax1.boxplot([i[0] for i in galdim])
    # ax2.boxplot([i[1] for i in galdim])
    # ax3.boxplot([i[2] for i in galdim])
    # ax1.set_ylabel("Number of Channels")
    # ax2.set_ylabel("Width")
    # ax3.set_ylabel("Height")
    # ax3.set_xlabel("Synthetic Noise-free Cube")
    # fig2.tight_layout()
    # plt.show()
    # fig2.savefig(output_dir+'galdim.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exploratory plots of noise-free cubes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output_dir', type=str, nargs='?', const='default', default="../plots/",
        help='Directory to output plots to')
    parser.add_argument(
        '--root', type=str, nargs='?', const='default', default="../data/training/",
        help='The root directory of the data')
    args = parser.parse_args()

    main(args.root, args.output_dir)
