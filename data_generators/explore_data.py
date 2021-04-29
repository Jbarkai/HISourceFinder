import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from os import listdir
import argparse
import gc


def get_plot_data(mask, eccentricity, flatness, vol, galdim, tot_flux, peak_flux):
    print(mask)
    cube_data = fits.getdata("./data/training/Input/" + mask)
    shape_needed = cube_data.shape
    new_mask = cube_data > 0
    object_labels = label(new_mask)
    # del new_mask
    # gc.collect()
    some_props = regionprops(object_labels)
    del object_labels
    gc.collect()
    # eigen_vals = [gal.inertia_tensor_eigvals for gal in some_props]
    galdims = [gal.image.shape for gal in some_props]
    # bbs = [gal.bbox for gal in some_props]
    del some_props
    gc.collect()

    # tot_fluxes = [np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]])
    #  for i in range(len(bbs))]
    # peak_fluxes = [np.max(np.sum(cube_data[bbs[i][0]:bbs[i][3], bbs[i][1]:bbs[i][4], bbs[i][2]:bbs[i][5]], axis=0))
    #  for i in range(len(bbs))]
    # tot_flux.append(tot_fluxes)
    # peak_flux.append(peak_fluxes)
    del cube_data
    gc.collect()

    # eccentricities = [e[0]/e[1] for e in eigen_vals]
    # flatnesses = [e[1]/e[2] for e in eigen_vals]
    vols = [np.prod(gal) for gal in galdims]
    pixel_percent = np.sum(vols)/np.prod(shape_needed)
    # eccentricity.append(eccentricities)
    # flatness.append(flatnesses)
    # vol.append(vols)
    # galdim.append(galdims)
    pixel_percents.append(pixel_percent)


def main(root, output_dir):
    tot_flux = []
    peak_flux = []
    eccentricity = []
    flatness = []
    vol = []
    galdim = []
    pixel_percents = []
    masks = [i for i in listdir(root+"Input")if ".fits" in i]


    for mask in masks:
        get_plot_data(mask, eccentricity, flatness, vol, galdim, tot_flux, peak_flux)
        # some_props = get_mask_data(mask, eccentricity, flatness, vol, galdim)
        # get_cube_data(mask, tot_flux, peak_flux, some_props)
        # del some_props
        # gc.collect()

    fig3 = plt.gcf()
    plt.hist(pixel_percents)
    plt.ylabel("Number of Cubes")
    plt.xlabel("Galaxy Pixel Percentage")
    plt.show()
    fig3.savefig(output_dir+'pixel_percents.png')

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
    parser = argparse.ArgumentParser(description="Create training and validation datasets")
    parser.add_argument(
        '--output_dir', type=str, nargs='?', const='default', default="../plots/",
        help='Directory to output plots to')
    parser.add_argument(
        '--root', type=str, nargs='?', const='default', default="../data/training/",
        help='The root directory of the data')
    args = parser.parse_args()

    main(args.root, args.output_dir)
