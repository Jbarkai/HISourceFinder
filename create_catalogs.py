
import skimage.measure as skmeas
import argparse
from os import listdir
import pickle
from astropy.io import fits
import numpy as np
import pandas as pd


def peak_flux(regionmask, intensity):
    return np.nanmax(np.nansum(intensity[regionmask], axis=0))

def tot_flux(regionmask, intensity):
    return np.nansum(intensity[regionmask])


def create_single_catalog(output_file, mask_file, real_file):
    # Load segmentation, real and mask cubes
    mask_data = fits.getdata(mask_file)
    seg_output = fits.getdata(output_file)
    orig_data = fits.getdata(real_file)
    # Number mask
    print("numbering mask...")
    mask_labels = skmeas.label(mask_data)
    # Catalog mask
    print("cataloging mask...")
    mask_df = pd.DataFrame(
        skmeas.regionprops_table(
        mask_labels, orig_data, properties=['label','inertia_tensor_eigvals', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux))
    )
    mask_df["file"] = mask_file
    max_locs = []
    brightest_pix = []
    for i, row in mask_df.iterrows():
        subcube = orig_data[
            int(row['bbox-0']):int(row['bbox-3']),
            int(row['bbox-1']):int(row['bbox-4']),
            int(row['bbox-2']):int(row['bbox-5'])]
        xyz = [row['bbox-0'], row['bbox-1'], row['bbox-2']]
        brightest_pix.append(np.nanmax(subcube))
        max_locs.append([int(i)+k for i, k in zip(np.where(subcube == np.nanmax(subcube)), xyz)])
    mask_df['brightest_pix'] = brightest_pix
    mask_df['max_loc'] = max_locs
    # Catalog segmentation
    print("cataloging segmentation...")
    source_props_df = pd.DataFrame(
        skmeas.regionprops_table(seg_output, orig_data,
        properties=['label','inertia_tensor_eigvals', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux))
    )
    source_props_df["file"] = output_file
    max_locs = []
    brightest_pix = []
    for i, row in source_props_df.iterrows():
        subcube = orig_data[
            int(row['bbox-0']):int(row['bbox-3']),
            int(row['bbox-1']):int(row['bbox-4']),
            int(row['bbox-2']):int(row['bbox-5'])]
        xyz = [row['bbox-0'], row['bbox-1'], row['bbox-2']]
        brightest_pix.append(np.nanmax(subcube))
        max_locs.append([int(i)+k for i, k in zip(np.where(subcube == np.nanmax(subcube)), xyz)])
    source_props_df['brightest_pix'] = brightest_pix
    source_props_df['max_loc'] = max_locs
    source_props_df['true_positive_mocks'] = [i in list(mask_df.max_loc.values) for i in source_props_df.max_loc]
    return source_props_df

def main(data_dir, method, scale, out_dir):
    cube_files = [data_dir + "training/" +scale+"Input/" + i for i in listdir(data_dir+"training/"+scale+"Input") if "_1245mos" in i]
    source_props_df = pd.DataFrame(columns=['label', 'inertia_tensor_eigvals-0', 'inertia_tensor_eigvals-1',
       'inertia_tensor_eigvals-2', 'centroid-0', 'centroid-1', 'centroid-2',
       'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area', 'flux', 'peak_flux', 'brightest_pix', 'max_loc', 'file',
       'true_positive_mocks'])
    for cube_file in cube_files:
        mos_name = cube_file.split("/")[-1].split("_")[-1].split(".fits")[0]
        print(mos_name)
        if method == "MTO":
            nonbinary_im = data_dir + "mto_output/mtocubeout_" + scale + "_" + mos_name+  ".fits"
        elif method == "VNET":
            nonbinary_im = data_dir + "vnet_output/vnet_cubeout_" + scale + "_" + mos_name+  ".fits"
        elif method == "SOFIA":
            nonbinary_im = data_dir + "sofia_output/sofia_" + scale + "_" + mos_name+  "_mask.fits"
        target_file = data_dir + "training/Target/mask_" + cube_file.split("/")[-1].split("_")[-1]
        source_props_df.append(create_single_catalog(nonbinary_im, target_file, cube_file))
    out_file = out_dir + "/" + scale + "_" + method + "_catalog.txt"
    with open(out_file, "wb") as fp:
        pickle.dump(source_props_df, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create catalog from output",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_dir', type=str, nargs='?', const='default', default="data/",
        help='The directory containing the data')
    parser.add_argument(
        '--method', type=str, nargs='?', const='default', default='MTO',
        help='The segmentation method being evaluated')
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default='loud',
        help='The scale of the inserted galaxies')
    parser.add_argument(
        '--output_dir', type=str, nargs='?', const='default', default="results/",
        help='The output directory for the results')
    args = parser.parse_args()

    main(args.data_dir, args.method, args.scale, args.output_dir)
