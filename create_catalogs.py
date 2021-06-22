
import skimage.measure as skmeas
import argparse
from os import listdir
import pickle
from astropy.io import fits
import numpy as np
from astropy import units as u
import astropy.constants as const
import pandas as pd
from astropy.coordinates import SkyCoord
from spectral_cube import SpectralCube


def peak_flux(regionmask, intensity):
    return np.nanmax(np.nansum(intensity[regionmask], axis=0))

def tot_flux(regionmask, intensity):
    return np.nansum(intensity[regionmask])


def create_mask_catalog(mask_file, real_file):
    # Load segmentation, real and mask cubes
    mask_data = fits.getdata(mask_file)
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
    mask_df["file"] = mask_file
    mask_df['n_channels'] = mask_df['bbox-3']-mask_df['bbox-0']
    # Convert to physical values
    d_channels = 36621.09375*u.Hz
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.065000001573*u.deg
    mask_df["n_vel"] = [(i*d_channels*const.c/rest_freq).to(u.km/u.s).value for i in mask_df.n_channels]
    mask_df['nx'] = mask_df['bbox-4']-mask_df['bbox-1']
    h_0 = 70*u.km/(u.Mpc*u.s)
    # Load reference cube
    hi_data = fits.open(real_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = (SpectralCube.read(hi_data)).spectral_axis
    hi_data.close()
    mask_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in mask_df['centroid-0'].astype(int)]
    mask_df["nx_mpc"] = 2*mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.nx/2))
    mask_df['ny'] = mask_df['bbox-5']-mask_df['bbox-2']
    mask_df["ny_mpc"] = 2*mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.ny/2))
    print(len(mask_df))
    return mask_df


def create_single_catalog(output_file, mask_file, real_file, catalog_df):
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
    source_props_df['elongation'] = source_props_df['inertia_tensor_eigvals-0']/source_props_df['inertia_tensor_eigvals-1']
    source_props_df['flatness'] = source_props_df['inertia_tensor_eigvals-1']/source_props_df['inertia_tensor_eigvals-2']
    max_locs = []
    brightest_pix = []
    # masks = []
    # gt_masks = []
    for i, row in source_props_df.iterrows():
        subcube = orig_data[
            int(row['bbox-0']):int(row['bbox-3']),
            int(row['bbox-1']):int(row['bbox-4']),
            int(row['bbox-2']):int(row['bbox-5'])]
        # mask_subcube = seg_output[
        #     int(row['bbox-0']):int(row['bbox-3']),
        #     int(row['bbox-1']):int(row['bbox-4']),
        #     int(row['bbox-2']):int(row['bbox-5'])]
        # gt_subcube = mask_labels[
        #     int(row['bbox-0']):int(row['bbox-3']),
        #     int(row['bbox-1']):int(row['bbox-4']),
        #     int(row['bbox-2']):int(row['bbox-5'])]
        # mask_subcube[mask_subcube > 0] = 1
        # gt_subcube[gt_subcube > 0] = 1
        xyz = [row['bbox-0'], row['bbox-1'], row['bbox-2']]
        brightest_pix.append(np.nanmax(subcube))
        max_locs.append([int(i)+k for i, k in zip(np.where(subcube == np.nanmax(subcube)), xyz)])
        # masks.append(mask_subcube)
        # gt_masks.append(gt_subcube)
    source_props_df['brightest_pix'] = brightest_pix
    source_props_df['max_loc'] = max_locs
    # source_props_df['seg_mask'] = masks
    # source_props_df['gt_mask'] = gt_masks
    source_props_df["file"] = output_file
    source_props_df['true_positive_mocks'] = [i in list(mask_df.max_loc.values) for i in source_props_df.max_loc]
    source_props_df['true_positive_real'] = False
    # Update real catalog with pixel values
    print("cross-referencing with real catalog")
    get_pixel_coords(real_file, catalog_df)
    file_abb = real_file.split("/")[-1].split(".")[0]
    real_cat = catalog_df[catalog_df.file_name == file_abb]
    source_cat = (source_props_df.file.str.contains(file_abb)) & (~source_props_df.true_positive_mocks)
    for i, row in real_cat.iterrows():
        source_cond = (
            (row.z_pos >= source_props_df[source_cat]['bbox-0']) & (row.z_pos >= source_props_df[source_cat]['bbox-3'])
            & (row.pixels_x >= source_props_df[source_cat]['bbox-1']) & (row.pixels_x >= source_props_df[source_cat]['bbox-4'])
            & (row.pixels_y >= source_props_df[source_cat]['bbox-2']) & (row.pixels_y >= source_props_df[source_cat]['bbox-5'])
        )
        source_props_df.loc[source_cat, 'true_positive_real'] = source_cond
    source_props_df['n_channels'] = source_props_df['bbox-3']-source_props_df['bbox-0']
    # Convert to physical values
    d_channels = 36621.09375*u.Hz
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.065000001573*u.deg
    source_props_df["n_vel"] = [(i*d_channels*const.c/rest_freq).to(u.km/u.s).value for i in source_props_df.n_channels]
    source_props_df['nx'] = source_props_df['bbox-4']-source_props_df['bbox-1']
    h_0 = 70*u.km/(u.Mpc*u.s)
    # Load reference cube
    hi_data = fits.open(real_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = (SpectralCube.read(hi_data)).spectral_axis
    hi_data.close()
    source_props_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in source_props_df['centroid-0'].astype(int)]
    source_props_df["nx_mpc"] = 2*source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.nx/2))
    source_props_df['ny'] = source_props_df['bbox-5']-source_props_df['bbox-2']
    source_props_df["ny_mpc"] = 2*source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.ny/2))
    print(len(source_props_df))
    return source_props_df


def get_pixel_coords(cube_file, catalog_df):
    # Load reference cube
    hi_data = fits.open(cube_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    orig_header = hi_data[0].header
    cube = SpectralCube.read(hi_data)
    hi_data.close()
    # Find frequencies withing cube
    cube_freqs = []
    for freq in catalog_df.freq:
        matching = cube.spectral_axis[
            (cube.spectral_axis <= freq*u.Hz + orig_header['CDELT3']*u.Hz) 
            & (cube.spectral_axis >= freq *u.Hz- orig_header['CDELT3']*u.Hz)
        ]
        if len(matching) < 1:
            cube_freq = np.nan
        else:
            # Find channel
            cube_freq = np.where(cube.spectral_axis == min(matching, key=lambda x:abs(x-freq*u.Hz)))[0][0]
        cube_freqs.append(cube_freq)
    # Allocate channels to galaxies in cube
    catalog_df.loc[~np.isnan(cube_freqs), 'z_pos'] = np.array(cube_freqs)[~np.isnan(cube_freqs)]
    catalog_df.loc[~np.isnan(cube_freqs), "file_name"] = cube_file.split("/")[-1].split(".")[0]
    # Get x-y co-ords
    co_ords = SkyCoord(ra=catalog_df.loc[~np.isnan(cube_freqs), 'RA_d'].values*u.deg,
               dec=catalog_df.loc[~np.isnan(cube_freqs), 'DEC_d'].values*u.deg,
               distance=catalog_df.loc[~np.isnan(cube_freqs), 'freq'].values*u.Hz
                      ).to_pixel(cube.wcs)
    catalog_df.loc[~np.isnan(cube_freqs), 'pixels_x'] = co_ords[0]
    catalog_df.loc[~np.isnan(cube_freqs), 'pixels_y'] = co_ords[1]
    return


def main(data_dir, method, scale, out_dir, catalog_loc, mask):
    cube_files = [data_dir + "training/" +scale+"Input/" + i for i in listdir(data_dir+"training/"+scale+"Input") if ".fits" in i]
    if mask:
        source_props_df_full = pd.DataFrame(columns=['label', 'inertia_tensor_eigvals-0',
        'inertia_tensor_eigvals-1', 'inertia_tensor_eigvals-2', 'centroid-0', 'centroid-1',
        'centroid-2', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
        'flux', 'peak_flux', 'elongation', 'flatness', 'brightest_pix', 'max_loc',
        'file', 'true_positive_mocks', 'true_positive_real', 'n_channels', 'n_vel', 'nx',
        'dist', 'nx_mpc', 'ny', 'ny_mpc'])
        for cube_file in cube_files:
            print(cube_file)
            target_file = data_dir + "training/Target/mask_" + cube_file.split("/")[-1].split("_")[-1]
            source_props_df = create_mask_catalog(target_file, cube_file)
            source_props_df_full = source_props_df_full.append(source_props_df)
        print("saving file...")
        out_file = out_dir + "/" + scale + "_" + method + "_catalog.txt"
        source_props_df_full.to_csv(out_file)
        return
    h_0 = 70*u.km/(u.Mpc*u.s)
    rest_freq = 1.420405758000E+09
    catalog_df = pd.read_csv(catalog_loc)
    catalog_df['dist'] = [(const.c*i/h_0).to(u.Mpc) for i in catalog_df.Z_VALUE]
    catalog_df["freq"] = [rest_freq/(i+1) for i in catalog_df.Z_VALUE]
    catalog_df["z_pos"] = np.nan
    catalog_df["pixels_x"] = np.nan
    catalog_df["pixels_y"] = np.nan
    catalog_df["file_name"] = np.nan
    source_props_df_full = pd.DataFrame(columns=['label', 'inertia_tensor_eigvals-0',
    'inertia_tensor_eigvals-1', 'inertia_tensor_eigvals-2', 'centroid-0', 'centroid-1',
    'centroid-2', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
    'flux', 'peak_flux', 'elongation', 'flatness', 'brightest_pix', 'max_loc',
    'file', 'true_positive_mocks', 'true_positive_real', 'n_channels', 'n_vel', 'nx',
    'dist', 'nx_mpc', 'ny', 'ny_mpc'])
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
        source_props_df = create_single_catalog(nonbinary_im, target_file, cube_file, catalog_df)
        source_props_df_full = source_props_df_full.append(source_props_df)
    print("saving file...")
    out_file = out_dir + "/" + scale + "_" + method + "_catalog.txt"
    source_props_df_full.to_csv(out_file)
    return


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
    parser.add_argument(
        '--catalog_loc', type=str, nargs='?', const='default', default="PP_redshifts_8x8.csv",
        help='The real catalog file')
    parser.add_argument(
        '--mask', type=bool, nargs='?', const='default', default=False,
        help='Whether to create mask catalog')
    args = parser.parse_args()

    main(args.data_dir, args.method, args.scale, args.output_dir, args.catalog_loc, args.mask)
