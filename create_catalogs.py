
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

def brightest_pix(regionmask, intensity):
    return np.nanmax(intensity[regionmask])

def max_loc(regionmask, intensity):
    return np.where(intensity == np.nanmax(intensity[regionmask]))

def tot_flux(regionmask, intensity):
    return np.nansum(intensity[regionmask])

def elongation(regionmask, intensity):
    eg0, eg1 = skmeas.inertia_tensor_eigvals(regionmask[int(regionmask.shape[0]/2)]) 
    return eg0/eg1

def detection_size(regionmask, intensity):
    return np.prod(intensity.shape)

def n_vel(regionmask, intensity):
    d_channels = 36621.09375*u.Hz
    rest_freq = 1.420405758000E+09*u.Hz
    return (regionmask.shape[0]*d_channels*const.c/rest_freq).to(u.km/u.s).value

def nx(regionmask, intensity):
    return regionmask.shape[1]

def ny(regionmask, intensity):
    return regionmask.shape[2]


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
        mask_labels, orig_data, properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux, brightest_pix, max_loc, elongation, n_vel, nx, ny))
    )
    mask_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in mask_df.iterrows()]
    # Convert to physical values
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.065000001573*u.deg
    h_0 = 70*u.km/(u.Mpc*u.s)
    # Load reference cube
    hi_data = fits.open(real_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = (SpectralCube.read(hi_data)).spectral_axis
    hi_data.close()
    mask_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in mask_df['centroid-0'].astype(int)]
    mask_df["nx_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.nx))*1e3
    mask_df["ny_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.ny))*1e3
    mask_df["file"] = mask_file
    print(len(mask_df))
    return mask_df[['label', 'centroid-0', 'centroid-1', 'centroid-2',
    'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
    'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
    'dist', 'nx', 'ny', 'n_vel', 'nx_kpc', 'ny_kpc', 'file']]


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
        mask_labels, orig_data, properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux, brightest_pix, max_loc, elongation))
    )
    mask_df["file"] = mask_file
    mask_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in mask_df.iterrows()]
    # Catalog segmentation
    print("cataloging segmentation...")
    source_props_df = pd.DataFrame(
        skmeas.regionprops_table(seg_output, orig_data,
        properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux, brightest_pix, max_loc, elongation, detection_size, n_vel, nx, ny))
    )
    source_props_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in source_props_df.iterrows()]
    # Load reference cube
    hi_data = fits.open(real_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = (SpectralCube.read(hi_data)).spectral_axis
    hi_data.close()
    # Convert to physical values
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.001666666707*u.deg
    h_0 = 70*u.km/(u.Mpc*u.s)
    source_props_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in source_props_df['centroid-0'].astype(int)]
    source_props_df["nx_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.nx))*1e3
    source_props_df["ny_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.ny))*1e3

    source_props_df["file"] = output_file
    source_props_df['true_positive_mocks'] = [i in list(mask_df.max_loc.values) for i in source_props_df.max_loc]
    overlap_areas = []
    area_gts = []
    source_props_df["hi_mass"] = np.nan
    for i, row in source_props_df.iterrows():
        mask_row = mask_df[mask_df.max_loc.astype(str) == str([int(i) for i in row.max_loc])]
        if len(mask_row) > 0:
            area_gts.append(int(mask_row.area))
            zp = [np.min([int(mask_row['bbox-0']), int(row['bbox-0'])]), np.max([int(mask_row['bbox-3']), int(row['bbox-3'])])]
            xp = [np.min([int(mask_row['bbox-1']), int(row['bbox-1'])]), np.max([int(mask_row['bbox-4']), int(row['bbox-4'])])]
            yp = [np.min([int(mask_row['bbox-2']), int(row['bbox-2'])]), np.max([int(mask_row['bbox-5']), int(row['bbox-5'])])]
            overlap_areas.append(len(np.where((np.logical_and(seg_output[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]], mask_labels[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]]).astype(int))>0)[0]))
        else:
            overlap_areas.append(np.nan)
            area_gts.append(np.nan)
        subcube = orig_data[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
        det_subcube = seg_output[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
        dF = 36621.09375*u.Hz
        dist = (row.dist*u.Mpc)
        scale = dist*np.tan(np.deg2rad(d_width))
        deltaV = (dF*const.c/rest_freq).to(u.km/u.s)
        redshift = h_0*dist/const.c
        S_v = np.sum(subcube*det_subcube, axis=0)
        mass = np.sum(2.36e5*S_v*dist**2)/(1+redshift)
        source_props_df.loc[i, "hi_mass"] = mass.value
    source_props_df['overlap_area'] = overlap_areas
    source_props_df['area_gt'] = area_gts
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
    print(len(source_props_df))
    return source_props_df[['label', 'centroid-0', 'centroid-1', 'centroid-2',
    'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
    'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
    'detection_size', 'dist', 'nx', 'ny', 'n_vel', 'nx_kpc', 'ny_kpc', 'file', 'true_positive_mocks',
    'overlap_area', 'area_gt', 'true_positive_real']]


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
        source_props_df_full = pd.DataFrame(columns=['label', 'centroid-0', 'centroid-1', 'centroid-2',
        'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
        'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
        'dist', 'nx', 'ny', 'n_vel', 'nx_kpc', 'ny_kpc', 'file'])
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
    source_props_df_full = pd.DataFrame(columns=['label', 'centroid-0', 'centroid-1', 'centroid-2',
    'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
    'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
    'detection_size', 'n_vel', 'nx_kpc', 'ny_kpc', 'file', 'true_positive_mocks', 'hi_mass',
    'overlap_area', 'area_gt', 'true_positive_real'])
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
        source_props_df['mos_name'] = mos_name
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
