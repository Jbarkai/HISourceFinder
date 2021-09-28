
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
from photutils.centroids import centroid_com


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

def hi_mass(row, orig_data, seg_output):
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.001666666707*u.deg
    h_0 = 70*u.km/(u.Mpc*u.s)
    dF = 36621.09375*u.Hz
    dist = (row.dist*u.Mpc)
    scale = dist*np.tan(np.deg2rad(d_width))
    deltaV = (dF*const.c/rest_freq).to(u.km/u.s)
    redshift = h_0*dist/const.c
    subcube = orig_data[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
    det_subcube = seg_output[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
    moment_0 = np.sum(subcube*det_subcube, axis=0)
    S_v = moment_0*deltaV
    mass = np.sum(2.36e5*S_v*dist**2)/(1+redshift)
    return mass.value

def get_asymmetry(row, orig_data, seg_output):
    subcube = orig_data[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
    det_subcube = seg_output[row['bbox-0']:row['bbox-3'], row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
    moment_0 = np.sum(subcube*det_subcube, axis=0)
    y, x = (np.where(moment_0 == np.max(moment_0)))
    if (len(x) == 1) & (len(y) == 1):
        mpx, mpy = float((centroid_com(moment_0)[0] + x)/2), float((centroid_com(moment_0)[1] + y)/2)
        dx = mpx - (moment_0.shape[1]/2)
        dy = mpy - (moment_0.shape[0]/2)
        z1, z2 = row['bbox-0'], row['bbox-3']
        if (dx > 0) & (dy < 0):
            x1, x2 = int(row['bbox-1'] + dy), int(row['bbox-4'])
            y1, y2 = row['bbox-2'], int(row['bbox-2'] + 2*mpx)
        if (dx > 0) & (dy > 0):
            x1, x2 = int(row['bbox-1']), int(row['bbox-1'] + 2*mpy)
            y1, y2 = row['bbox-2'], int(row['bbox-2'] + 2*mpx)
        if (dx < 0) & (dy < 0):
            x1, x2 = int(row['bbox-1'] + dy), int(row['bbox-4'])
            y1, y2 = int(row['bbox-2']+dx), row['bbox-5']
        if (dx < 0) & (dy > 0):
            x1, x2 = int(row['bbox-1']), int(row['bbox-1'] + 2*mpy)
            y1, y2 = int(row['bbox-2']+dx), row['bbox-5']
        new_cube = subcube[z1:z2, x1:x2, y1:y2]*det_subcube[z1:z2, x1:x2, y1:y2]
        new_cube[det_subcube[z1:z2, x1:x2, y1:y2] != int(row.label)] = 0
        new_cube[new_cube != 0] = 1
        moment_0_pad = np.sum(new_cube, axis=0)
        rotated = np.rot90(moment_0_pad, 2)
        subtracted = np.abs(moment_0_pad - rotated)
        asymmetry = np.sum(subtracted)/(np.sum(moment_0_pad)+ np.sum(rotated))
    else:
        asymmetry = np.nan
    return asymmetry

def create_single_catalog(output_file, mask_file, real_file, cols, mask=False):
    rest_freq = 1.420405758000E+09*u.Hz
    d_width = 0.001666666707*u.deg
    h_0 = 70*u.km/(u.Mpc*u.s)
    # Load reference cube
    hi_data = fits.open(real_file)
    hi_data[0].header['CTYPE3'] = 'FREQ'
    hi_data[0].header['CUNIT3'] = 'Hz'
    spec_cube = (SpectralCube.read(hi_data)).spectral_axis
    hi_data.close()
    # Load segmentation, real and mask cubes
    mask_data = fits.getdata(mask_file)
    orig_data = fits.getdata(real_file)
    # noisefree_data = fits.getdata("data/training/Input/" + mask_data.split("/")[-1].replace("mask", "noisefree"))
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
    mask_df["file"] = mask_file
    mask_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in mask_df.iterrows()]
    if mask:
        mask_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in mask_df['centroid-0'].astype(int)]
        mask_df["nx_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.nx))*1e3
        mask_df["ny_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.ny))*1e3
        mask_df["hi_mass"] = np.nan
        mask_df["asymmetry"] = np.nan
        for i, row in mask_df.iterrows():
            row.label = 1
            # Calculate HI mass
            mask_df.loc[i, "hi_mass"] = hi_mass(row, orig_data, mask_data)
            # Calculate asymmetry
            mask_df.loc[i, "asymmetry"] = get_asymmetry(row, orig_data, mask_data)
        return mask_df[cols]
    # Catalog segmentation
    print("cataloging segmentation...")
    seg_output = fits.getdata(output_file)
    source_props_df = pd.DataFrame(
        skmeas.regionprops_table(seg_output, orig_data,
        properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(tot_flux, peak_flux, brightest_pix, max_loc, elongation, detection_size, n_vel, nx, ny))
    )
    source_props_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in source_props_df.iterrows()]
    # Convert to physical values
    source_props_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in source_props_df['centroid-0'].astype(int)]
    source_props_df["nx_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.nx))*1e3
    source_props_df["ny_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.ny))*1e3
    source_props_df["file"] = output_file
    source_props_df['true_positive_mocks'] = [i in list(mask_df.max_loc.values) for i in source_props_df.max_loc]
    overlap_areas = []
    area_gts = []
    source_props_df["hi_mass"] = np.nan
    source_props_df["asymmetry"] = np.nan
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
        # Calculate HI mass
        source_props_df.loc[i, "hi_mass"] = hi_mass(row, orig_data, seg_output)
        # Calculate asymmetry
        source_props_df.loc[i, "asymmetry"] = get_asymmetry(row, orig_data, seg_output)
    source_props_df['overlap_area'] = overlap_areas
    source_props_df['area_gt'] = area_gts
    return source_props_df[cols]


def main(data_dir, method, scale, out_dir):
    """Creates a catalog of the source detections

    Args:
        data_dir (str): The location of the data
        method (str): The detection method used
        scale (str): The scale of the inserted galaxies
        out_dir (str): The output directory for the results
    Outputs:
        A catalog of all the detected sources and their properties,
        including whether they match a ground truth source, for every
        cube is created.
    """
    cube_files = [data_dir + "training/" +scale+"Input/" + i for i in listdir(data_dir+"training/"+scale+"Input") if ".fits" in i]
    for cube_file in cube_files:
        mos_name = cube_file.split("/")[-1].split("_")[-1].split(".fits")[0]
        print(mos_name)
        target_file = data_dir + "training/Target/mask_" + cube_file.split("/")[-1].split("_")[-1]
        if method == "MASK":
            cols = ['label', 'centroid-0', 'centroid-1', 'centroid-2',
            'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
            'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
            'dist', 'nx', 'ny', 'n_vel', 'nx_kpc', 'ny_kpc', 'file', 'hi_mass',
            'asymmetry', 'mos_name']
            source_props_df_full = pd.DataFrame(columns=cols)
            source_props_df = create_single_catalog("", target_file, cube_file, source_props_df_full.columns, mask=True)
        else:
            cols = ['label', 'centroid-0', 'centroid-1', 'centroid-2',
            'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5', 'area',
            'tot_flux', 'peak_flux', 'brightest_pix', 'max_loc', 'elongation',
            'detection_size', 'dist', 'nx', 'ny', 'n_vel', 'nx_kpc', 'ny_kpc', 'file',
            'true_positive_mocks', 'hi_mass', 'asymmetry', 'overlap_area', 'area_gt', 'mos_name']
            source_props_df_full = pd.DataFrame(columns=cols)
            if method == "MTO":
                nonbinary_im = data_dir + "mto_output/mtocubeout_" + scale + "_" + mos_name+  ".fits"
            elif method == "VNET":
                nonbinary_im = data_dir + "vnet_output/vnet_cubeout_" + scale + "_" + mos_name+  ".fits"
            elif method == "SOFIA":
                nonbinary_im = data_dir + "sofia_output/sofia_" + scale + "_" + mos_name+  "_mask.fits"
            elif method == "MASK":
                nonbinary_im = data_dir + "training/target_" + scale + "_" + mos_name+  "_mask.fits"
            source_props_df = create_single_catalog(nonbinary_im, target_file, cube_file, source_props_df_full.columns)
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
    args = parser.parse_args()

    main(args.data_dir, args.method, args.scale, args.output_dir)
