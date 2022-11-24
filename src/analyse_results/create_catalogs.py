
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


def flux(row, orig_data, seg_output, rms):
    subcube = orig_data[int(row['bbox-0']):int(row['bbox-3']), int(row['bbox-1']):int(row['bbox-4']), int(row['bbox-2']):int(row['bbox-5'])]
    det_subcube = seg_output[int(row['bbox-0']):int(row['bbox-3']), int(row['bbox-1']):int(row['bbox-4']), int(row['bbox-2']):int(row['bbox-5'])]
    det_subcube[det_subcube != row.label] = 0
    det_subcube[det_subcube == row.label] = 1
    descaled = np.sum(subcube[det_subcube == row.label], axis=0)*rms[row['bbox-1']:row['bbox-4'], row['bbox-2']:row['bbox-5']]
    h_0 = 70*u.km/(u.Mpc*u.s)
    rest_freq = 1.4204058e9*u.Hz
    z = float(row.dist)*u.Mpc*h_0/const.c.to(u.km/u.s)
    freq = rest_freq/(1+z)
    dF = 36621.09375*u.Hz
    Iv = np.sum(descaled)*u.Jy*const.c.to(u.km/u.s)*dF/freq
    sigma_beam = (np.pi*5*25/(4*np.log(2)))*(1+z)**2
    tot_flux = (6**2)*Iv/sigma_beam
    peak_flux = np.max(descaled*u.Jy*const.c.to(u.km/u.s)*dF*(6**2)/(freq*sigma_beam))
    return tot_flux.value, peak_flux.value


def hi_mass(row, tot_flux):
    h_0 = 70*u.km/(u.Mpc*u.s)
    z = float(row.dist)*u.Mpc*h_0/const.c.to(u.km/u.s)
    mass = 2.35e5*tot_flux*(float(row.dist)*u.Mpc)**2/((1+z)**2)
    return mass.value


def brightest_pix(regionmask, intensity):
    return np.nanmax(intensity[regionmask])


def max_loc(regionmask, intensity):
    return np.where(intensity == np.nanmax(intensity[regionmask]))


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


def create_single_catalog(output_file, mock_mask_file, real_mask_file, real_file, mask=False):
    mos_name = real_file.split("/")[-1].split("_")[-1].split(".fits")[0]
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
    mock_mask_data = fits.getdata(mock_mask_file)
    real_mask_data = fits.getdata(real_mask_file)[:, 400:-400, 400:-400]
    orig_data = fits.getdata(real_file)
    seg_output = fits.getdata(output_file)
    rms = np.sqrt(np.nanmean(orig_data**2, axis=0))
    # noisefree_data = fits.getdata("data/training/Input/" + mask_data.split("/")[-1].replace("mask", "noisefree"))
    # Number mask
    print("numbering mask...")
    mock_mask_labels = skmeas.label(mock_mask_data)
    real_mask_labels = skmeas.label(real_mask_data)
    # Catalog mask
    print("cataloging mask...")
    mock_mask_df = pd.DataFrame(
        skmeas.regionprops_table(
        mock_mask_labels, orig_data, properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(brightest_pix, max_loc, elongation, n_vel, nx, ny))
    )
    real_mask_df = pd.DataFrame(
        skmeas.regionprops_table(
        real_mask_labels, orig_data, properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(brightest_pix, max_loc, elongation, n_vel, nx, ny))
    )
    mock_mask_df['mos_name'] = mos_name
    real_mask_df['mos_name'] = mos_name
    mock_mask_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in mock_mask_df.iterrows()]
    real_mask_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in real_mask_df.iterrows()]
    # if mask:
    #     mask_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in mask_df['centroid-0'].astype(int)]
    #     mask_df["nx_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.nx))*1e3
    #     mask_df["ny_kpc"] = mask_df.dist*np.tan(np.deg2rad(d_width*mask_df.ny))*1e3
    #     mask_df["hi_mass"] = np.nan
    #     # mask_df["asymmetry"] = np.nan
    #     for i, row in mask_df.iterrows():
    #         row.label = 1
    #         # Calculate HI mass
    #         tot_flux, peak_flux = flux(row, orig_data, seg_output, rms)
    #         mask_df.loc[i, "tot_flux"] = tot_flux
    #         mask_df.loc[i, "peak_flux"] = peak_flux
    #         mask_df.loc[i, "hi_mass"] = hi_mass(row, orig_data, mask_data)
    #         # Calculate asymmetry
    #         # mask_df.loc[i, "asymmetry"] = get_asymmetry(row, orig_data, mask_data)
    #     mask_df['mos_name'] = mos_name
    #     return mask_df
    # Catalog segmentation
    print("cataloging segmentation...")
    seg_output = fits.getdata(output_file)
    source_props_df = pd.DataFrame(
        skmeas.regionprops_table(seg_output, orig_data,
        properties=['label', 'centroid', 'bbox', 'area'],
        extra_properties=(brightest_pix, max_loc, elongation, detection_size, n_vel, nx, ny))
    )
    source_props_df['mos_name'] = mos_name
    source_props_df['max_loc'] = [[int(row['max_loc-0'] + row['bbox-0']), int(row['max_loc-1'] + row['bbox-1']), int(row['max_loc-2'] + row['bbox-2'])] for i, row in source_props_df.iterrows()]
    # Convert to physical values
    source_props_df["dist"] = [((const.c*((rest_freq/spec_cube[i])-1)/h_0).to(u.Mpc)).value for i in source_props_df['centroid-0'].astype(int)]
    source_props_df["nx_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.nx))*1e3
    source_props_df["ny_kpc"] = source_props_df.dist*np.tan(np.deg2rad(d_width*source_props_df.ny))*1e3
    source_props_df["file"] = output_file
    source_props_df["overlap_areas"] = np.nan
    source_props_df["area_gts"] = np.nan
    source_props_df["hi_mass"] = np.nan
    # source_props_df["asymmetry"] = np.nan
    source_props_df["true_positive_mocks"] = False
    source_props_df["type"] = np.nan
    for i, row in source_props_df.reset_index(drop=True).iterrows():
        mock_mask_row = mock_mask_df[(mock_mask_df.max_loc.astype(str) == str(row.max_loc)) & (mock_mask_df.mos_name == row.mos_name)]
        real_mask_row = real_mask_df[(real_mask_df.max_loc.astype(str) == str(row.max_loc)) & (real_mask_df.mos_name == row.mos_name)]
        if len(mock_mask_row) > 0:
            source_props_df.loc[i, "true_positive_mocks"] = True
            source_props_df.loc[i, "type"] = 'mock'
            source_props_df.loc[i, "area_gt"] = int(mock_mask_row.area)
            zp = [np.min([int(mock_mask_row['bbox-0']), int(row['bbox-0'])]), np.max([int(mock_mask_row['bbox-3']), int(row['bbox-3'])])]
            xp = [np.min([int(mock_mask_row['bbox-1']), int(row['bbox-1'])]), np.max([int(mock_mask_row['bbox-4']), int(row['bbox-4'])])]
            yp = [np.min([int(mock_mask_row['bbox-2']), int(row['bbox-2'])]), np.max([int(mock_mask_row['bbox-5']), int(row['bbox-5'])])]
            source_props_df.loc[i, "overlap_area"] = len(np.where((np.logical_and(seg_output[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]], mock_mask_labels[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]]).astype(int))>0)[0])
        elif len(real_mask_row) > 0:
            source_props_df.loc[i, "true_positive_mocks"] = True
            source_props_df.loc[i, "type"] = 'real'
            source_props_df.loc[i, "area_gt"] = int(real_mask_row.area)
            zp = [np.min([int(real_mask_row['bbox-0']), int(row['bbox-0'])]), np.max([int(real_mask_row['bbox-3']), int(row['bbox-3'])])]
            xp = [np.min([int(real_mask_row['bbox-1']), int(row['bbox-1'])]), np.max([int(real_mask_row['bbox-4']), int(row['bbox-4'])])]
            yp = [np.min([int(real_mask_row['bbox-2']), int(row['bbox-2'])]), np.max([int(real_mask_row['bbox-5']), int(row['bbox-5'])])]
            source_props_df.loc[i, "overlap_area"] = len(np.where((np.logical_and(seg_output[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]], real_mask_labels[zp[0]:zp[1], xp[0]:xp[1], yp[0]:yp[1]]).astype(int))>0)[0])
        # Calculate HI mass
        tot_flux, peak_flux = flux(row, orig_data, seg_output, rms)
        source_props_df.loc[i, "tot_flux"] = tot_flux
        source_props_df.loc[i, "peak_flux"] = peak_flux
        source_props_df.loc[i, "hi_mass"] = hi_mass(row, tot_flux)
        # Calculate asymmetry
        # source_props_df.loc[i, "asymmetry"] = get_asymmetry(row, orig_data, seg_output)
    return source_props_df


def main(data_dir, method, out_dir):
    """Creates a catalog of the source detections

    Args:
        data_dir (str): The location of the data
        method (str): The detection method used
        out_dir (str): The output directory for the results
    Outputs:
        A catalog of all the detected sources and their properties,
        including whether they match a ground truth source, for every
        cube is created.
    """
    cube_files = [data_dir + "training/InputBoth/" + i for i in listdir(data_dir+"training/InputBoth") if (".fits" in i)]
    source_props_df_full = pd.DataFrame()
    for cube_file in cube_files:
        mos_name = cube_file.split("/")[-1].split("_")[-1].split(".fits")[0]
        print(mos_name)
        target_file_mock = data_dir + "training/Target/mask_" + mos_name+  ".fits"
        target_file_real = data_dir + "training/TargetReal/mask_" + mos_name+  ".fits"
        mask = True if method == "MASK" else False
        if method == "MTO":
            nonbinary_im = data_dir + "mto_output/mtocubeout_" + mos_name+  ".fits"
        elif method == "VNET":
            nonbinary_im = data_dir + "vnet_output/vnet_cubeout_" + mos_name+  ".fits"
        elif method == "SOFIA":
            nonbinary_im = data_dir + "sofia_output/sofia_" + mos_name+  "_mask.fits"
        elif method == "MASK":
            nonbinary_im = data_dir + "training/target_" + mos_name+  "_mask.fits"
        source_props_df = create_single_catalog(output_file=nonbinary_im, mock_mask_file=target_file_mock, real_mask_file=target_file_real, real_file=cube_file, mask=mask)
        source_props_df_full = source_props_df_full.append(source_props_df)
    print("saving file...")
    out_file = out_dir + "/" + method + "_catalog.txt"
    source_props_df_full.to_csv(out_file, index=False)
    return


def connect_detections_to_ground_truth(detection_catalogue, gt_catalgue):
    mto_and_gt_df = pd.DataFrame()
    k = 0
    for mos_name in mos_names:
        overlap_df = pd.merge(mto_cat_df[mto_cat_df.type=='mock'], mock_cat_df, on=["mos_name", "max_loc", "type"], suffixes=("_mto", "_mask"), how='outer', indicator=True)
        left_only = overlap_df[overlap_df._merge == 'left_only']
        right_only = overlap_df[overlap_df._merge == 'right_only']
        both = overlap_df[overlap_df._merge == 'both'][overlap_df.columns[:-1]]
        nearest_overlap_df = pd.DataFrame()
        for i, row in right_only.iterrows():
            max_loc_z, max_loc_x, max_loc_y = [int(m) for m in row.max_loc.replace("]", "").replace("[", "").split(", ")]
            max_locs = left_only.max_loc.str.replace("]", "").str.replace("[", "").str.split(", ", expand=True).astype(int)
            max_locs.columns = ["max_loc_z", "max_loc_x", "max_loc_y"]
            distances = np.sqrt((max_locs.max_loc_z-max_loc_z)**2 + (max_locs.max_loc_x-max_loc_x)**2 + (max_locs.max_loc_y-max_loc_y)**2)
            mapped_row = left_only[distances == distances.min()]
            if mapped_row.area_gt.values[0] == row.area_mask:
                joined_df = mapped_row[mapped_row.columns[:36]].reset_index(drop=True).join(pd.DataFrame([row[right_only.columns[36:-1]]]).reset_index(drop=True))
                joined_df["overlap"] = distances.min()
                nearest_overlap_df = nearest_overlap_df.append(joined_df)
        # Take closest if 2
    #     nearest_overlap_df.loc[nearest_overlap_df.groupby('label')['overlap'].transform('min').eq(nearest_overlap_df['overlap'])].reset_index(drop=True)
        nearest_overlap_df = nearest_overlap_df[nearest_overlap_df.columns[:-1]]
        all_mapped_df = nearest_overlap_df.append(both).reset_index(drop=True)
        all_mapped_df["mos_name"] == mos_name
        mto_and_gt_df = mock_galaxy_gt_df.append(all_mapped_df)
        print(mos_name)
        if (len(all_mapped_df[all_mapped_df.label_mto.isnull()]), len(all_mapped_df[all_mapped_df.new_mass.isnull()])) != (0, 0):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create catalog from output",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data_dir', type=str, nargs='?', const='default', default="data/",
        help='The directory containing the data')
    parser.add_argument(
        '--method', type=str, nargs='?', const='default', default='SOFIA',
        help='The segmentation method being evaluated')
    parser.add_argument(
        '--output_dir', type=str, nargs='?', const='default', default="results/",
        help='The output directory for the results')
    args = parser.parse_args()

    main(args.data_dir, args.method, args.output_dir)

