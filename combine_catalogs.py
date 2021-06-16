
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

def comb_cats(df1, df2, cols_to_keep, suffixes):
    result = pd.merge(
            df1[cols_to_keep], df2[cols_to_keep],
            on=["mos_name", "max_loc"], suffixes=suffixes, how='outer'
        )
    comb_df = pd.DataFrame(np.nan, index=result.index, columns=cols_to_keep)
    for col in cols_to_keep:
        if col in ['max_loc', 'mos_name']:
            comb_df[col] = result[col]
            continue
        if col == 'file':
            continue
        cond_mto = ~result[col+suffixes[1]].isnull() & result[col+suffixes[0]].isnull()
        cond_sofia = ~result[col+suffixes[0]].isnull() & result[col+suffixes[1]].isnull()
        cond_both_sofia = (result.area_sofia >= result.area_mto) & ~result[col+suffixes[0]].isnull() & ~result[col+suffixes[1]].isnull()
        cond_both_mto = ~(result.area_sofia >= result.area_mto) & ~result[col+suffixes[0]].isnull() & ~result[col+suffixes[1]].isnull()
        comb_df.loc[cond_mto, col] = result[cond_mto][col+suffixes[1]]
        comb_df.loc[cond_mto, "file"] = result[cond_mto].file_mto
        comb_df.loc[cond_sofia, col] = result[cond_sofia][col+suffixes[0]]
        comb_df.loc[cond_sofia, "file"] = result[cond_sofia].file_sofia
        comb_df.loc[cond_both_mto, col] = result[cond_both_mto][col+suffixes[1]]
        comb_df.loc[cond_both_mto, "file"] = result[cond_both_mto].file_mto
        comb_df.loc[cond_both_sofia, col] = result[cond_both_sofia][col+suffixes[0]]
        comb_df.loc[cond_both_sofia, "file"] = result[cond_both_sofia].file_sofia
    comb_df['mos_name'] = comb_df.file.str.split("_", expand=True)[3].str.replace(".fits", "")
    return comb_df


def add_to_mask(file_name, all_df):
    print(file_name)
    mask_file = "../data/training/Target/mask_" + file_name + ".fits"
    gt_cube = fits.getdata(mask_file)
    subset = all_df[(~all_df.true_positive_mocks) & all_df.true_positive_real & (all_df.mos_name == file_name)]
    for  i, row in subset.iterrows():
        new_mask = fits.getdata("../" + row.file)[
            int(row['bbox-0']):int(row['bbox-3']),
            int(row['bbox-1']):int(row['bbox-4']),
            int(row['bbox-2']):int(row['bbox-5'])
        ]
        new_mask[new_mask > 0] = 1
        gt_cube[
            int(row['bbox-0']):int(row['bbox-3']),
            int(row['bbox-1']):int(row['bbox-4']),
            int(row['bbox-2']):int(row['bbox-5'])
        ] = new_mask
    new_file = "../data/training/TargetCat/mask_" + file_name + ".fits"
    fits.write_to(new_file, gt_cube)


def main(scale, output_dir):
    # Load catalogs
    mto_cat_df = pd.read_csv(output_dir+ scale + "_MTO_catalog.txt", index_col=0)
    # vnet_cat_df = pd.read_csv(output_dir+ scale + "_VNET_catalog.txt", index_col=0)
    sofia_cat_df = pd.read_csv(output_dir+ scale + "_SOFIA_catalog.txt", index_col=0)
    sofia_cat_df["mos_name"] = sofia_cat_df.file.str.split("_", expand=True)[3]
    mto_cat_df["mos_name"] = mto_cat_df.file.str.replace(".fits", "").str.split("_", expand=True)[3]
    # vnet_cat_df["mos_name"] = vnet_cat_df.file.str.replace(".fits", "").str.split("_", expand=True)[3]
    cols_to_keep = ['mos_name', 'max_loc', 'area', 'peak_flux', 'eccentricity', 'flatness', 'brightest_pix',
         'n_channels', 'nx', 'ny', 'tot_flux', 'true_positive_mocks', 'true_positive_real',
        'bbox-0', 'bbox-3', 'bbox-1', 'bbox-4', 'bbox-2', 'bbox-5', 'file']
    # Combine them taking the masks with the largest area
    all_df = comb_cats(sofia_cat_df, mto_cat_df, cols_to_keep, suffixes=("_sofia", "_mto"))
    # all_df = comb_cats(all_df, vnet_cat_df, cols_to_keep, suffixes=("_sofiamto", "_vnet"))
    # Save final catalog
    out_file = output_dir + scale + "_all_catalog.txt"
    all_df.to_csv(out_file)
    # Insert false positives that were cross-referenced and found to be real sources
    for file_name in all_df.mos_name.unique():
        add_to_mask(file_name, all_df)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine catalogs and add to masks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--scale', type=str, nargs='?', const='default', default='loud',
        help='The scale of the inserted galaxies')
    parser.add_argument(
        '--output_dir', type=str, nargs='?', const='default', default="results/",
        help='The output directory for the results')
    args = parser.parse_args()

    main(args.scale, args.output_dir)
