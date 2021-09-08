import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
import sys
sys.path.insert(0,'..')
import skimage.measure as skmeas
from datetime import datetime
import pandas as pd


def insert_gals(row, sofia_data, mask_cube, mto_data=None, comb=False):
    if comb:
        z1 = np.min([int(row['bbox-0_sof']), int(row['bbox-0'])])
        z2 = np.max([int(row['bbox-3_sof']), int(row['bbox-3'])])
        x1 = np.min([int(row['bbox-1_sof']), int(row['bbox-1'])])
        x2 = np.max([int(row['bbox-4_sof']), int(row['bbox-4'])])
        y1 = np.min([int(row['bbox-2_sof']), int(row['bbox-2'])])
        y2 = np.max([int(row['bbox-5_sof']), int(row['bbox-5'])])
        sof_det = sofia_data[z1:z2, x1:x2, y1:y2].copy()
        mto_det = mto_data[z1:z2, x1:x2, y1:y2].copy()
        sof_det[sof_det == int(row.label_sof)] = 1
        sof_det[sof_det != 1] = 0
        mto_det[mto_det == int(row.label)] = 1
        mto_det[mto_det != 1] = 0
        mask_cube[z1:z2, x1:x2, y1:y2] = np.max([mask_cube[z1:z2, x1:x2, y1:y2], (sof_det | mto_det)], axis=0)
    else:
        z1, z2, x1, x2, y1, y2 = int(row['bbox-0']), int(row['bbox-3']), int(row['bbox-1']), int(row['bbox-4']), int(row['bbox-2']), int(row['bbox-5'])
        mask_cube[z1:z2, x1:x2, y1:y2] = np.max([mask_cube[z1:z2, x1:x2, y1:y2], sofia_data[z1:z2, x1:x2, y1:y2]], axis=0)
    return


def main(args, data_dir="./data", scale="loud"):
    mto_cat_df = pd.read_csv("./results/loud_MTO_catalog.txt", index_col=0)
    vnet_cat_df = pd.read_csv("./results/loud_VNET_catalog.txt", index_col=0)
    sofia_cat_df = pd.read_csv("./results/loud_SOFIA_catalog.txt", index_col=0)
    mask_cat_df = pd.read_csv("./results/loud_MASK_catalog.txt", index_col=0)
    mask_cat_df["mos_name"] = mask_cat_df.file.str.split("_", expand=True)[1].str.replace(".fits", "")
    sofia_cat_df["mos_name"] = sofia_cat_df.file.str.split("_", expand=True)[3]
    mto_cat_df["mos_name"] = mto_cat_df.file.str.split("_", expand=True)[3].str.replace(".fits", "")
    vnet_cat_df["mos_name"] = vnet_cat_df.file.str.split("_", expand=True)[4].str.replace(".fits", "")
    # Label real sources
    real_catalog = pd.read_csv("./fp_to_drop.csv")
    new_sof = pd.merge(sofia_cat_df, real_catalog[real_catalog.method=="sofia"], on=["mos_name", "label"], how="left")
    new_vnet = pd.merge(vnet_cat_df, real_catalog[real_catalog.method=="vnet"], on=["mos_name", "label"], how="left")
    new_mto = pd.merge(mto_cat_df, real_catalog[real_catalog.method=="mto"], on=["mos_name", "label"], how="left")
    # Take only real ones
    real_gals = pd.merge(new_mto[~new_mto.type.isnull() & (new_mto.type!="notsure") & (new_mto.mos_name.str.contains("1245"))], pd.merge(new_sof[~new_sof.type.isnull() & (new_sof.type!="notsure") & (new_sof.mos_name.str.contains("1245"))], new_vnet[~new_vnet.type.isnull() & (new_vnet.type!="notsure") & (new_vnet.mos_name.str.contains("1245"))], on=['mos_name', 'max_loc'], how='outer', suffixes=("_sof", "_vnet")), on=['mos_name', 'max_loc'], how='outer', suffixes=("_mto", ""))
    # for methods detected by only vnet and another method, take the other method
    # for methods only detected by a single method, take that mask
    sof_sources = real_gals[(~real_gals.type_sof.isnull() & real_gals.type.isnull())]
    sof_sources = sof_sources[list(sof_sources.columns[sof_sources.columns.str.contains("_sof")])+ ["max_loc", "mos_name"]]
    sof_sources.columns = sof_sources.columns.str.replace("_sof", "")
    vnet_sources = real_gals[(real_gals.type_sof.isnull() & ~real_gals.type_vnet.isnull() & real_gals.type.isnull())]
    vnet_sources = vnet_sources[list(vnet_sources.columns[vnet_sources.columns.str.contains("_vnet")])+ ["max_loc", "mos_name"]]
    vnet_sources.columns = vnet_sources.columns.str.replace("_vnet", "")
    mto_sources = real_gals[(real_gals.type_sof.isnull() & ~real_gals.type.isnull())]
    mto_sources = mto_sources[list(mto_sources.columns[~mto_sources.columns.str.contains("_sof") & ~mto_sources.columns.str.contains("_vnet")])+ ["max_loc", "mos_name"]]
    mto_sof_sources = real_gals[~real_gals.type_sof.isnull() & ~real_gals.type.isnull()]
    # merge all the sources together
    # Add SoFiA sources:
    for mos_name in new_sof.mos_name.unique():
        arr_shape = (652, 1800, 2400)
        mask_cube = np.zeros(arr_shape)
        # mask_cube = fits.getdata("./data/training/Target/mask_%s.fits"%mos_name)
        mto_data = fits.getdata(data_dir + "mto_output/mtocubeout_" + scale + "_" + row.mos_name+  ".fits")
        vnet_data = fits.getdata(data_dir + "vnet_output/vnet_cubeout_" + scale + "_" + row.mos_name+  ".fits")
        sofia_data = fits.getdata(data_dir + "sofia_output/sofia_" + scale + "_" + row.mos_name+  "_mask.fits")
        for i, row in sof_sources[sof_sources.mos_name == "mos_name"].iterrows():
            insert_gals(row, sofia_data, mask_cube)
        for i, row in mto_sources[mto_sources.mos_name == "mos_name"].iterrows():
            insert_gals(row, mto_data, mask_cube)    
        for i, row in vnet_sources[vnet_sources.mos_name == "mos_name"].iterrows():
            insert_gals(row, vnet_data, mask_cube)  
        for i, row in mto_sof_sources[mto_sof_sources.mos_name == "mos_name"].iterrows():
            insert_gals(row, sofia_data, mask_cube, mto_data, comb=True)
        fits.writeto("data/training/TargetReal/mask_%s.fits"%mos_name, mask_cube)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add real masks to GT",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cuda', type=bool, nargs='?', const='default', default=True,
        help='Memory allocation')
    args = parser.parse_args()
    main(args)

