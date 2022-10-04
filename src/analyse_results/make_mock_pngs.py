from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

overlap_mocks = pd.read_csv("mock_overlaps.csv")

def get_size(row, data):
    z1 = np.min([row['bbox-0'], row['bbox-0_vnet'], row['bbox-0_sofia']])
    z2 = np.max([row['bbox-3'], row['bbox-3_vnet'], row['bbox-3_sofia']])
    x1 = np.min([row['bbox-1'], row['bbox-1_vnet'], row['bbox-1_sofia']])
    x2 = np.max([row['bbox-4'], row['bbox-4_vnet'], row['bbox-4_sofia']])
    y1 = np.min([row['bbox-2'], row['bbox-2_vnet'], row['bbox-2_sofia']])
    y2 = np.max([row['bbox-5'], row['bbox-5_vnet'], row['bbox-5_sofia']])
    return data[z1:z2, x1:x2, y1:y2]

for mos_name in np.sort(overlap_mocks.mos_name.unique()):
    real_cube = fits.getdata("./data/training/loudInput/loud_%s.fits"%mos_name)
    subset = overlap_mocks[overlap_mocks.mos_name == mos_name]
    for i, row in subset.iterrows():
        if (((row.mos_name == "1245mosB") & (row.label == 32)) |
               ((row.mos_name == "1245mosC") & (row.label == 516)) |
               ((row.mos_name == "1245mosD") & (row.label == 385)) |
               ((row.mos_name == "1245mosE") & (row.label == 402)) |
               ((row.mos_name == "1245mosF") & (row.label == 281)) |
               ((row.mos_name == "1245mosG") & (row.label == 395)) |
               ((row.mos_name == "1245mosH") & (row.label == 1026))):
            gt_subcube = get_size(row, fits.getdata("./data/training/Target/mask_%s.fits"%mos_name))
            gt_moment_0 = np.sum(get_size(row, real_cube)*gt_subcube, axis=0)
            fig4, ax4 = plt.subplots(1, 1, figsize=(5, 5))
            ax4.contour(gt_moment_0, origin='lower')
            ax4.axis('off')
            plt.savefig("mock_examples/GT_" + row.mos_name + "_" + str(row.label) + ".png")
            plt.close('all')
        # mto_subcube[mto_subcube != row.label] = 0
        # mto_subcube[mto_subcube == row.label] = 1
        # vnet_subcube = get_size(row, fits.getdata(row.file_vnet))
        # vnet_subcube[vnet_subcube != row.label_vnet] = 0
        # vnet_subcube[vnet_subcube == row.label_vnet] = 1
        # sofia_subcube = get_size(row, fits.getdata(row.file_sofia))
        # sofia_subcube[sofia_subcube != row.label_sofia] = 0
        # sofia_subcube[sofia_subcube == row.label_sofia] = 1
        # mto_moment_0 = np.sum(get_size(row, real_cube)*mto_subcube, axis=0)
        # vnet_moment_0 = np.sum(get_size(row, real_cube)*vnet_subcube, axis=0)
        # sofia_moment_0 = np.sum(get_size(row, real_cube)*sofia_subcube, axis=0)
        # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        # ax1.contour(mto_moment_0, origin='lower')
        # ax1.axis('off')
        # plt.savefig("mock_examples/MTO_" + row.mos_name + "_" + str(row.label) + ".png")
        # fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        # ax2.contour(vnet_moment_0, origin='lower')
        # ax2.axis('off')
        # plt.savefig("mock_examples/VNET_" + row.mos_name + "_" + str(row.label_vnet) + ".png")
        # fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
        # ax3.contour(sofia_moment_0, origin='lower')
        # ax3.axis('off')
        # plt.savefig("mock_examples/SOFIA_" + row.mos_name + "_" + str(row.label_sofia) + ".png")
        # plt.close('all')
