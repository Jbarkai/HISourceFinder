import argparse
from os import listdir
import os
import pickle
from astropy.io import fits
import numpy as np
from scipy import ndimage as ndi
from datetime import datetime


def main(sofia_loc, cube_dir, param_dir):
    cubes = listdir(cube_dir)
    time_taken = {}
    for cube in cubes:
        before = datetime.now()
        print(cube)
        param_file = [i for i in listdir(param_dir) if cube in i][0]
        success = os.system('%s %s >> data/sofia_output/sofia_output.log'%(sofia_loc, param_file))
        print(success)
        after = datetime.now()
        difference = (after - before).total_seconds()
        time_taken.update({cube: difference})
    out_file = "sofia_performance_" + cube_dir.split("/")[-1] + ".txt"
    with open(out_file, "wb") as fp:
        pickle.dump(time_taken, fp)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SoFiA",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--sofia_loc', type=str, nargs='?', const='default', default="/net/blaauw/data2/users/vdhulst/SoFiA-2/sofia",
        help='The sofia executable location')
    parser.add_argument(
        '--cube_dir', type=str, nargs='?', const='default', default="./data/training/loudInput",
        help='The directory of the cubes')
    parser.add_argument(
        '--param_dir', type=str, nargs='?', const='default', default="./run_segmentation/params",
        help='The directory containing the parameter files')
    args = parser.parse_args()

    main(args.sofia_loc, args.cube_dir, args.param_dir)

