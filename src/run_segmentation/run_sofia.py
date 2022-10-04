import argparse
from os import listdir
import os
import pickle
from datetime import datetime


def main(sofia_loc, cube_dir, param_dir):
    """Runs SoFiA on a list of cubes to create detections

    Args:
        sofia_loc (str): The location of the SoFiA executable
        cube_dir (str): The directory of the input data
        param_dir (str): The location of the SoFiA parameter files
    Outputs:
        For each cube, a masked output of the detections found by SoFiA
        is outputted, with each source given a different label. In
        addition, the time taken to create the detections is outputted.
    """
    cubes = [i for i in listdir(cube_dir) if ".fits" in i]
    time_taken = {}
    for cube in cubes:
        before = datetime.now()
        print(cube)
        param_file = [param_dir + "/" + i for i in listdir(param_dir) if cube.split(".")[0] in i][0]
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
        '--cube_dir', type=str, nargs='?', const='default', default="./data/training/InputBoth",
        help='The directory of the cubes')
    parser.add_argument(
        '--param_dir', type=str, nargs='?', const='default', default="./run_segmentation/params",
        help='The directory containing the parameter files')
    args = parser.parse_args()

    main(args.sofia_loc, args.cube_dir, args.param_dir)

