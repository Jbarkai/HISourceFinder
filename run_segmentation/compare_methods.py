import argparse
from os import listdir
import skimage.measure as skmeas
from scipy import stats
from astropy.io import fits
import pickle
import numpy as np


class Evaluator:
    """
    Code was adapted and mofified from https://gitlab.com/michaelvandeweerd/mto-lvq
    A class to hold the ground truth segment properties
    """

    def __init__(self, img, gt, mos_name, opt_method=0):
        self.method = opt_method
        self.original_img = img
        self.mos_name = mos_name
        img_shape = self.original_img.shape

        self.target_map = gt.ravel()
        
        # Sort the target ID map for faster pixel retrieval
        sorted_ids = self.target_map.argsort()
        id_set = np.unique(self.target_map)
        id_set.sort()

        # Get the locations in sorted_ids of the matching pixels
        right_indices = np.searchsorted(self.target_map, id_set, side='right', sorter=sorted_ids)
        left_indices = np.searchsorted(self.target_map, id_set, side='left', sorter=sorted_ids)

        # Create an id-max_index dictionary
        self.id_to_max = {}

        # Create an id - area dictionary (for merging error comparisons)
        self.id_to_area = {}

        # Iterate over object IDs
        for n in range(len(id_set)):
            # Find the location of the brightest pixel in each object
            pixel_indices = np.unravel_index(sorted_ids[left_indices[n]:right_indices[n]], img_shape)

            m = np.argmax(self.original_img[pixel_indices])
            max_pixel_index = (pixel_indices[0][m], pixel_indices[1][m], pixel_indices[2][m])

            # Save the location and area in dictionaries
            self.id_to_max[id_set[n]] = max_pixel_index
            self.id_to_area[id_set[n]] = right_indices[n] - left_indices[n]

    def match_to_bp_list(self, detection_map):
        """Match at most one detection to each target object"""

        # Reverse to ensure 1:1 mapping
        det_to_target = {}
        target_ids = list(self.id_to_max.keys())

        # Find the id of the background detection (0 or -1)
        det_min = detection_map.min()
        det_to_target[det_min] = -1
        target_ids.remove(det_min)

        # Map each id to the detection containing it
        for t_id in target_ids:
            max_loc = self.id_to_max[t_id]
            d_id = detection_map[max_loc]

            if d_id == det_min:
                continue

            # Assign detections covering multiple maxima to the object with the largest maximum
            if d_id in det_to_target:
                old_max_val = self.original_img[self.id_to_max[det_to_target[d_id]]]
                new_max_val = self.original_img[max_loc]

                if old_max_val < new_max_val:
                    det_to_target[d_id] = t_id

            else:
                det_to_target[d_id] = t_id

        return det_to_target

    def get_basic_stats(self, detection_map, det_to_target):
        """Calculate statistics for F-score"""

        tp = len(det_to_target)
        fp = len(set(detection_map.ravel())) - tp
        fn = len(self.id_to_max) - tp

        print("OUTSTAT True positive:", tp)
        print("OUTSTAT False negative:", fn)
        print("OUTSTAT False positive:", fp)

        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0

        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0

        print("OUTSTAT Recall:", r)
        print("OUTSTAT Precision:", p)

        try:
            f_score = 2 * ((p * r) / (p + r))
        except ZeroDivisionError:
            f_score = 0

        print("OUTSTAT F score:", f_score)

        return f_score, tp, fp, fn

    def get_merging_scores(self, detection_map):
        """Calculate overmerging and undermerging scores."""

        t_map = self.target_map
        d_map = detection_map.ravel()

        # Sort the detection ID map for faster pixel retrieval
        sorted_ids = d_map.argsort()
        id_set = np.unique(d_map)
        id_set.sort()

        # Get the locations in sorted_ids of the matching pixels
        right_indices = np.searchsorted(d_map, id_set, side='right', sorter=sorted_ids)
        left_indices = np.searchsorted(d_map, id_set, side='left', sorter=sorted_ids)

        # Calculate under-merging score
        um_score = 0.0
        om_score = 0.0

        d_id_to_area = {}

        # UNDER-MERGING - match to target map
        for n in range(len(id_set)):
            # Find the target labels of the pixels with detection label n
            target_vals = t_map[sorted_ids[left_indices[n]:right_indices[n]]]

            d_id_to_area[id_set[n]] = right_indices[n] - left_indices[n]

            # Find the number of pixels with each ID in the target map
            t_ids, t_counts = np.unique(target_vals, return_counts=True)

            # Find the area of the object with the most overlap 
            target_area = self.id_to_area[t_ids[np.argmax(t_counts)]]

            # Find overlap area
            correct_area = np.max(t_counts)

            um_score += (np.double(target_area - correct_area) * correct_area) / target_area

        # Sort the target ID map for faster pixel retrieval
        sorted_ids = t_map.argsort()
        id_set = np.unique(t_map)
        id_set.sort()

        # Get the locations in sorted_ids of the matching pixels
        right_indices = np.searchsorted(t_map, id_set, side='right', sorter=sorted_ids)
        left_indices = np.searchsorted(t_map, id_set, side='left', sorter=sorted_ids)

        # OVER-MERGING (modified) - match to detection map
        for n in range(len(id_set)):
            # Find the target labels of the pixels with detection label n
            detection_vals = d_map[sorted_ids[left_indices[n]:right_indices[n]]]

            # Find the number of pixels with each ID in the target map
            d_ids, d_counts = np.unique(detection_vals, return_counts=True)

            # Find the area of the object with the most overlap 
            detection_area = d_id_to_area[d_ids[np.argmax(d_counts)]]

            # Find overlap area
            correct_area = np.max(d_counts)

            om_score += (np.double(detection_area - correct_area) * correct_area) / detection_area

        # Calculate over-merging score        
        print("OUTSTAT Undermerging score:", um_score / detection_map.size)

        # Calculate over-merging score        
        print("OUTSTAT Overmerging score:", om_score / detection_map.size)

        total_score = np.sqrt((om_score ** 2) + (um_score ** 2)) / detection_map.size
        print("OUTSTAT Total area score:", total_score)

        return um_score / detection_map.size, om_score / detection_map.size, total_score

    def bg_distribution_test(self, detection_map):
        """Calculate the properties of the background pixels of the image."""
        bg_id = detection_map.min()
        bg_pixels = self.original_img[np.where(detection_map == bg_id)].ravel()

        s, p = stats.skewtest(bg_pixels)
        k, p = stats.kurtosistest(bg_pixels)
        bg_mean = bg_pixels.mean()

        print("OUTSTAT Background skew score:", s)
        print("OUTSTAT Background kurtosis score:", k)
        print("OUTSTAT Background mean score:", bg_mean)

        return s, k, bg_mean

    def get_p_score(self, detection_map):
        """Calculate all the scores for an image, and return them (+ one to optimise on)"""

        # Match detected ids to target ids
        det_to_target = self.match_to_bp_list(detection_map)

        # F score
        f_score, tp, fp, fn = self.get_basic_stats(detection_map, det_to_target)

        # Area score
        um, om, area_score = self.get_merging_scores(detection_map)

        # Background skew score
        s, k, bg_mean = self.bg_distribution_test(detection_map)

        # Combined scores
        combined_one = np.sqrt((f_score ** 2) + (area_score ** 2))
        combined_two = np.cbrt((1 - om) * (1 - um) * f_score)

        print("OUTSTAT Combined A:", combined_one)
        print("OUTSTAT Combined B:", combined_two)

        eval_stats = [tp, fp, fn, f_score, um, om, area_score, s, k, bg_mean, combined_one, combined_two]

        # matches = list(det_to_target.items())
        return [self.mos_name, eval_stats]


def eval_cube(cube_file, data_dir, scale, method):
    orig_cube = fits.getdata(cube_file)
    target_file = data_dir + "training/Target/mask_" + cube_file.split("/")[-1].split("_")[-1]
    target_cube = fits.getdata(target_file)
    mask_labels = skmeas.label(target_cube)
    mos_name = cube_file.split("/")[-1].split("_")[-1].split(".fits")[0]
    eve = Evaluator(orig_cube, mask_labels, mos_name)
    if method == "MTO":
        binary_im = fits.getdata(data_dir + "mto_output/mtocubeout_" + scale + "_" + mos_name+  ".fits")
        nonbinary_im = skmeas.label(binary_im)
    elif method == "VNET":
        binary_im = fits.getdata(data_dir + "vnet_output/vnet_cubeout_" + scale + "_" + mos_name+  ".fits")
        nonbinary_im = skmeas.label(binary_im)
    elif method == "SOFIA":
        nonbinary_im = fits.getdata(data_dir + "sofia_output/sofia_" + scale + "_" + mos_name+  "_mask.fits")
    evaluated = eve.get_p_score(nonbinary_im)
    return evaluated


def main(data_dir, scale, output_dir, method):
    cube_files = [data_dir + "training/" +scale+"Input/" + i for i in listdir(data_dir+scale+"Input") if "_1245mos" in i]
    eval_stats = []
    for cube_file in cube_files:
        final_eval = eval_cube(cube_file, data_dir, scale, method)
        eval_stats += final_eval
    with open(output_dir+scale+'_' + method + '_eval.txt', "wb") as fp:
        pickle.dump(eval_stats, fp)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Methods",
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

    main(args.data_dir, args.scale, args.output_dir, args.method)