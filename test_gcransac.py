import cv2
import h5py
import os
import numpy as np
import argparse
import yaml
import math
import time
from database import load_h5
from joblib import Parallel, delayed
from tqdm import tqdm
from errors import reprojection_error, homography_pose_error, calc_mAA, calc_mAA_pose, homography_pose_error, reprojection_error
import pygcransac

def get_LAF(kps, sc, ori):
    '''
    Converts OpenCV keypoints into transformation matrix
    and pyramid index to extract from for the patch extraction
    '''
    num = len(kps)
    out = np.zeros((num, 2, 2)).astype(np.float64)
    for i, kp in enumerate(kps):
        s = 12.0 * sc[i]
        a = ori[i]
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)
        out[i, 0, 0] = s * cos
        out[i, 0, 1] = -s * sin
        out[i, 1, 0] = s * sin
        out[i, 1, 1] = s * cos
    return out

def run_gcransac(pair, scene_scale, matches, relative_pose, image_size1, image_size2, K1, K2, args):
    # Initialize the errors
    error = 1e10
    rotation_error = 1e10
    translation_error = 1e10
    absolute_translation_error = 1e10
    
    # The SNN ratio
    snn_ratio = matches[:, 8]
    # A flag determining whether the point is GT inlier or not.
    is_inlier_gt = matches[:,9].astype(bool)
    
    # Filter by the SNN ratio threshold
    snn_mask = snn_ratio < args.snn_threshold

    # Run point-based solver
    if args.solver == 0:
        # The point correspondences
        correspondences = matches[snn_mask, :4].astype(np.float64)  
    # Run SIFT-based solver
    elif args.solver == 1:
        # The point correspondences
        points = matches[snn_mask, :4]
        # SIFT angles in the source image
        angle1 = matches[snn_mask, 4] / 180.0 * np.pi
        # SIFT angles in the destinstion image
        angle2 = matches[snn_mask, 5] / 180.0 * np.pi
        # SIFT scale in the source image
        scale1 = matches[snn_mask, 6]
        # SIFT scale in the destination image
        scale2 = matches[snn_mask, 7]
        # The point correspondences
        correspondences = np.concatenate((points, np.expand_dims(scale1, axis=1), np.expand_dims(scale2, axis=1), np.expand_dims(angle1, axis=1), np.expand_dims(angle2, axis=1)), axis=1)
    # Run affine-based solver
    elif args.solver == 2:
        # SIFT angles in the source image
        angle1 = matches[snn_mask, 4] / 180.0 * np.pi
        # SIFT angles in the destinstion image
        angle2 = matches[snn_mask, 5] / 180.0 * np.pi
        # SIFT scale in the source image
        scale1 = matches[snn_mask, 6]
        # SIFT scale in the destination image
        scale2 = matches[snn_mask, 7]

        laf1 = get_LAF(matches[snn_mask, :2], scale1, angle1)
        laf2 = get_LAF(matches[snn_mask, 2:4], scale2, angle2)

        # Calculating the affine transformation from the source to the destination image
        affines = [np.matmul(a2, np.linalg.inv(a1)) for a1, a2 in zip(laf1, laf2)]
        affines = np.array(affines).reshape(-1, 4)

        # Constructing the correspondences
        correspondences = np.concatenate((matches[snn_mask, :4], affines), axis=1)
    # Inlier probabilities
    probabilities = []

    # Return if there are fewer than 4 correspondences
    if correspondences.shape[0] < 4:
        return 0, error, rotation_error, translation_error, 0, absolute_translation_error

    # If the sampler is something else than uniform sampler, the points should be ordered by SNN ratio
    if args.sampler > 0:
        snn_ratio = snn_ratio[snn_mask]
        indices = np.argsort(snn_ratio)
        snn_ratio = snn_ratio[indices]
        correspondences = correspondences[indices, :]
        
        # If Importance sampler or ARSampler is used, the SNN ratios are converted as probabilities
        if args.sampler == 3 or args.sampler == 4:
            point_number = correspondences.shape[0]
            for i in range(point_number):
                probabilities.append(1.0 - i / point_number)

    # Run the homography estimation implemented in OpenCV
    tic = time.perf_counter()
    H_est, inliers = pygcransac.findHomography(
        np.ascontiguousarray(correspondences), 
        int(image_size1[0][0]), int(image_size1[1][0]), 
        int(image_size2[0][0]), int(image_size2[1][0]),
        use_sprt = False,
        use_space_partitioning = True,
        probabilities = probabilities,
        spatial_coherence_weight = args.spatial_coherence_weight,
        neighborhood_size = args.neighborhood_size,
        conf = args.confidence,
        sampler = args.sampler,
        min_iters = args.minimum_iterations,
        max_iters = args.maximum_iterations,
        solver = args.solver,
        threshold = args.inlier_threshold)
    toc = time.perf_counter()
    runtime = toc - tic

    # Count the inliers
    inlier_number = inliers.sum()

    # Calculate the error if enough inliers are found
    if inlier_number >= 4:  
        # The original matches without SNN filtering
        all_x1y1 = matches[:, :2]
        all_x2y2 = matches[:, 2:4]
        # Calculate the re-projection error of the estimated homography given the ground truth inliers
        error = reprojection_error(all_x1y1[is_inlier_gt], all_x2y2[is_inlier_gt], H_est) if is_inlier_gt.sum() > 1 else 1e10
        # Calculate the pose error of the estimated homography given the ground truth relative pose
        rotation_error, translation_error, absolute_translation_error = homography_pose_error(H_est, scene_scale, relative_pose, K1, K2)

    return runtime, error, rotation_error, translation_error, inlier_number, absolute_translation_error

def estimate_homographies(pairs, data, scene_scale, args):
    assert args.snn_threshold > 0
    assert args.inlier_threshold > 0
    assert args.maximum_iterations > 0
    
    # Initialize the arrays where the results will be stored
    times = {}
    errors = {}
    rotation_errors = {}
    translation_errors = {}
    inlier_numbers = {}
    absolute_translation_errors = {}

    # Run homography estimation on all image pairs
    keys = range(len(pairs))
    results = Parallel(n_jobs=min(args.core_number, len(pairs)))(delayed(run_gcransac)(
        pairs[k], # Name of the current pair
        scene_scale, # The ground truth scene scale
        data[f'corr_{pairs[k]}'], # The SIFT correspondences
        data[f'pose_{pairs[k]}'], # The ground truth relative pose coming from the COLMAP reconstruction
        data[f"size_{ '_'.join(pairs[k].split('_')[0:3]) }"], # The size of the source image
        data[f"size_{ '_'.join(pairs[k].split('_')[3:6]) }"], # The size of the destination image
        data[f"K_{ '_'.join(pairs[k].split('_')[0:3]) }"], # The intrinsic matrix of the source image
        data[f"K_{ '_'.join(pairs[k].split('_')[3:6]) }"], # The intrinsic matrix of the destination image
        args) for k in tqdm(keys)) # Other parameters

    for i, k in enumerate(keys):
        times[k] = results[i][0]
        errors[k] = results[i][1]
        rotation_errors[k] = results[i][2]
        translation_errors[k] = results[i][3]
        inlier_numbers[k] = results[i][4]
        absolute_translation_errors[k] = results[i][5]
    return times, errors, rotation_errors, translation_errors, inlier_numbers, absolute_translation_errors

def sampler_flag(name):
    if name == "Uniform":
        return 0
    if name == "PROSAC":
        return 1
    if name == "PNAPSAC":
        return 2
    if name == "Importance":
        return 3
    if name == "ARSampler":
        return 4
    return 0

def solver_flag(name):
    if name == "point":
        return 0
    if name == "sift":
        return 1
    if name == "affine":
        return 2
    return 0

if __name__ == "__main__":
    # Passing the arguments
    parser = argparse.ArgumentParser(description="Running on the HEB benchmark")
    parser.add_argument('--path', type=str, help="The path to the dataset. It should contain two folders: 'test' and 'train'", default="/")
    parser.add_argument('--split', type=str, help='Choose a split: train, test', default='train', choices=['test', 'train'])
    parser.add_argument('--scene', type=str, help='Choose a scene.', default='all', choices=['all', 'NYC_Library', 'Alamo', 'Yorkminster', 'Tower_of_London', 'Madrid_Metropolis', 'Ellis_Island', 'Roman_Forum', 'Vienna_Cathedral', 'Piazza_del_Popolo', 'Union_Square'])
    parser.add_argument("--config_path", type=str, default='dataset_configuration.yaml')
    parser.add_argument("--snn_threshold", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--inlier_threshold", type=float, default=15.0)
    parser.add_argument("--minimum_iterations", type=int, default=50)
    parser.add_argument("--maximum_iterations", type=int, default=1000)
    parser.add_argument("--sampler", type=str, help="Choose from: Uniform, PROSAC, PNAPSAC, Importance, ARSampler.", choices=["Uniform", "PROSAC", "PNAPSAC", "Importance", "ARSampler"], default="PROSAC")
    parser.add_argument("--spatial_coherence_weight", type=float, default=0.1)
    parser.add_argument("--neighborhood_size", type=float, default=20)
    parser.add_argument("--solver", type=str, help="Choose from: point, sift, affine.", choices=["point", "sift", "affine"], default="point")
    parser.add_argument("--core_number", type=int, default=4)

    args = parser.parse_args()
    
    split = args.split.upper()
    print(f"Running GC-RANSAC on the '{split}' split of HEB")
    args.sampler = sampler_flag(args.sampler)
    args.solver = solver_flag(args.solver)

    # Loading the configuration file
    with open(args.config_path, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
            
    # Iterating through the scenes
    for scene in configuration[f'{split}_SCENES']:
        # Check if the method should run on a single scene
        if args.scene != "all" and scene['name'] != args.scene:
            continue
        
        print(100 * "-")
        print(f"Loading scene '{scene['name']}'")
        print(100 * "-")
        
        # Getting the ground truth reconstruction scale that transform it to metric reconstruction.
        scale = scene['scale']
        print(f"The scene scale is {scale}")
    
        # Loading the dataset
        input_fname = os.path.join(args.path, args.split, scene['filename'])
        data = load_h5(input_fname)
        pairs = sorted([x.replace('corr_','') for x in data.keys() if x.startswith('corr_')])
        print(f"{len(pairs)} image pairs are loaded.")
        
        # Run homography estimation on the entire scene
        times, errors, rotation_errors, translation_errors, inlier_numbers, absolute_translation_errors = estimate_homographies(pairs, data, scale, args)
        
        # Calculating the pose error as the maximum of the rotation and translation errors
        maximum_pose_errors = {}
        for i in range(len(pairs)):
            maximum_pose_errors[i] = max(rotation_errors[i], translation_errors[i])

        # Calculating the mean Average Accuracy
        mAA_repr = calc_mAA(errors)
        mAA_max_pose_error = calc_mAA_pose(maximum_pose_errors)
        mAA_rotation = calc_mAA_pose(rotation_errors)
        mAA_abs_translation = calc_mAA_pose(absolute_translation_errors, ths=np.linspace(0.1, 5, 10))
        
        print(f"mAA re-projection error = {mAA_repr:0.4f}")
        print(f"mAA angular pose error = {mAA_max_pose_error:0.4f}")
        print(f"mAA rotation error = {mAA_rotation:0.4f}")
        print(f"mAA abs. translation error = {mAA_abs_translation:0.4f}")
        print(f"Avg. inlier number = {np.mean(list(inlier_numbers.values())):0.1f}")
        print(f"Avg. run-time = {np.mean(list(times.values())):0.4f} secs")
        