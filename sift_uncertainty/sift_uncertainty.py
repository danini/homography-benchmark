import numpy as np
import h5py
import cv2
import math
import multiprocessing as mp
from scipy.io import savemat
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import logm, expm
from multiprocessing import Pool


def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load

def evaluate_R_t(R_gt, t_gt, R, t, scale=None, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15
    err_abs_t = 0

    if scale != None:
        t_gt = scale * t_gt
        t = np.linalg.norm(t_gt) * t / (np.linalg.norm(t) + eps)
        err_abs_t = np.linalg.norm(t - t_gt)
        
    R2R1 = np.dot(R_gt, np.transpose(R))
    cos_angle = max(min(1.0, 0.5 * (np.trace(R2R1) - 1.0)), -1.0)
    err_r = math.acos(cos_angle)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))
        

    if np.sum(np.isnan(err_r)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_r, err_t, err_abs_t
    
def homography_pose_error(H1to2, scene_scale, pose, K1, K2):
    R = pose[:, 0:3]
    t = pose[:, 3]

    normalizedHomography = np.linalg.inv(K2).dot(H1to2).dot(K1)

    retval, rotations, translations, normals = cv2.decomposeHomographyMat(normalizedHomography, np.identity(3))

    minRotationError = 1e10
    minTranslationError = 1e10
    minAbsoluteTranslationError = 1e10
    minError = 1e10

    for i in range(len(rotations)):
        Rest = rotations[i]
        test = translations[i] 

        if np.isnan(Rest).any() or np.isnan(test).any():
            continue

        try:
            err_R, err_t, err_abs_t = evaluate_R_t(R, t, Rest, test, scale=scene_scale, q_gt=None)

            if err_R + err_t + err_abs_t < minError:
                minError = err_R + err_t + err_abs_t
                minRotationError = err_R
                minTranslationError = err_t
                minAbsoluteTranslationError = err_abs_t
        except:
            print("Error!")
            continue
    
    return 180.0 / math.pi * minRotationError, 180.0 / math.pi * minTranslationError, minAbsoluteTranslationError

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    return (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

def filter_pairs(pairs, data, scale, kept_pairs):
    for i in range(len(pairs)):
        corrs = data[f'corr_{pairs[i]}']
        is_inlier_gt = corrs[:,9].astype(bool)
        
        if is_inlier_gt.sum() < 10:
            continue
        
        pose = data[f'pose_{pairs[i]}']
        K1 = data[f"K_{ '_'.join(pairs[i].split('_')[0:3]) }"]
        K2 = data[f"K_{ '_'.join(pairs[i].split('_')[3:6]) }"]
        H_est = data[f"hom_{pairs[i]}"]

        rotation_error, translation_error, absolute_translation_error = homography_pose_error(H_est, scale, pose, K1, K2)

        if rotation_error <= 10 and absolute_translation_error <= 5:
            kept_pairs.append(int(i))
    return kept_pairs

def decompose_affinity_logm(A):
    Bm = np.matrix([[1,0,1,0],[0,1,0,1],[0,-1,0,1],[1,0,-1,0]])
    lA = logm(A)
    p = np.linalg.inv(Bm) * np.matrix(lA.reshape((-1,),order='F')).T
    # A_scale = expm(np.reshape(Bm[:,0]*p[0],(2,2),order='F'))
    A_rotation = np.real(expm(np.reshape(Bm[:,1]*p[1],(2,2),order='F')))
    # A_shear1 = expm(np.reshape(Bm[:,2]*p[2],(2,2),order='F'))
    # A_shear2 = expm(np.reshape(Bm[:,3]*p[3],(2,2),order='F'))
    shear_magnitude = np.real(np.exp(np.sqrt(p[2]**2 + p[3]**2)))
    return (A_rotation, shear_magnitude)  # A_scale, , A_shear1, A_shear2

def decompose_affinity(A):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    R = U @ V.T
    angle_svd = math.atan2(R[1, 0], R[0,0])    
    cond_num_svd = S[0]/S[1]

    # R, Q = np.linalg.qr(A, mode='reduced')
    # angles_qt.append(math.atan2(R[1, 0], R[0,0]))      

    # Rt, Q = np.linalg.qr(A.T, mode='reduced')
    # R = Rt.T
    # angles_qt2.append(math.atan2(R[1, 0], R[0,0]))

    A_rotation, shear_magnitude = decompose_affinity_logm(A)
    angle_logm = math.atan2(A_rotation[1, 0], A_rotation[0,0])
    shear_mag_logm = shear_magnitude

    scale = math.sqrt(np.abs(np.linalg.det(A)))
    return (angle_svd, angle_logm, scale, cond_num_svd, shear_mag_logm)

def decompose_affines(ACs):
    angles_svd = []
    # angles_qt = np.zeros((len(ACs),))
    # angles_qt2 = np.zeros((len(ACs),))
    angles_logm = []
    cond_num_svd = []
    shear_mag_logm = []
    scales = []
    
    with Pool(processes=64) as pool:
        for res in tqdm(pool.imap(decompose_affinity, ACs), total=len(ACs)):
            angles_svd.append(res[0]) 
            angles_logm.append(res[1])
            scales.append(res[2])  
            cond_num_svd.append(res[3]) 
            shear_mag_logm.append(res[4]) 
    # for A in ACs:
    #     res = decompose_affinity(A)
    #     angles_svd.append(res[0]) 
    #     angles_logm.append(res[1])
    #     scales.append(res[2])  
    #     cond_num_svd.append(res[3]) 
    #     shear_mag_logm.append(res[4]) 

    return (angles_svd, angles_logm, scales, cond_num_svd, shear_mag_logm)  #angles_qt, angles_qt2,

def affines_from_homography(H, pts1, pts2):
    ACs = np.zeros((pts1.shape[0], 2, 2))

    hom1 = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    hom2 = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)
    s = H[2,:] @ hom1.T
    
    a11s = (H[0, 0] - H[2, 0] * pts2[:,0]) / s
    a12s = (H[0, 1] - H[2, 1] * pts2[:,1]) / s
    a21s = (H[1, 0] - H[2, 0] * pts2[:,0]) / s
    a22s = (H[1, 1] - H[2, 1] * pts2[:,1]) / s

    ACs[:, 0, 0] = a11s
    ACs[:, 0, 1] = a12s
    ACs[:, 1, 0] = a21s
    ACs[:, 1, 1] = a22s

    return ACs

def affines_from_homography_wf(H, x):
    hom_x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1).T
    y = H @ hom_x
    ACs = np.zeros((len(y.T), 2, 2))
    for i in range(len(y.T)):
        Ty = np.matrix([[1,0,-y[0,i]/y[2,i]],[0,1,-y[1,i]/y[2,i]],[0,0,1]])
        Hc = Ty @ H
        ACs[i, 0:2, 0:2] = Hc[0:2,0:2] / y[2,i]
    return ACs


def affines_from_homography_wf2(H, pts1):
    ACs = np.zeros((pts1.shape[0], 2, 2))
    hom1 = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    pts2 = (H @ hom1.T).T 
    s = pts2[:,2] 
    sq = s*s
    a11s = (H[0, 0] * s - H[2, 0] * pts2[:,0] ) / sq
    a12s = (H[0, 1] * s - H[2, 1] * pts2[:,0] ) / sq
    a21s = (H[1, 0] * s - H[2, 0] * pts2[:,1] ) / sq
    a22s = (H[1, 1] * s - H[2, 1] * pts2[:,1] ) / sq
    ACs[:, 0, 0] = a11s
    ACs[:, 0, 1] = a12s
    ACs[:, 1, 0] = a21s
    ACs[:, 1, 1] = a22s
    return ACs

def get_translation_error(x1y1, x2y2, H1to2):
    hom1 = np.concatenate([x1y1, np.ones((x1y1.shape[0], 1))], axis=1).T
    H_hom1 = H1to2 @ hom1
    H_hom1[0,:] = H_hom1[0,:] / H_hom1[2,:]
    H_hom1[1,:] = H_hom1[1,:] / H_hom1[2,:]
    H_hom1 = H_hom1[0:2,:]

    error = np.zeros(x1y1.shape[0])
    for i in range(x1y1.shape[0]):
        error[i] = np.sqrt((H_hom1[0,i] - x2y2[i,0])**2 + (H_hom1[1,i] - x2y2[i,1])**2)

    return error

def collect_all(scene, scale, data):
    data_path = f'./scenes/updated_{scene}_homographies.h5'
    # print(f"Loading '{data_path}' ...")
    h5_data = load_h5(data_path)
    pairs = sorted([x.replace('corr_','') for x in h5_data.keys() if x.startswith('corr_')])
    
    # print(f"Filter pairs '{scene}' ...")
    kept_pairs = []
    kept_pairs = filter_pairs(pairs, h5_data, scale, kept_pairs)
    accessed_mapping = map(pairs.__getitem__, kept_pairs)
    pairs = list(accessed_mapping)

    # print(f"Collecting values from {len(pairs)} pairs ...")
    for pair in pairs:
        corrs = h5_data[f'corr_{pair}']
        is_inlier_gt = corrs[:,9].astype(bool)
        inlier_corrs = corrs[is_inlier_gt]
        data['sift_a1'].extend(inlier_corrs[:, 4])
        data['sift_a2'].extend(inlier_corrs[:, 5])
        data['sift_s1'].extend(inlier_corrs[:, 6])
        data['sift_s2'].extend(inlier_corrs[:, 7])
        data['sift_u1'].extend(inlier_corrs[:, :2])
        data['sift_u2'].extend(inlier_corrs[:, 2:4])
        data['gt_Hids'].extend(np.ones((np.shape(inlier_corrs)[0],),dtype=int)*len(data['gt_H1to2']))
        data['gt_H1to2'].append(h5_data[f"hom_{pair}"])

    N = len(data['sift_a1'])
    print(f"Loaded {N} keypoint pairs in '{scene}'.")
    return data

def cluster_reprojection_errors(s1, s2, eps_x, num_inliers, clusters_mean, N_clusters):
    eps_s1s2 = [ [ [] for i in range(N_clusters) ] for j in range(N_clusters) ]
    eps_s1s2_robust = [ [ [] for i in range(N_clusters) ] for j in range(N_clusters) ]
    for i in tqdm(range(len(eps_x))):
        j = np.argmin((clusters_mean - s1[i])**2)
        t = np.argmin((clusters_mean - s2[i])**2)
        eps_s1s2[j][t].append(eps_x[i])
        eps_s1s2_robust[j][t].append(eps_x[i] / np.sqrt((4*num_inliers[i]-8)/(4*num_inliers[i])))
    return (eps_s1s2, eps_s1s2_robust)

def positional_statistics_for_bins(eps_grid, eps_grid_robust, N_clusters, min_samples):
    eps_std = np.empty((N_clusters,N_clusters))
    eps_std_robust = np.empty((N_clusters,N_clusters))
    for i in range(N_clusters):
        for j in range(N_clusters):
            if len(eps_grid[i][j]) >= min_samples:
                eps_std[i][j] = np.sqrt(np.sum(np.square(eps_grid[i][j]))/len(eps_grid[i][j]))
                eps_std_robust[i][j] = 1.76 * np.median(eps_grid_robust[i][j])
            else:
                eps_std[i][j] = np.nan
                eps_std_robust[i][j] = np.nan
    return (eps_std, eps_std_robust)

def plot_heatmap(values, x_ticks, y_ticks, title, xlabel, ylabel, save_as):
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=cm.gray)
    ax.set_xticks(np.arange(len(x_ticks)), labels=x_ticks)
    ax.set_yticks(np.arange(len(y_ticks)), labels=y_ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(y_ticks)):
        for j in range(len(x_ticks)):
            text = ax.text(j, i, f'{values[i, j]:.2}', ha="center", va="center", color="w")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_as)
    plt.show()

def directions_to_differential_angle(d2, Xd1):
    col_norms = np.sqrt((Xd1**2).sum(axis=0))
    Xd1 = Xd1 / np.array([col_norms, col_norms])
    cos_d12 = np.sum(d2 * Xd1, axis=0)
    sin_d12 = d2[0,:] * Xd1[1,:] - d2[1,:] * Xd1[0,:]
    delta_alpha = np.arctan2(sin_d12, cos_d12)
    return delta_alpha