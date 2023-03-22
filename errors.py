
from typing import Dict
import numpy as np
import cv2
import math

def calc_mAA(MAEs, ths = np.logspace(np.log2(1.0), np.log2(20), 10, base=2.0)):
    res = 0
    cur_results = []
    for k, MAE in MAEs.items():
        acc = []
        for th in ths:
            A = (MAE <= th).astype(np.float32).mean()
            acc.append(A)
        cur_results.append(np.array(acc).mean())
    res = np.array(cur_results).mean()
    return res

def calc_mAA_pose(MAEs, ths = np.linspace(1.0, 10, 10)):
    res = 0
    cur_results = []
    if isinstance(MAEs, dict):
        for k, MAE in MAEs.items():
            acc = []
            for th in ths:
                A = (MAE <= th).astype(np.float32).mean()
                acc.append(A)
            cur_results.append(np.array(acc).mean())
        res = np.array(cur_results).mean()
    else:
        acc = []
        for th in ths:
            A = (MAEs <= th).astype(np.float32).mean()
            acc.append(A)
        res = np.array(acc).mean()

    return res

def reprojection_errors(pts1, pts2, H1to2):
    pts1_in_2 = cv2.convertPointsFromHomogeneous(cv2.transform(cv2.convertPointsToHomogeneous(pts1), H1to2)).squeeze()
    error = np.sum((pts2 - pts1_in_2)**2, axis=1)
    return np.sqrt(np.abs(error))

def reprojection_error(pts1, pts2, H1to2):
    pts1_in_2 = cv2.convertPointsFromHomogeneous(cv2.transform(cv2.convertPointsToHomogeneous(pts1), H1to2)).squeeze()
    error = np.sum((pts2 - pts1_in_2)**2, axis=1)
    error = np.sqrt(np.abs(error)).mean()
    return error

def decomposeHomography(homography):
    u, s, vt = np.linalg.svd(homography)

    H2 = homography / s[1]
    
    U2, S2, Vt2 = np.linalg.svd(H2.T @ H2)
    V2 = Vt2.T

    if np.linalg.det(V2) < 0:
        V2 *= -1

    s1 = S2[0]
    s3 = S2[2]

    v1 = V2[:,0]
    v2 = V2[:,1]
    v3 = V2[:,2]

    if abs(s1 - s3) < 1e-14:
        return 0, [], [], []

    # compute orthogonal unit vectors
    u1 = (math.sqrt(1.0 - s3) * v1 + math.sqrt(s1 - 1.0) * v3) / math.sqrt(s1 - s3)
    u2 = (math.sqrt(1.0 - s3) * v1 - math.sqrt(s1 - 1.0) * v3) / math.sqrt(s1 - s3)

    U1 = np.zeros((3,3)) 
    W1 = np.zeros((3,3)) 
    U2 = np.zeros((3,3)) 
    W2 = np.zeros((3,3)) 

    U1[:,0] = v2
    U1[:,1] = u1
    U1[:,2] = np.cross(v2, u1)

    W1[:,0] = H2 @ v2
    W1[:,1] = H2 @ u1
    W1[:,2] = np.cross(H2 @ v2, H2 @ u1)

    U2[:,0] = v2
    U2[:,1] = u2
    U2[:,2] = np.cross(v2, u2)

    W2[:,0] = H2 @ v2
    W2[:,1] = H2 @ u2
    W2[:,2] = np.cross(H2 @ v2, H2 @ u2)

    # compute the rotation matrices
    R1 = W1 @ U1.T
    R2 = W2 @ U2.T

    # build the solutions, discard those with negative plane normals
    # Compare to the original code, we do not invert the transformation.
    # Furthermore, we multiply t with -1.
    Rs = []
    ts = []
    ns = []
    
    n = np.cross(v2, u1)
    ns.append(n)
    Rs.append(R1)
    t = -(H2 - R1) @ n
    ts.append(t)

    ns.append(-n)
    t = (H2 - R1) @ n
    Rs.append(R1)
    ts.append(t)

    n = np.cross(v2, u2)
    ns.append(n)
    t = -(H2 - R2) @ n
    Rs.append(R2)
    ts.append(t)

    ns.append(-n)
    t = (H2 - R2) @ n
    ts.append(t)
    Rs.append(R2)

    return 1, Rs, ts, ns

def homography_pose_error(H1to2, scene_scale, pose, K1, K2):
    R = pose[:, 0:3]
    t = pose[:, 3]

    normalizedHomography = np.linalg.inv(K2).dot(H1to2).dot(K1)

    #retval, rotations, translations, normals = decomposeHomography(normalizedHomography)
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

def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

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

    #if q_gt is None:
    #    q_gt = quaternion_from_matrix(R_gt)
    #q = quaternion_from_matrix(R)
    #q = q / (np.linalg.norm(q) + eps)
    #q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    #loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    #err_r = np.arccos(1 - 2 * loss_q)

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