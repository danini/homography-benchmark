import numpy as np
import h5py
#import cv2
import time
import multiprocessing
import os
import string
import math
from copy import deepcopy
import sys
import torch
from tqdm import tqdm
if True:
    sys.path.append('third_party/OANet/core')
    # Monkey patch to run oanet
    torch_version = torch.version.__version__
    t_major = torch_version.split('.')[0]
    t_minor = torch_version.split('.')[1]
    #if int(t_major) >1:
    #    torch.symeig = torch.linalg.eigh
    #elif int(t_minor) > 9:
    #    torch.symeig = torch.linalg.eigh

    
    from oan import OANet
else:
    print ("Cannot import OANet stuff")
    sys.exit(0)
import numpy as np
import argparse

import glob

from collections import namedtuple
from kornia.feature.adalam.adalam import AdalamFilter, get_adalam_default_config

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
#import cv2
import time
from tqdm import utils
import yaml
import csv
import gc
import argparse
import multiprocessing as mp
mp.set_start_method('spawn')

from copy import deepcopy
from statistics import median
from database import load_h5, save_results


class NNMatcher(object):
    """docstring for NNMatcher"""
    def __init__(self, ):
        super(NNMatcher, self).__init__()

    def run(self, nkpts, descs):
        # pts1, pts2: N*2 GPU torch tensor
        # desc1, desc2: N*C GPU torch tensor
        # corr: N*4
        # sides: N*2
        # corr_idx: N*2

        pts1, pts2, desc1, desc2 = nkpts[0], nkpts[1], descs[0], descs[1]
        d1, d2 = (desc1**2).sum(1), (desc2**2).sum(1)
        distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc1, desc2.transpose(0,1))).sqrt()
        dist_vals, nn_idx1 = torch.topk(distmat, k=2, dim=1, largest=False)
        nn_idx1 = nn_idx1[:,0]
        _, nn_idx2 = torch.topk(distmat, k=1, dim=0, largest=False)
        nn_idx2= nn_idx2.squeeze()
        mutual_nearest = (nn_idx2[nn_idx1] == torch.arange(nn_idx1.shape[0]).cuda())
        ratio_test = dist_vals[:,0] / dist_vals[:,1].clamp(min=1e-15)
        pts2_match = pts2[nn_idx1, :]
        corr = torch.cat([pts1, pts2_match], dim=-1)
        corr_idx = torch.cat([torch.arange(nn_idx1.shape[0]).unsqueeze(-1), nn_idx1.unsqueeze(-1).cpu()], dim=-1)
        sides = torch.cat([ratio_test.unsqueeze(1), mutual_nearest.float().unsqueeze(1)], dim=1)
        return corr, sides, corr_idx

    def infer(self, kpt_list, desc_list):
        nkpts = [torch.from_numpy(i[:,:2].astype(np.float32)).cuda() for i in kpt_list]
        descs = [torch.from_numpy(desc.astype(np.float32)).cuda() for desc in desc_list]
        corr, sides, corr_idx = self.run(nkpts, descs)
        inlier_idx = np.where(sides[:,1].cpu().numpy())
        matches = corr_idx[inlier_idx[0], :].numpy().astype('int32')
        corr0 = kpt_list[0][matches[:, 0]]
        corr1 = kpt_list[1][matches[:, 1]]
        return matches, corr0, corr1

        
class LearnedMatcher(object):
    def __init__(self, model_path, inlier_threshold=0, use_ratio=2, use_mutual=2, device=torch.device('cpu')):
        self.default_config = {}
        self.default_config['net_channels'] = 128
        self.default_config['net_depth'] = 12
        self.default_config['clusters'] = 500
        self.default_config['use_ratio'] = use_ratio
        self.default_config['use_mutual'] = use_mutual
        self.default_config['iter_num'] = 1
        self.default_config['inlier_threshold'] = inlier_threshold
        self.default_config = namedtuple("Config", self.default_config.keys())(*self.default_config.values())

        self.model = OANet(self.default_config)
        self.device = device
        print('load model from ' +model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

    def normalize_kpts(self, kpts):
        x_mean = np.mean(kpts, axis=0)
        dist = kpts - x_mean
        meandist = np.sqrt((dist**2).sum(axis=1)).mean()
        scale = np.sqrt(2) / meandist
        T = np.zeros([3,3])
        T[0,0], T[1,1], T[2,2] = scale, scale, 1
        T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
        nkpts = kpts * np.asarray([T[0, 0], T[1, 1]]) + np.array([T[0, 2], T[1, 2]])
        return nkpts



    def infer(self, x1y1, x2y2, is_mutual, snn_ratio):
        with torch.no_grad():
            nkpts = [torch.from_numpy(self.normalize_kpts(i[:,:2]).astype(np.float32)).to(self.device) for i in [x1y1, x2y2]]
            sides = torch.cat([snn_ratio.view(-1,1), is_mutual], dim=1)
            corr = torch.cat(nkpts, dim=1)
            corr, sides = corr.unsqueeze(0).unsqueeze(0), sides.unsqueeze(0)
            data = {}
            data['xs'] = corr
            # currently supported mode:
            if self.default_config.use_ratio==2 and self.default_config.use_mutual==2:
                data['sides'] = sides
            y_hat, e_hat = self.model(data)
            y = y_hat[-1][0, :].cpu().numpy()
        return y.reshape(-1)

def run_adalam(data, pairs):
    verbose=False
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    except:
        device = torch.device('cpu')
    out = {}
    
    config_ = get_adalam_default_config()
    config_['device'] = device
    adalam_object = AdalamFilter(config_)
    for ki in tqdm(range(len(pairs))):
        k = pairs[ki]
        corrs = data[f'corr_{k}']
        x1y1 = torch.from_numpy(corrs[:, :2]).float().to(device)
        x2y2 = torch.from_numpy(corrs[:, 2:4]).float().to(device)
        angle1 = torch.from_numpy(corrs[:, 4]).float().to(device)
        angle2 = torch.from_numpy(corrs[:, 5]).float().to(device)
        scale1 = torch.from_numpy(corrs[:, 6]).float().to(device)
        scale2 = torch.from_numpy(corrs[:, 7]).float().to(device)
        snn_ratio = torch.from_numpy(corrs[:, 8]).float().to(device)
        h1w1 = data[f"size_{ '_'.join(k.split('_')[0:3]) }"]
        h2w2 = data[f"size_{ '_'.join(k.split('_')[3:6]) }"]
        with torch.no_grad():
            idxs = adalam_object.filter_matches(
                x1y1,
                x2y2,
                torch.arange(len(x1y1)).to(device),
                snn_ratio,
                im1shape = h1w1,
                im2shape = h2w2,
                o1 = angle1,
                o2 = angle2,
                s1 = scale1,
                s2 = scale2,
                return_dist = False
            ).cpu()
        w = torch.zeros(len(x1y1))
        w[idxs[:,0]] = 1.0
        is_inlier_gt = corrs[:,9].astype(bool)
        inlier_ratio = is_inlier_gt.sum() / len(is_inlier_gt)
        if verbose:
            surv1  = np.argsort(w.numpy())[::-1]
            surv2  = (w > 0.5)
            mask1 = is_inlier_gt[surv1][:100]
            mask2 = is_inlier_gt[surv2]
            inlier_ratio1 = mask1.sum() / len(mask1)
            inlier_ratio2 = mask2.sum() / len(mask2)
            print (f'Orig inl_ratio={inlier_ratio:.4f}, Adalam top100={inlier_ratio1:.4f}, adalam thresholded at 0.5={inlier_ratio2:.4f}') 
            print (f'Orig inl num={is_inlier_gt.sum():.4f}, adalam top100={mask1.sum():.4f}, adalam thresholded at 0.5={mask2.sum():.4f}') 
            print ("""""""")
        out[k] = w.detach().cpu().numpy()
    return out

def run_oanet(data, pairs):
    verbose = True
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    except:
        device = torch.device('cpu')
    out = {}
    model_path = os.path.join('third_party/OANet/model/yfcc/fundamental/sift-side-8k', 'model_best.pth')
    matcher = LearnedMatcher(model_path, 0, use_ratio=2, use_mutual=2, device=device)
    for ki in tqdm(range(len(pairs))):
        k = pairs[ki]
        corrs = data[f'corr_{k}']
        x1y1 = corrs[:, :2]
        x2y2 = corrs[:, 2:4]
        snn_ratio = torch.from_numpy(corrs[:, 8:9]).float().to(device)
        with torch.no_grad():
            w =  matcher.infer(x1y1, x2y2,torch.ones_like(snn_ratio), snn_ratio )
        is_inlier_gt = corrs[:,9].astype(bool)
        inlier_ratio = is_inlier_gt.sum() / len(is_inlier_gt)
        if verbose:
            surv1  = np.argsort(w.numpy())[::-1]
            surv2  = (w > 0.5)
            mask1 = is_inlier_gt[surv1][:100]
            mask2 = is_inlier_gt[surv2]
            inlier_ratio1 = mask1.sum() / len(mask1)
            inlier_ratio2 = mask2.sum() / len(mask2)
            print (f'Orig inl_ratio={inlier_ratio:.4f}, OANet top100={inlier_ratio1:.4f}, OANet thresholded at 0.5={inlier_ratio2:.4f}') 
            print (f'Orig inl num={is_inlier_gt.sum():.4f}, OANet top100={mask1.sum():.4f}, OANet thresholded at 0.5={mask2.sum():.4f}') 
            print ("""""""")
        out[k] = w.reshape(-1)
    return out
def run_deep_method(method, data, filtered_pairs):
    if method.lower()== "oanet":
        return run_oanet(data, filtered_pairs)
    if method.lower()== "adalam":
        return run_adalam(data, filtered_pairs)
    return {}



def save_h5(dict_to_save, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep pretrained inferentce")
    parser.add_argument('--path', type=str, help="The path to the dataset. It should contain two folders: 'test' and 'train'", default="/")
    parser.add_argument('--split', type=str, help='Choose a split: train, test', default='train', choices=['test', 'train'])
    parser.add_argument('--scene', type=str, help='Choose a scene.', default='all', choices=['all', 'NYC_Library', 'Alamo', 'Yorkminster', 'Tower_of_London', 'Madrid_Metropolis', 'Ellis_Island', 'Roman_Forum', 'Vienna_Cathedral', 'Piazza_del_Popolo', 'Union_Square'])
    parser.add_argument("--config_path", type=str, default='dataset_configuration.yaml')
    parser.add_argument(
        "--deepmethod", type=str, default='OANet', choices=['OANet', 'AdaLAM'])
    parser.add_argument(
        "--save_to_dir", type=str, default='deep_filtered')
    args = parser.parse_args()
    
    # Parameters
    save_to_file = True
    print (args)
    out_fname = os.path.join(args.save_to_dir, args.deepmethod)
    root = args.path
    config_fname = args.config_path
    split = args.split.upper()
    print(f"Running {args.deepmethod} filtering on the '{split}' split of HEB")

    # Loading the configuration file
    with open(config_fname, "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    print (configuration)
    for scene in configuration[f'{split}_SCENES']:
        # Check if the method should run on a single scene
        if args.scene != "all" and scene['name'] != args.scene:
            continue
        
        print(100 * "-")
        print(f"Loading scene '{scene['name']}'")
        print(100 * "-")
        
        # Loading the dataset
        # Getting the ground truth reconstruction scale that transform it to metric reconstruction.
        scale = scene['scale']
        print(f"The scene scale is {scale}")
    
        # Loading the dataset
        input_fname = os.path.join(args.path, args.split, scene['filename'])
        data = load_h5(input_fname)
        pairs = sorted([x.replace('corr_','') for x in data.keys() if x.startswith('corr_')])
        print(f"{len(pairs)} image pairs are loaded.")
        
        gt_inlier_ratios = []
        current_out_fname = os.path.join(out_fname, scene['name']+'.h5')
        print(f"Processing scenes '{scene['name']}' with {len(pairs)} image pairs")


        scores_dict = run_deep_method(args.deepmethod, data, pairs)
        save_h5(scores_dict, current_out_fname)
        del data
        del pairs
        gc.collect()