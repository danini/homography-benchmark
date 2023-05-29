# Dataset

The dataset is available at [link](https://polybox.ethz.ch/index.php/s/R5sPelZ8688It92).

# Requirements

- Eigen 3.0 or higher
- OpenCV 3.0 or higher

# Running homography estimation with OpenCV

The tests running OpenCV can be started by calling
```
python test_opencv.py --path=<path to the database folder>
```

The following additional parameters are accepted:
```
--split = { train, test } - Select the split to be used. Default: train
--scene = { all, NYC_Library, Alamo, Yorkminster, Tower_of_London, Madrid_Metropolis, Ellis_Island, Roman_Forum, Vienna_Cathedral, Piazza_del_Popolo, Union_Square } - Select the scene. Default: all
--config_path - The path to the config file
--snn_threshold - SNN ratio threshold for SIFT correspondence filtering. Default: 0.80
--confidence - The RANSAC confidence. Default: 0.99
--inlier_threshold - The inlier-outlier threshold in pixels. Default: 15.0
--maximum_iterations - The maximum number of RANSAC iterations. Default: 1000
--core_number - The core number for the parallel processing. Default: 4
--opencv_flag = { RANSAC, LMEDS, RHO, USAC_MAGSAC } - The flag to select the robust estimator from OpenCV. Default: RANSAC
```

# Running homography estimation with MAGSAC++

To build and install `MAGSAC`/`MAGSAC++`, clone the repository as:
```shell
$ git clone https://github.com/danini/magsac --recursive
```

Then it has to be installed by calling
```bash 
$ pip3 install -e .
```

The tests running MAGSAC++ can be started by calling
```
python test_magsac.py --path=<path to the database folder>
```

The following additional parameters are accepted:
```
--split = { train, test } - Select the split to be used. Default: train
--scene = { all, NYC_Library, Alamo, Yorkminster, Tower_of_London, Madrid_Metropolis, Ellis_Island, Roman_Forum, Vienna_Cathedral, Piazza_del_Popolo, Union_Square } - Select the scene. Default: all
--config_path - The path to the config file
--snn_threshold - SNN ratio threshold for SIFT correspondence filtering. Default: 0.80
--confidence - The RANSAC confidence. Default: 0.99
--inlier_threshold - The inlier-outlier threshold in pixels. Default: 15.0
--minimum_iterations - The maximum number of RANSAC iterations. Default: 50
--maximum_iterations - The maximum number of RANSAC iterations. Default: 1000
--use_magsac_plus_plus = { 0, 1 } - A boolean deciding if MAGSAC++ or MAGSAC should be used.
--sampler = { Uniform, PROSAC, PNAPSAC, Importance, ARSampler  } - The sampler used: Uniform, PROSAC [1], PNAPSAC [2], Importance [3], ARSampler [4]. Importance and ARSampler are initialized with the SNN ratios as inlier probabilities. Default: PROSAC
--core_number - The core number for the parallel processing. Default: 4
```

# Running homography estimation with Graph-Cut RANSAC

To build and install `GC-RANSAC`, clone the repository as:
```shell
$ git clone https://github.com/danini/graph-cut-ransac
```

Then it has to be installed by calling
```bash 
$ pip3 install -e .
```

The tests running GC-RANSAC can be started by calling
```
python test_gcransac.py --path=<path to the database folder>
```

The following additional parameters are accepted:
```
--split = { train, test } - Select the split to be used. Default: train
--scene = { all, NYC_Library, Alamo, Yorkminster, Tower_of_London, Madrid_Metropolis, Ellis_Island, Roman_Forum, Vienna_Cathedral, Piazza_del_Popolo, Union_Square } - Select the scene. Default: all
--config_path - The path to the config file
--snn_threshold - SNN ratio threshold for SIFT correspondence filtering. Default: 0.80
--confidence - The RANSAC confidence. Default: 0.99
--inlier_threshold - The inlier-outlier threshold in pixels. Default: 15.0
--minimum_iterations - The maximum number of RANSAC iterations. Default: 50
--maximum_iterations - The maximum number of RANSAC iterations. Default: 1000
--spatial_coherence_weight = A boolean deciding if MAGSAC++ or MAGSAC should be used.
--neighborhood_size = The radius of the hyper-sphere determining the neighborhood. Default: 20 px
--sampler = { Uniform, PROSAC, PNAPSAC, Importance, ARSampler  } - The sampler used: Uniform, PROSAC [1], PNAPSAC [2], Importance [3], ARSampler [4]. Importance and ARSampler are initialized with the SNN ratios as inlier probabilities. Default: PROSAC
--solver = { point, sift, affine } - The minimal solver used for homography estimation. "point": 4PC algorithm, "sift": 2SIFT estimator [5], "affine": 2AC estimator [6].
--core_number - The core number for the parallel processing. Default: 4
```

# Benchmarking results

Results with traditional methods with and without SNN filtering.

![results_traditional](assets/heb_benchmark_traditional.png)

Results with deep filtering.

![results_deep](assets/heb_benchmark_deep.png)

# Acknowledgements

When using the dataset, please cite

```
@inproceedings{HEB2023,
	author = {Daniel Barath, Dmytro Mishkin, Michal Polic, Wolfgang Förstner, Jiri Matas},
	title = {A Large Scale Homography Benchmark},
	booktitle = {Conference on Computer Vision and Pattern Recognition},
	year = {2023},
}

```

## References
<a id="1">[1]</a> 
Chum, Ondrej, and Jiri Matas. "Matching with PROSAC-progressive sample consensus." 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05). Vol. 1. IEEE, 2005.

<a id="2">[2]</a> 
Barath, Daniel, et al. "MAGSAC++, a fast, reliable and accurate robust estimator." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

<a id="3">[3]</a> 
Brachmann, Eric, and Carsten Rother. "Neural-guided RANSAC: Learning where to sample model hypotheses." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

<a id="4">[4]</a> 
Tong, Wei, Jiri Matas, and Daniel Barath. "Deep magsac++." arXiv preprint arXiv:2111.14093 (2021).

<a id="5">[5]</a> 
Barath, Daniel, and Zuzana Kukelova. "Homography from two orientation-and scale-covariant features." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

<a id="6">[6]</a> 
Barath, Daniel, and Levente Hajder. "A theory of point-wise homography estimation." Pattern Recognition Letters 94 (2017): 7-14.
