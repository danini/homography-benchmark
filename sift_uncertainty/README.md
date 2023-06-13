# README.md

## Overview

This repository contains evaluation scripts to assess the uncertainty in SIFT (Scale-Invariant Feature Transform) transformations. The evaluations are carried out in three separate Jupyter notebooks:

1. **evaluate_angle_uncertainty.ipynb:** This notebook assesses the uncertainty in SIFT angle transformations.

2. **evaluate_scale_uncertainty.ipynb:** This notebook evaluates the uncertainty in SIFT scale transformations.

3. **evaluate_positional_uncertainty.ipynb:** This notebook evaluates the uncertainty in SIFT positional transformations.

The evaluations are performed using ten different scenes which need to be downloaded separately and placed into the `scenes` subdirectory.

## Getting Started

To get started, clone the repository and run the `init.sh` script which sets up the necessary environment and dependencies. Then, you can open the Jupyter notebooks in your web browser for further inspection and execution.

```bash
$ git clone https://github.com/danini/homography-benchmark.git
$ cd sift_uncertainty
$ ./init.sh
```

Then navigate to any of the three Jupyter notebooks (`evaluate_angle_uncertainty.ipynb`, `evaluate_scale_uncertainty.ipynb` or `evaluate_positional_uncertainty.ipynb`) in the Jupyter notebook interface and run the cells sequentially from top to bottom.

**Note:** Ensure that the scenes are downloaded and correctly placed into the `scenes` subdirectory before running the evaluations. The scenes can be obtained from [link](https://polybox.ethz.ch/index.php/s/R5sPelZ8688It92).

## Notebooks

1. **evaluate_angle_uncertainty.ipynb:** 

   This notebook assesses the transformation uncertainty in the angles of SIFT keypoints. It visualizes the angle errors in histogram form, providing insight into the nature and distribution of these errors across different scenes.

2. **evaluate_scale_uncertainty.ipynb:** 

   This notebook calculates the SIFT scale transformations, approximates the homography locally using an affinity matrix, decomposes the affinity matrix into scales and angles, and then calculates and filters the scale transformation errors. The results are displayed as histograms.

3. **evaluate_positional_uncertainty.ipynb:** 

   This notebook focuses on positional transformations in SIFT. It computes the squared mean reprojection error for each keypoint pair. The results are presented as a histogram and a heatmap, indicating the distribution of reprojection errors and how they vary across different scale clusters.

## Contributing

Contributions to improve the evaluation methods or to add more types of evaluations are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.

## Contact

For any issues or questions, please create an issue on GitHub or contact the michal.polic(at)cvut.cz directly.
