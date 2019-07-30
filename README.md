# Hierarchical visual localization
Visual localization pipeline using following steps
1) Find similar database images by using global descriptors
2) Extract local descriptors from database and query image
3) Match local descriptors
4) Calculate 6-DoF pose using RANSAC scheme

## Credits
### Idea
Hierarchical localization was introduced in this [paper](https://arxiv.org/abs/1812.03506)

### Code
We reused code from the following repositories.
- [HF-Net](https://www.github.com/ethz-asl/hfnet)
- [Pytorch NetVLAD](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html)
- [CNN Image Retrieval](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
- [Superpoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)

## Current performance
Evaluation via [online evaluation system](https://www.visuallocalization.net)  
[Benchmark results](https://www.visuallocalization.net/benchmark/)  

| Cirtorch/Colmap  | Day  | Night |
|------------------|------|-------|
|  High precision  | 76.3 | 19.4  |
| Medium precision | 83.7 | 28.6  |
| Coarse precision | 87.7 | 36.7  |

Command to reproduce result:  
``` python evaluate.py --ratio_thresh 0.75 --n_neighbors 20 --global_method Cirtorch ```


| Cirtorch/Superpoint | Day  | Night |
|---------------------|------|-------|
| High precision      | 61.4 | 31.6  |
| Medium precision    | 77.3 | 49.0  |
| Coarse precision    | 88.6 | 61.2  |

``` python evaluate.py --ratio_thresh 0.75 --n_neighbors 20 --global_method Cirtorch --local_method Superpoint ```

