# Hierarchical visual localization

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

| Cirtorch/Colmap  | Day  | Night |
|------------------|------|-------|
|  High precision  | 76.3 | 19.4  |
| Medium precision | 83.7 | 28.6  |
| Coarse precision | 87.7 | 36.7  |

Command to reproduce result:
''' python evaluate.py --ratio_thresh 0.75 --n_neighbors 20 --global_method Cirtorch '''


| Cirtorch/Superpoint | Day  | Night |
|---------------------|------|-------|
| High precision      | 61.4 | 31.6  |
| Medium precision    | 77.3 | 49.0  |
| Coarse precision    | 88.6 | 61.2  |

''' python evaluate.py --ratio_thresh 0.75 --n_neighbors 20 --global_method Cirtorch --local_method Superpoint '''

Benchmark results are available [here](https://www.visuallocalization.net/benchmark/)
