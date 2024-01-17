Multi-task learning in computer vision
==============================

This work explores some methodologies for multi-task learning (MTL) in the context of machine perception. The main goal is to study some of the models that represent different paradigms in MTL by implementing and evaluating them on established benchmarks.

## Models

Following the model taxonomy outlined in [here](https://github.com/Manchery/awesome-multi-task-learning), the following models are implemented:

- hard parameter sharing, a baseline with optimal task weights in the loss function
- soft parameter sharing, **Cross-Stitch Networks** Misra, I., Shrivastava, A., Gupta, A., & Hebert, M.  [Cross-Stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539). CVPR, 2016.
- modulation & adapters, **[MTAN]** Liu, S., Johns, E., & Davison, A. J.  [End-to-End Multi-Task Learning with Attention](http://arxiv.org/abs/1803.10704). CVPR, 2019.

## Pre-requisites

### Environment

- `python` >= 3.8
- `pip install -r requirements.txt`

### Other

- (optional) Comet ML account for experiment tracking (see [here](https://www.comet.ml/docs/python-sdk/quickstart/)
- `vision_mtl/.env` file of the form:

```
data_base_dir=your_data_base_dir
comet_api_key=your_comet_api_key
comet_username=your_comet_username
```

## Data

The models are trained and evaluated on the following datasets:

- [Cityscapes](https://www.cityscapes-dataset.com/) downloaded from [here](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data)
- [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) downloaded from [here](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

The chosen tasks are semantic segmentation and depth estimation. both tasks are formulated as a dense prediction problem, where the prediction is of the same spatial dimensions as the input image. Semantic segmentation is a pixel-wise classification problem, where each pixel is assigned a class label, while depth estimation is a pixel-wise regression problem, where each pixel is assigned a depth value. For the case of Cityscapes, the semantic segmentation task has 19 classes, while depths are represented as relative values ([inverse depth](https://robotics.stackexchange.com/questions/6334/what-is-inverse-depth-in-odometry-and-why-would-i-use-it)) in the range [0, 1].

## Imlementation details

The models are implemented in PyTorch and tested on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. The models are trained and evaluated on the task of semantic segmentation and depth estimation. The models are trained on the training set of Cityscapes and evaluated on the validation set. The models are trained on a single GPU with 8Gb of memory.

Entry point for the pipeline is `training_lit.py`.

## Results

#TODO
