Multi-task learning in computer vision
==============================

This work explores some methodologies for multi-task learning (MTL) in the context of machine perception. The main goal is to study some of the models that represent different paradigms in MTL by implementing and evaluating them on established benchmarks.

## Models

Following the model taxonomy outlined in [1], the following models are implemented:

- hard parameter sharing, a baseline with optimal task weights in the loss function
- soft parameter sharing, **Cross-Stitch Networks** Misra, I., Shrivastava, A., Gupta, A., & Hebert, M.  [Cross-Stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539). CVPR, 2016.
- modulation & adapters, **[MTAN]** Liu, S., Johns, E., & Davison, A. J.  [End-to-End Multi-Task Learning with Attention](http://arxiv.org/abs/1803.10704). CVPR, 2019.

## Data

The models are trained and evaluated on the following datasets:

- [Cityscapes](https://www.cityscapes-dataset.com/) downloaded from [here](https://www.kaggle.com/datasets/sakshaymahna/cityscapes-depth-and-segmentation/data)
- [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) downloaded from [here](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

The chosen tasks are semantic segmentation and depth estimation. both tasks are formulated as a dense prediction problem, where the prediction is of the same spatial dimensions as the input image. Semantic segmentation is a pixel-wise classification problem, where each pixel is assigned a class label, while depth estimation is a pixel-wise regression problem, where each pixel is assigned a depth value. For the case of Cityscapes, the semantic segmentation task has 19 classes, while depths are represented as relative values ([inverse depth](https://robotics.stackexchange.com/questions/6334/what-is-inverse-depth-in-odometry-and-why-would-i-use-it)) in the range [0, 1].

## Pre-requisites

### Environment

- `python` >= 3.9
- `conda create -n mtl python=3.9`
- `pip install -r requirements.txt`

### Other

- (optional) Comet ML account for experiment tracking (see [here](https://www.comet.ml/docs/python-sdk/quickstart/)
- `vision_mtl/.env` file of the form:

```
data_base_dir=your_data_base_dir
comet_api_key=your_comet_api_key
comet_username=your_comet_username
```

## Imlementation details

### Training

The models are implemented in PyTorch. They are trained and evaluated on the task of semantic segmentation and depth estimation. A single GPU with 8Gb of memory is used for the experiments.

Entry point for the pipeline is `training_lit.py`. `utils.py` contains arguments for the command line.

To train a hard parameter sharing model ("basic") with pretrained weights on ImageNet, a sample command could be:

```
python training_lit.py --do_plot_preds --exp_tags test --num_workers=4 --batch_size=8 --val_epoch_freq=1 --save_epoch_freq=5 --num_epochs=20 --lr=5e-4 --model_name=basic --backbone_weights imagenet
```

For CSnet the `model_name` argument is `csnet`, while for MTAN it is `mtan`. These two models do not support pretrained weights and should be trained from scratch.

### Metrics

For semantic segmentation, the following metrics are used:

- accuracy, with `threshold=0.5`
- Jaccard index, with `threshold=0.5`
- F-beta score, with `beta=1, threshold=0.5, average='weighted'`

For depth estimation only mean absolute error (MAE) is used.

### Models

The models are aligned in terms of the number of parameters and amount to approximately 13.3M parameters.

The models are not tailored to match the performance claimed in the original papers, but rather to explore how different MTL paradigms perform on the chosen tasks under the same setup and without any tuning (except for the naive model).

#### Hard parameter sharing

A naive model that shares the parameters of the backbone between the tasks and has shallow one-layer heads for each task to get task-specific predictions.

The backbone is the `timm-mobilenetv3_large_100` model from the `segmentation_models_pytorch` library with **pretrained Imagenet weights**.

Files:

- `models/basic_model.py`

#### Soft parameter sharing

The model is based on the Cross-Stitch Networks architecture. It implements both channel-wise and layer-wise cross-stitching.

The idea of stitching is to allow task-specific networks to exchange information at different levels of the architecture. Layer-wise stitching defines a linear combination of the activations of the tasks at a given layer, while channel-wise stitching works at the more granular level of channels. Visual representation of the stitching idea is shown in [2].

Since the model requires access at the channel level for propagation, it is built with a custom UNet backbone inspired by [3].

Files:

- `models/cross_stitch_model.py`

#### Modulation & adapters

The model is based on the MTAN (Multi-Task Attention Network) architecture. Every task has its own subnetwork. It can be viewed as a task-specific feature extractor that receives as input weights from the main network which is a global feature extractor.

As the previous model, it is built with a custom UNet backbone. Please refer to [4] for the official implementation.

Files:

- `models/mtan_model.py`

## Results

Here I discuss the results and dive into the details of the implementation of the models.

Aggregated table of results:

| Metrics       |   HS_non_pretrained | HS_imagenet |   HS_imagenet_tuned |   CSnet |   MTAN |
|:--------------|-----------------------:|------:|--------------:|--------:|-------:|
| Loss          |                  4.549 | 3.402 |         3.42  |   5.596 |  3.639 |
| Accuracy      |                  0.805 | 0.857 |         0.856 |   0.774 |  0.86  |
| Jaccard index |                  0.288 | 0.368 |         0.367 |   0.253 |  0.396 |
| F-beta score   |                  0.793 | 0.851 |         0.849 |   0.759 |  0.855 |
| MAE           |                  0.043 | 0.045 |         0.038 |   0.058 |  0.06  |

where:

- HS_non_pretrained: hard parameter sharing with no pretrained weights
- HS_imagenet: hard parameter sharing with pretrained weights
- HS_imagenet_tuned: hard parameter sharing with pretrained weights and tuned hyperparameters
- CSnet: soft parameter sharing
- MTAN: modulation & adapters

## References

[1] ["Awesome" list for multi-task learning](https://github.com/Manchery/awesome-multi-task-learning)
[2] [Tensorflow implementation of the cross-stitch network](https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning)
[3] [UNet architecture implementation](https://github.com/milesial/Pytorch-UNet/tree/master)
[4] [Official implementation of MTAN](https://github.com/lorenmt/mtan)
