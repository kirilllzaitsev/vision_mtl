import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead

from vision_mtl.models.basic_model import Backbone
from vision_mtl.models.model_utils import concat_slightly_diff_sized_tensors, get_joint_layer_names_before_stitch_for_unet
from vision_mtl.utils import get_module_by_name


class CrossStitchLayer(nn.Module):
    def __init__(self, num_tasks):
        super(CrossStitchLayer, self).__init__()
        self.num_tasks = num_tasks
        self.weights = nn.Parameter(torch.Tensor(num_tasks, num_tasks))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.weights)

    def forward(self, mt_activations):
        y = torch.einsum("aa,abcij->abcij", self.weights, mt_activations)
        return y


class CSNet(nn.Module):
    def __init__(self, models: dict):
        """A meta-network that stitches together multiple models using cross-stitch units
        as means of sharing information in the multi-task setting.
        Args:
            models: a map task_name -> model instance
            num_classes (int): number of classes for the final classification layer"""
        super().__init__()
        self.encoder_block_regex = r"0.encoder.model.blocks.(\d+)$"
        self.decoder_block_regex = r"0.decoder.blocks.(\d+)$"
        self.num_tasks = len(models)
        # self.models = nn.ModuleList([BasicCNN(num_classes=num_classes) for _ in range(3)])
        self.model_names = list(models.keys())
        self.models = models
        # assuming all models have the same layers for simplicity
        random_model = self.models[self.model_names[0]]
        self.joint_layer_names = [x[0] for x in list(random_model.named_modules())[1:]]
        self.joint_layer_names_before_stitch = (
            get_joint_layer_names_before_stitch_for_unet(self.joint_layer_names)
        )
        self.cross_stitch_layers = {
            layer_name: CrossStitchLayer(num_tasks=self.num_tasks)
            for layer_name in self.joint_layer_names_before_stitch
        }
        self.num_encoder_layers = len(
            [
                x
                for x in get_module_by_name(
                    random_model, "0.encoder.model.blocks"
                ).named_children()
            ]
        )
        self.num_decoder_layers = len(
            [
                x
                for x in get_module_by_name(
                    random_model, "0.decoder.blocks"
                ).named_children()
            ]
        )

    def parameters(self):
        params = []
        for task_name in self.model_names:
            params.extend(self.models[task_name].parameters())
        for layer_name in self.joint_layer_names_before_stitch:
            params.extend(self.cross_stitch_layers[layer_name].parameters())
        return params

    def cuda(self):
        for task_name in self.model_names:
            self.models[task_name].cuda()
        for layer_name in self.joint_layer_names_before_stitch:
            self.cross_stitch_layers[layer_name].cuda()
        return self

    def forward(self, x):
        model_features = {task_name: x.clone() for task_name in self.model_names}
        layers_applied = []
        encoder_features = {task_name: [] for task_name in self.model_names}
        for layer_idx, layer_name in enumerate(self.joint_layer_names):
            for task_name, model in self.models.items():
                layer = get_module_by_name(model, layer_name)
                match = re.match(self.encoder_block_regex, layer_name)
                if match:
                    # save the latest features of an encoder layer
                    # ignore the first and the last features
                    layer_idx = int(match.group(1))
                    if layer_idx != 0 and layer_idx != self.num_encoder_layers - 1:
                        if layer_idx != self.num_decoder_layers - 1:
                            encoder_features[task_name].append(
                                model_features[task_name].clone()
                            )
                match = re.match(self.decoder_block_regex, layer_name)
                if match:
                    # save the latest features of an encoder layer
                    # ignore the first and the last features
                    layer_idx = int(match.group(1))
                    if layer_idx != self.num_decoder_layers - 1:
                        model_features[task_name] = concat_slightly_diff_sized_tensors(
                            model_features[task_name],
                            encoder_features[task_name][-layer_idx - 1],
                        )
                    else:
                        model_features[task_name] = F.interpolate(
                            model_features[task_name], scale_factor=2, mode="nearest"
                        )

                has_children = any(
                    child for child_name, child in layer.named_children()
                )
                if has_children:
                    continue
                model_features[task_name] = layer(model_features[task_name])
                layers_applied.append(layer_name)
            if layer_name in self.joint_layer_names_before_stitch:
                # continue
                cross_stitch = self.cross_stitch_layers[layer_name]
                model_features = cross_stitch(
                    torch.stack(
                        [model_features[task_name] for task_name in self.model_names],
                        dim=0,
                    )
                )
                model_features = {
                    task_name: model_features[idx]
                    for (idx, task_name) in enumerate(self.model_names)
                }
        return model_features


def get_model_with_dense_preds(segm_classes=10, activation=None):
    backbone = Backbone(in_channels=3)
    head = SegmentationHead(
        in_channels=backbone.decoder_channels[-1],
        out_channels=segm_classes,
        activation=activation,
        kernel_size=3,
    )
    model = nn.Sequential(backbone, head)
    return model


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    num_classes = 10
    models = {
        "task1": get_model_with_dense_preds(segm_classes=1, activation=None),
        "task2": get_model_with_dense_preds(segm_classes=1, activation=None),
        "task3": get_model_with_dense_preds(segm_classes=1, activation=None),
    }
    cs_model = CSNet(models)
    y_tasks = cs_model(x)
    print([y.shape for y in y_tasks.values()])
