import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead

from vision_mtl.models.basic_model import Backbone
from vision_mtl.models.model_utils import (
    concat_slightly_diff_sized_tensors,
    get_joint_layer_names_before_stitch_for_unet,
)
from vision_mtl.utils import get_module_by_name


class CrossStitchLayer(nn.Module):
    def __init__(self, num_tasks, num_channels=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.channel_wise_stitching = num_channels is not None
        if self.channel_wise_stitching:
            self.weights = nn.Parameter(
                torch.Tensor(num_tasks, num_tasks, num_channels)
            )
        else:
            self.weights = nn.Parameter(torch.Tensor(num_tasks, num_tasks))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weights)

    def forward(self, mt_activations):
        if self.channel_wise_stitching:
            y = torch.einsum("aac,abcij->abcij", self.weights, mt_activations)
        else:
            y = torch.einsum("aa,abcij->abcij", self.weights, mt_activations)
        return y


class CSNet(nn.Module):
    def __init__(self, models: dict, channel_wise_stitching=False):
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
        self.models = nn.ModuleDict(models)
        # assuming all models have the same layers for simplicity
        random_model = self.models[self.model_names[0]]
        self.joint_layer_names = [x[0] for x in list(random_model.named_modules())[1:]]
        self.joint_layer_names_before_stitch = (
            get_joint_layer_names_before_stitch_for_unet(self.joint_layer_names)
        )
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
        self.valid_cross_stitch_layer_names = [
            layer_name.replace(".", "_")
            for layer_name in self.joint_layer_names_before_stitch
        ]
        self.true_cross_stitch_layer_names = [
            layer_name for layer_name in self.joint_layer_names_before_stitch
        ]
        if channel_wise_stitching:
            self.stitch_channels = self.get_stitch_channels(
                random_model, self.joint_layer_names_before_stitch
            )
            self.cross_stitch_layers = {
                layer_name: CrossStitchLayer(
                    num_tasks=self.num_tasks,
                    num_channels=self.stitch_channels[layer_idx],
                )
                for layer_idx, layer_name in enumerate(
                    self.valid_cross_stitch_layer_names
                )
            }
        else:
            self.cross_stitch_layers = {
                layer_name: CrossStitchLayer(num_tasks=self.num_tasks)
                for layer_name in self.valid_cross_stitch_layer_names
            }
        self.cross_stitch_layers = nn.ModuleDict(self.cross_stitch_layers)

    def parameters(self):
        params = []
        for task_name in self.model_names:
            params.extend(self.models[task_name].parameters())
        for layer_name in self.cross_stitch_layers.keys():
            params.extend(self.cross_stitch_layers[layer_name].parameters())
        return params

    def cuda(self):
        for task_name in self.model_names:
            self.models[task_name].cuda()
        for layer_name in self.cross_stitch_layers.keys():
            self.cross_stitch_layers[layer_name].cuda()
        return self

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            self.cuda()
        else:
            super().to(device)

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
                valid_layer_name = layer_name.replace(".", "_")
                cross_stitch = self.cross_stitch_layers[valid_layer_name]
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

    def consider_encoder_layer_at_idx(self, layer_idx):
        return (
            layer_idx != 0
            and layer_idx != self.num_encoder_layers - 1
            and layer_idx != self.num_decoder_layers - 1
        )

    def consider_decoder_layer_at_idx(self, layer_idx):
        return layer_idx != self.num_decoder_layers - 1

    def get_stitch_channels(self, random_model, joint_layer_names_before_stitch):
        stitch_channels = []
        encoder_channels = []
        for stitch_layer_name in joint_layer_names_before_stitch:
            named_modules = [x for x in list(random_model.named_modules())[1:]]
            for i, (layer_name, params) in enumerate(named_modules):
                if layer_name == stitch_layer_name:
                    last_conv_layer_idx = i - 1
                    # while "conv" not in named_modules[last_conv_layer_idx][0]:
                    while not isinstance(
                        named_modules[last_conv_layer_idx][1], nn.Conv2d
                    ):
                        last_conv_layer_idx -= 1
                    # print(f"{named_modules[last_conv_layer_idx][0]=}")
                    # print(layer_name, named_modules[last_conv_layer_idx][1].out_channels)
                    num_channels = named_modules[last_conv_layer_idx][1].out_channels
                    if "encoder" in layer_name:
                        layer_idx = int(
                            re.match(self.encoder_block_regex, layer_name).group(1)
                        )
                        if self.consider_encoder_layer_at_idx(layer_idx):
                            encoder_channels.append(num_channels)
                    if "decoder" in layer_name:
                        layer_idx = int(
                            re.match(self.decoder_block_regex, layer_name).group(1)
                        )
                        if self.consider_decoder_layer_at_idx(layer_idx):
                            num_channels += encoder_channels[-layer_idx - 1]
                    stitch_channels.append(num_channels)
                    # break
        return stitch_channels


def get_model_with_dense_preds(segm_classes=10, activation=None, backbone_params=None):
    backbone_params = backbone_params or {}
    backbone = Backbone(in_channels=3, **backbone_params)
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
    cs_model = CSNet(models, channel_wise_stitching=True)
    y_tasks = cs_model(x)
    print([y.shape for y in y_tasks.values()])
