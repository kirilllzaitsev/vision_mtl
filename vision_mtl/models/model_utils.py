import typing as t

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from segmentation_models_pytorch.base import SegmentationHead
from torch import nn


class Backbone(nn.Module):
    def __init__(
        self,
        encoder_name: str = "timm-mobilenetv3_large_100",
        encoder_weights: str = "imagenet",
        decoder_first_channel: int = 256,
        num_decoder_layers: int = 5,
        in_channels: int = 3,
    ):
        super().__init__()

        self.decoder_channels = [decoder_first_channel]
        for i in range(1, num_decoder_layers):
            self.decoder_channels.append(decoder_first_channel // (2**i))

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            encoder_depth=len(self.decoder_channels),
            decoder_channels=self.decoder_channels,
        )

        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        return decoder_output


def concat_slightly_diff_sized_tensors(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2. Based on https://github.com/milesial/Pytorch-UNet/tree/master."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: t.Optional[int] = None
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


def get_joint_layer_names(all_layer_names: list) -> list:
    """Fetches the layer names that should go into the stitching operation in the model. These are the layers at each level of the encoder and decoder, one for each level."""

    joint_layer_names = []
    for i, layer_name in enumerate(all_layer_names):
        module_names_in_full_layer_name = layer_name.split(".")
        if (
            "encoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 5
        ) or (
            "decoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 4
        ):
            joint_layer_names.append(layer_name)
    return joint_layer_names


def get_joint_layer_names_before_stitch_for_unet(joint_layer_names: list) -> list:
    """Fetches the layer names that should go prior to the stitch operation in the model."""

    joint_layer_names_before_stitch = []
    for i, layer_name in enumerate(joint_layer_names):
        module_names_in_full_layer_name = layer_name.split(".")
        if (
            "encoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 5
            and int(module_names_in_full_layer_name[-1]) != 0
        ) or (
            "decoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 4
        ):
            joint_layer_names_before_stitch.append(layer_name)
    return joint_layer_names_before_stitch


def get_model_with_dense_preds(
    segm_classes: int = 10,
    activation: t.Any = None,
    backbone_params: t.Optional[dict] = None,
) -> nn.Module:
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
