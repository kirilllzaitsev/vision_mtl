import typing as t

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead

from vision_mtl.models.model_utils import Backbone


class BasicMTLModel(nn.Module):
    def __init__(
        self,
        segm_classes: int,
        activation: t.Any = None,
        encoder_name: str = "timm-mobilenetv3_large_100",
        encoder_weights: str = "imagenet",
        decoder_first_channel: int = 256,
        num_decoder_layers: int = 5,
        in_channels: int = 3,
    ):
        super().__init__()

        self.backbone = Backbone(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_first_channel=decoder_first_channel,
            num_decoder_layers=num_decoder_layers,
            in_channels=in_channels,
        )
        self.segm_head = SegmentationHead(
            in_channels=self.backbone.decoder_channels[-1],
            out_channels=segm_classes,
            activation=activation,
            kernel_size=3,
        )
        self.depth_head = SegmentationHead(
            in_channels=self.backbone.decoder_channels[-1],
            out_channels=1,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        decoder_output = self.backbone(x)

        depth = self.depth_head(decoder_output)
        segm = self.segm_head(decoder_output)

        return dict(depth=depth, segm=segm)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            self.eval()

        out = self.forward(x)

        return out


if __name__ == "__main__":
    model = BasicMTLModel(19)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y["depth"].shape)
