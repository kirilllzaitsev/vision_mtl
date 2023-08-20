import copy

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DepthModel(nn.Module):
    def __init__(
        self,
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        activation=None,
        segm_classes=19,
        decoder_first_channel=256,
        in_channels=3,
    ):
        super().__init__()

        decoder_channels = [decoder_first_channel]
        for i in range(1, 5):
            decoder_channels.append(decoder_first_channel // (2**i))

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=segm_classes,
            activation=activation,
            encoder_depth=len(decoder_channels),
            decoder_channels=decoder_channels,
        )

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.segm_head = model.segmentation_head
        self.depth_head = copy.deepcopy(model.segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        depth = self.depth_head(decoder_output)
        segm = self.segm_head(decoder_output)

        return dict(depth=depth, segm=segm)

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


if __name__ == "__main__":
    model = DepthModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
