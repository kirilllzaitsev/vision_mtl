import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead


class Backbone(nn.Module):
    def __init__(
        self,
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        decoder_first_channel=256,
        num_decoder_layers=5,
        in_channels=3,
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

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        return decoder_output


class BasicMTLModel(nn.Module):
    def __init__(
        self,
        activation=None,
        segm_classes=19,
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        decoder_first_channel=256,
        num_decoder_layers=5,
        in_channels=3,
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

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        decoder_output = self.backbone(x)

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
    model = BasicMTLModel()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y["depth"].shape)
