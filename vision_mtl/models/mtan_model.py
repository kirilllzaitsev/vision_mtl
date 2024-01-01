import logging

import torch
from torch import nn

from vision_mtl.models.model_utils import DoubleConv, concat_slightly_diff_sized_tensors

log = logging.getLogger(__name__)


class AttentionModuleEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        prev_layer_out_channels=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        self.is_first = prev_layer_out_channels is None

        # conv1 -> conv2 -> concat -> sigmoid -> conv3
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, conv1_shared, conv2_shared, prev_layer_out=None):
        if self.is_first:
            conv1 = conv1_shared
        else:
            # downsample prev_layer_out
            # prev_layer_out = self.maxpool(prev_layer_out)
            assert (
                prev_layer_out is not None
            ), "prev_layer_out must be provided for non-first AttentionModuleEncoder"
            conv1 = torch.cat((conv1_shared, prev_layer_out), dim=1)

        conv1 = self.conv1(conv1)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        conv1 = self.conv2(conv1)
        conv1 = self.bn2(conv1)
        conv1_attn = self.sigmoid(conv1)

        conv2 = conv2_shared * conv1_attn

        conv2 = self.conv3(conv2)
        conv2 = self.bn3(conv2)
        conv2 = self.relu2(conv2)

        conv2 = self.maxpool(conv2)

        return conv2


class AttentionModuleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=None, prev_layer_out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if prev_layer_out_channels is None:
            prev_layer_out_channels = in_channels

        # in_channels * 2 since we are concatenating conv1_shared and prev_layer_out put in advance to have the same in_channels dim of the conv1_shared
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        # converts prev attn layer out to in_channels to make the prev attn out maergeable with conv1_shared
        self.conv3 = nn.Conv2d(
            prev_layer_out_channels, in_channels, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=in_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, conv1_shared, prev_layer_out, conv2_shared):
        prev_layer_out = self.conv3(prev_layer_out)
        prev_layer_out = self.bn3(prev_layer_out)
        prev_layer_out = self.relu2(prev_layer_out)

        if conv1_shared.shape[2:] != prev_layer_out.shape[2:]:
            prev_layer_out = self.up(prev_layer_out)

        assert conv1_shared.shape[2:] == conv2_shared.shape[2:]

        # merge samp and prev_layer_out
        log.debug(
            f"{conv1_shared.shape=}\n{prev_layer_out.shape=}\n{conv2_shared.shape=}"
        )
        merged = torch.cat((conv1_shared, prev_layer_out), dim=1)

        conv1 = self.conv1(merged)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        conv1 = self.conv2(conv1)
        conv1 = self.bn2(conv1)
        conv1_attn = self.sigmoid(conv1)

        conv2_shared = conv2_shared * conv1_attn

        return conv2_shared


class MTANDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, task_attn_module: AttentionModuleEncoder
    ):
        super().__init__()
        self.dconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.task_attn_module = task_attn_module

    def forward(self, x, prev_layer_out=None):
        dconv_out = self.dconv(x)
        task_attn_out = self.task_attn_module(
            conv1_shared=x, conv2_shared=dconv_out, prev_layer_out=prev_layer_out
        )
        pool_out = self.pool(x)
        return pool_out, task_attn_out


class MTANUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        task_attn_module: AttentionModuleDecoder,
        bilinear=True,
    ):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        self.task_attn_module = task_attn_module

    def forward(self, x1, x2, task_attn_prev_out):
        x1 = self.up(x1)
        merged_enc_dec = concat_slightly_diff_sized_tensors(x1, x2)
        conv_out = self.conv(merged_enc_dec)
        log.debug(
            f"{merged_enc_dec.shape=}\n{task_attn_prev_out.shape=}\n{conv_out.shape=}"
        )
        # TODO: there should be many task_attn_outs, since each corresponds to a different task
        task_attn_out = self.task_attn_module(
            conv1_shared=merged_enc_dec,
            prev_layer_out=task_attn_prev_out,
            conv2_shared=conv_out,
        )
        return conv_out, task_attn_out


class MTANMiniUnet(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        self.in_hidden_channels = 128
        self.in_conv = DoubleConv(in_channels, self.in_hidden_channels)

        factor = 2 if bilinear else 1

        # global and local subnets are not related. the only connection between them is that local subnet needs to know dimensionality of conv1 and conv2. it defines its own output dims!
        global_subnet_enc_out_channels = [128, 256 // factor, 256 // factor]
        global_subnet_dec_out_channels = [128 // factor, 256, 256]
        global_subnet_enc_in_channels = [
            self.in_hidden_channels,
            global_subnet_enc_out_channels[0],
            global_subnet_enc_out_channels[1],
        ]
        global_subnet_dec_in_channels = [
            global_subnet_enc_out_channels[1] + global_subnet_enc_out_channels[2],
            global_subnet_enc_out_channels[0] + global_subnet_dec_out_channels[0],
            self.in_hidden_channels + global_subnet_dec_out_channels[1],
        ]

        task_subnet_out_channels_enc = [128, 128, 128]
        task_subnet_out_channels_dec = [128 // factor, 256, 256]
        task_attn_prev_layer_out_channels_enc = [None] + global_subnet_enc_out_channels[
            :-1
        ]
        task_attn_in_channels_enc = [
            self.in_hidden_channels,
            task_subnet_out_channels_enc[0] + global_subnet_enc_out_channels[0],
            task_subnet_out_channels_enc[1] + global_subnet_enc_out_channels[1],
        ]
        task_attn_in_channels_dec = [
            global_subnet_enc_out_channels[0] + global_subnet_enc_out_channels[1],
            global_subnet_enc_out_channels[1] + global_subnet_dec_out_channels[0],
            global_subnet_enc_out_channels[2] + global_subnet_dec_out_channels[1],
        ]

        task_attn_prev_layer_out_channels_dec = [
            task_subnet_out_channels_enc[-1]
        ] + task_subnet_out_channels_dec[:-1]

        task_attn_modules_enc = [
            AttentionModuleEncoder(
                in_channels=task_attn_in_channels_enc[i],
                out_channels=task_subnet_out_channels_enc[i],
                prev_layer_out_channels=task_attn_prev_layer_out_channels_enc[i],
            )
            for i in range(len(task_attn_in_channels_enc))
        ]
        task_attn_modules_dec = [
            AttentionModuleDecoder(
                in_channels=task_attn_in_channels_dec[i],
                out_channels=task_subnet_out_channels_dec[i],
                prev_layer_out_channels=task_attn_prev_layer_out_channels_dec[i],
            )
            for i in range(len(task_attn_in_channels_dec))
        ]

        self.enc_layers = nn.ModuleList(
            [
                MTANDown(
                    in_channels=global_subnet_enc_in_channels[i],
                    out_channels=global_subnet_enc_out_channels[i],
                    task_attn_module=task_attn_modules_enc[i],
                )
                for i in range(len(global_subnet_enc_in_channels))
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                MTANUp(
                    in_channels=global_subnet_dec_in_channels[i],
                    out_channels=global_subnet_dec_out_channels[i],
                    bilinear=bilinear,
                    task_attn_module=task_attn_modules_dec[i],
                )
                for i in range(len(global_subnet_dec_in_channels))
            ]
        )

        # supervision for the global net comes from the task heads
        self.task_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        task_attn_out_enc = None
        encoder = None
        decoder = None
        task_attn_out_dec = None
        encoder_features = [x1]
        for i in range(len(self.enc_layers)):
            if i == 0:
                encoder, task_attn_out_enc = self.enc_layers[i](x1)
            else:
                encoder, task_attn_out_enc = self.enc_layers[i](
                    encoder, task_attn_out_enc
                )
            encoder_features.append(encoder)
            log.debug(f"{encoder.shape=} {task_attn_out_enc.shape=}")
        for i in range(len(self.dec_layers)):
            if i == 0:
                decoder, task_attn_out_dec = self.dec_layers[i](
                    encoder_features[-1], encoder_features[-2], task_attn_out_enc
                )
            else:
                decoder, task_attn_out_dec = self.dec_layers[i](
                    decoder, encoder_features[-(i + 2)], task_attn_out_dec
                )
            log.debug(f"{decoder.shape=} {task_attn_out_dec.shape=}")

        return {
            "task_out": self.task_head(task_attn_out_dec),
        }


if __name__ == "__main__":
    mtan_mini_unet = MTANMiniUnet(in_channels=3)
    x = torch.randn(1, 3, 256, 256)
    y = mtan_mini_unet(x)
    log.debug(y["task_out"].shape)
