import logging

import torch
from torch import nn

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class AttentionModuleEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        prev_layer_out_channels=None,
        is_first=False,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        # if prev_layer_out_channels is None:
        #     prev_layer_out_channels = in_channels

        if is_first:
            assert (
                prev_layer_out_channels is None
            ), "prev_layer_out_channels must be None for the first AttentionModuleEncoder"
        self.is_first = is_first

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


x_enc1 = torch.randn(1, 128, 256, 256)
x_enc2 = torch.randn(1, 128, 256, 256)
enc = AttentionModuleEncoder(in_channels=128, is_first=True)
y = enc(conv1_shared=x_enc1, conv2_shared=x_enc2)
log.debug(y.shape)


class AttentionModuleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=None, prev_layer_out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if prev_layer_out_channels is None:
            prev_layer_out_channels = in_channels

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
        # downsample prev_layer_out
        # prev_layer_out = self.maxpool(prev_layer_out)

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


prev_attn_layer_out = torch.randn(1, 256, 128, 128)
x_dec1 = torch.randn(1, 256, 128, 128)
x_dec2 = torch.randn(1, 64, 128, 128)

# fake inputs:
# conv1_shared.shape = torch.Size([1, 256, 128, 128])
# prev_layer_out.shape = torch.Size([1, 256, 128, 128])
# conv2_shared.shape = torch.Size([1, 256, 128, 128])

# real inputs:
# conv1_shared.shape = torch.Size([1, 256, 128, 128])
# prev_layer_out.shape = torch.Size([1, 128, 64, 64])
# conv2_shared.shape = torch.Size([1, 64, 128, 128])

dec = AttentionModuleDecoder(in_channels=256, out_channels=64)
y = dec(conv1_shared=x_dec1, prev_layer_out=prev_attn_layer_out, conv2_shared=x_dec2)
log.debug(y.shape)

from vision_mtl.models.model_utils import DoubleConv, concat_slightly_diff_sized_tensors


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

        self.in_conv = DoubleConv(in_channels, 128)

        # enc = AttentionModuleEncoder(in_channels=128)
        # dec = AttentionModuleDecoder(in_channels=256)
        factor = 2 if bilinear else 1
        task_attn_modules_enc = [
            AttentionModuleEncoder(in_channels=128, out_channels=128, is_first=True),
            AttentionModuleEncoder(
                in_channels=128 + 128, out_channels=128, prev_layer_out_channels=128
            ),
            AttentionModuleEncoder(
                in_channels=128 + 128, out_channels=128, prev_layer_out_channels=128
            ),
        ]
        task_attn_modules_dec = [
            AttentionModuleDecoder(
                in_channels=256, out_channels=128 // factor, prev_layer_out_channels=128
            ),
            AttentionModuleDecoder(
                in_channels=128 // factor + 128,
                out_channels=256,
                prev_layer_out_channels=128 // factor,
            ),
            AttentionModuleDecoder(
                in_channels=256 + 128,
                out_channels=256,
                prev_layer_out_channels=256,
            ),
        ]

        # global and local subnets are not related. the only connection between them is that local subnet needs to know dimensionality of conv1 and conv2. it defines its own output dims!

        self.encoder1 = MTANDown(128, 128, task_attn_module=task_attn_modules_enc[0])
        self.encoder2 = MTANDown(
            128, 256 // factor, task_attn_module=task_attn_modules_enc[1]
        )
        self.encoder3 = MTANDown(
            128, 256 // factor, task_attn_module=task_attn_modules_enc[2]
        )

        self.decoder1 = MTANUp(
            256,
            128 // factor,
            bilinear=bilinear,
            task_attn_module=task_attn_modules_dec[0],
        )
        self.decoder2 = MTANUp(
            128 + 128 // factor,
            256,
            bilinear=bilinear,
            task_attn_module=task_attn_modules_dec[1],
        )
        self.decoder3 = MTANUp(
            256 + 128,
            256,
            bilinear=bilinear,
            task_attn_module=task_attn_modules_dec[2],
        )

        # supervision for the global net comes from the task heads
        # self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)
        self.task_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        encoder1, task_attn_out_enc1 = self.encoder1(x1)
        encoder2, task_attn_out_enc2 = self.encoder2(encoder1, task_attn_out_enc1)
        log.debug(f"{encoder1.shape=} {encoder2.shape=}")
        log.debug(f"{task_attn_out_enc1.shape=} {task_attn_out_enc2.shape=}")

        encoder3, task_attn_out_enc3 = self.encoder3(encoder2, task_attn_out_enc2)
        log.debug(f"{encoder3.shape=} {task_attn_out_enc3.shape=}")

        decoder1, task_attn_out_dec1 = self.decoder1(
            encoder3, encoder2, task_attn_out_enc3
        )
        decoder2, task_attn_out_dec2 = self.decoder2(
            decoder1, encoder1, task_attn_out_dec1
        )
        log.debug(f"{decoder2.shape=} {x1.shape=}")
        log.debug(f"{task_attn_out_dec2.shape=}")
        _, task_attn_out_dec3 = self.decoder3(
            x1=decoder2, x2=x1, task_attn_prev_out=task_attn_out_dec2
        )

        return {
            "task_out": self.task_head(task_attn_out_dec3),
        }


mtan_mini_unet = MTANMiniUnet(in_channels=3)
x = torch.randn(1, 3, 256, 256)
y = mtan_mini_unet(x)
log.debug(y["task_out"].shape)
