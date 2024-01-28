import logging
import typing as t

import torch
from torch import nn

from vision_mtl.utils.model_utils import DoubleConv, concat_slightly_diff_sized_tensors

log = logging.getLogger(__name__)


class AttentionModuleEncoder(nn.Module):
    def __init__(
        self,
        shared_1_channels: int,
        out_channels: int,
        shared_2_channels: int,
        prev_layer_out_channels: t.Optional[int] = None,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.is_first = prev_layer_out_channels is None
        self.prev_layer_out_channels = prev_layer_out_channels or 0
        self.shared_1_channels = shared_1_channels
        self.out_channels = out_channels
        self.shared_2_channels = shared_2_channels
        self.hidden_channels = hidden_channels

        # conv1 -> conv2 -> concat -> sigmoid -> conv3
        self.conv1 = nn.Conv2d(
            self.shared_1_channels + self.prev_layer_out_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_channels, self.shared_2_channels, kernel_size=1, padding=0
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.shared_2_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(
            self.shared_2_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(
        self,
        conv1_shared: torch.Tensor,
        conv2_shared: torch.Tensor,
        prev_layer_outs: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_first:
            conv1 = conv1_shared
        else:
            # downsample prev_layer_outs
            # prev_layer_outs = self.maxpool(prev_layer_outs)
            assert (
                prev_layer_outs is not None
            ), "prev_layer_outs must be provided for non-first AttentionModuleEncoder"
            conv1 = torch.cat((conv1_shared, prev_layer_outs), dim=1)

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
    def __init__(
        self,
        shared_1_channels: int,
        shared_2_channels: int,
        prev_layer_out_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.shared_1_channels = shared_1_channels
        self.shared_2_channels = shared_2_channels
        self.prev_layer_out_channels = prev_layer_out_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # in_channels * 2 since we are concatenating conv1_shared and prev_layer_outs put in advance
        # to have the same in_channels dim of the conv1_shared
        self.conv1 = nn.Conv2d(
            shared_1_channels + hidden_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(num_features=hidden_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            hidden_channels, shared_2_channels, kernel_size=1, padding=0
        )
        self.bn2 = nn.BatchNorm2d(num_features=shared_2_channels)
        self.sigmoid = nn.Sigmoid()
        # converts prev attn layer out to in_channels to make the prev attn out maergeable with conv1_shared
        self.conv3 = nn.Conv2d(
            prev_layer_out_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=hidden_channels)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_out = nn.Conv2d(
            shared_2_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn_out = nn.BatchNorm2d(num_features=out_channels)
        self.relu_out = nn.ReLU()

    def forward(
        self,
        conv1_shared: torch.Tensor,
        prev_layer_outs: torch.Tensor,
        conv2_shared: torch.Tensor,
    ) -> torch.Tensor:
        prev_layer_outs = self.conv3(prev_layer_outs)
        prev_layer_outs = self.bn3(prev_layer_outs)
        prev_layer_outs = self.relu2(prev_layer_outs)

        if conv1_shared.shape[2:] != prev_layer_outs.shape[2:]:
            prev_layer_outs = self.up(prev_layer_outs)

        assert conv1_shared.shape[2:] == conv2_shared.shape[2:]

        # merge samp and prev_layer_outs
        log.debug(
            f"{conv1_shared.shape=}\n{prev_layer_outs.shape=}\n{conv2_shared.shape=}"
        )
        merged = torch.cat((conv1_shared, prev_layer_outs), dim=1)

        conv1 = self.conv1(merged)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        conv1 = self.conv2(conv1)
        conv1 = self.bn2(conv1)
        conv1_attn = self.sigmoid(conv1)

        conv2_shared = conv2_shared * conv1_attn

        # conv2_shared is at full scale. should apply extra conv to make the task decoder more lightweight
        conv2_shared = self.conv_out(conv2_shared)
        conv2_shared = self.bn_out(conv2_shared)
        conv2_shared = self.relu_out(conv2_shared)

        return conv2_shared


class MTANDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_attn_modules: list[AttentionModuleEncoder],
        apply_pool: bool = True,
    ):
        super().__init__()
        self.dconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2) if apply_pool else nn.Identity()
        self.task_attn_modules = task_attn_modules

    def forward(
        self, x: torch.Tensor, prev_layer_outs: t.Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        dconv_out = self.dconv(x)
        task_attn_outs = []
        for i, task_attn_module in enumerate(self.task_attn_modules):
            task_attn_outs.append(
                task_attn_module(
                    conv1_shared=x,
                    conv2_shared=dconv_out,
                    prev_layer_outs=prev_layer_outs[i] if prev_layer_outs else None,
                )
            )
        pool_out = self.pool(dconv_out)
        return pool_out, task_attn_outs


class MTANUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        task_attn_modules: list[AttentionModuleDecoder],
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
        self.task_attn_modules = task_attn_modules
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        task_attn_prev_outs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x1 = self.up(x1)
        merged_enc_dec = concat_slightly_diff_sized_tensors(x1, x2)
        conv_out = self.conv(merged_enc_dec)
        log.debug(
            f"{merged_enc_dec.shape=}\n{task_attn_prev_outs[0].shape=}\n{conv_out.shape=}"
        )
        task_attn_outs = []
        for i, task_attn_module in enumerate(self.task_attn_modules):
            task_attn_outs.append(
                task_attn_module(
                    conv1_shared=merged_enc_dec,
                    prev_layer_outs=task_attn_prev_outs[i],
                    conv2_shared=conv_out,
                )
            )
        return conv_out, task_attn_outs


class MTANMiniUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        map_tasks_to_num_channels: dict[str, int],
        task_subnets_hidden_channels: int = 128,
        encoder_first_channel: int = 64,
        encoder_num_channels: int = 4,
    ):
        super().__init__()

        self.num_tasks = len(map_tasks_to_num_channels)
        self.in_channels = in_channels

        # global and local subnets are not related. the only connection between them is that local subnet needs to
        # know the dimensionality of conv1 and conv2. The local subnet defines its own output dims!
        self.global_subnet_enc_out_channels = [
            encoder_first_channel * (2**i) for i in range(encoder_num_channels)
        ]
        self.global_subnet_enc_in_channels = [
            self.in_channels
        ] + self.global_subnet_enc_out_channels[:-1]

        # dec_0 is at the bottleneck of the global subnet
        self.global_subnet_dec_out_channels = self.global_subnet_enc_out_channels[::-1]

        self.bottleneck = DoubleConv(
            in_channels=self.global_subnet_enc_out_channels[-1],
            out_channels=self.global_subnet_enc_out_channels[-1] * 2,
        )

        self.global_subnet_dec_in_channels = [
            self.global_subnet_enc_out_channels[-1] * 2
        ] + self.global_subnet_dec_out_channels[:-1]

        self.task_attn_out_channels_enc = [
            x for x in self.global_subnet_enc_out_channels
        ]
        self.task_attn_prev_layer_out_channels_enc = [
            None
        ] + self.task_attn_out_channels_enc[:-1]
        self.task_attn_in_channels_enc = [
            self.in_channels
        ] + self.global_subnet_enc_out_channels[:-1]

        self.task_subnet_out_channels_dec = [
            x for x in self.global_subnet_dec_out_channels
        ]
        self.task_attn_in_channels_dec = [
            self.global_subnet_enc_out_channels[-1] * 2,
        ] + self.global_subnet_dec_out_channels[:-1]

        self.task_attn_prev_layer_out_channels_dec = [
            self.task_attn_out_channels_enc[-1]
        ] + self.task_subnet_out_channels_dec[:-1]

        task_attn_modules_enc = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        AttentionModuleEncoder(
                            shared_1_channels=self.task_attn_in_channels_enc[i],
                            shared_2_channels=self.global_subnet_enc_out_channels[i],
                            out_channels=self.task_attn_out_channels_enc[i],
                            prev_layer_out_channels=self.task_attn_prev_layer_out_channels_enc[
                                i
                            ],
                            hidden_channels=task_subnets_hidden_channels,
                        )
                        for _ in range(self.num_tasks)
                    ]
                )
                for i in range(len(self.task_attn_in_channels_enc))
            ]
        )
        task_attn_modules_dec = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        AttentionModuleDecoder(
                            shared_1_channels=self.task_attn_in_channels_dec[i],
                            shared_2_channels=self.global_subnet_dec_out_channels[i],
                            prev_layer_out_channels=self.task_attn_prev_layer_out_channels_dec[
                                i
                            ],
                            out_channels=self.task_subnet_out_channels_dec[i],
                            hidden_channels=task_subnets_hidden_channels,
                        )
                        for _ in range(self.num_tasks)
                    ]
                )
                for i in range(len(self.task_attn_in_channels_dec))
            ]
        )

        self.enc_layers = nn.ModuleList(
            [
                MTANDown(
                    in_channels=self.global_subnet_enc_in_channels[i],
                    out_channels=self.global_subnet_enc_out_channels[i],
                    task_attn_modules=task_attn_modules_enc[i],
                    apply_pool=False,
                )
                for i in range(len(self.global_subnet_enc_in_channels))
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                MTANUp(
                    in_channels=self.global_subnet_dec_in_channels[i],
                    out_channels=self.global_subnet_dec_out_channels[i],
                    task_attn_modules=task_attn_modules_dec[i],
                )
                for i in range(len(self.global_subnet_dec_in_channels))
            ]
        )

        self.pool = nn.MaxPool2d(2)

        # supervision for the global net comes from the task heads
        self.map_tasks_to_heads = nn.ModuleDict(
            {
                task_name: nn.Conv2d(
                    self.task_subnet_out_channels_dec[-1],
                    out_channels,
                    kernel_size=1,
                )
                for task_name, out_channels in map_tasks_to_num_channels.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        task_attn_outs_enc = None
        encoder = x
        encoder_features = []
        for i in range(len(self.enc_layers)):
            encoder, task_attn_outs_enc = self.enc_layers[i](
                encoder, task_attn_outs_enc
            )
            log.debug(f"{encoder.shape=} {task_attn_outs_enc[0].shape=}")
            encoder_features.append(encoder)
            encoder = self.pool(encoder)

        bottleneck = self.bottleneck(encoder)
        decoder = bottleneck
        task_attn_outs_dec = task_attn_outs_enc

        for i in range(len(self.dec_layers)):
            # encoder features at idxs -1 and -2 are considered for the first decoder layer with idx 0
            decoder, task_attn_outs_dec = self.dec_layers[i](
                decoder, encoder_features[-(i + 1)], task_attn_outs_dec
            )
            log.debug(f"{decoder.shape=} {task_attn_outs_dec[0].shape=}")

        return {
            task_name: head(task_attn_outs_dec[i])
            for i, (task_name, head) in enumerate(self.map_tasks_to_heads.items())
        }


if __name__ == "__main__":
    num_tasks = 2
    map_tasks_to_heads = {
        f"task{i}": nn.Conv2d(
            256,
            1,
            kernel_size=1,
        )
        for i in range(num_tasks)
    }
    mtan_mini_unet = MTANMiniUnet(
        in_channels=3, map_tasks_to_num_channels=map_tasks_to_heads
    )
    x = torch.randn(1, 3, 256, 256)
    y = mtan_mini_unet(x)
    print(y["task0"].shape)
