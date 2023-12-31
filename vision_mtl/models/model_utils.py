import torch
import torch.nn.functional as F
from torch import nn


def concat_slightly_diff_sized_tensors(x1, x2):
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
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
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




def get_joint_layer_names(all_layer_names):
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


def get_joint_layer_names_before_stitch_for_unet(joint_layer_names):
    joint_layer_names_before_stitch = []
    for i, layer_name in enumerate(joint_layer_names):
        module_names_in_full_layer_name = layer_name.split(".")
        if (
            "encoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 5
        ) or (
            "decoder" in module_names_in_full_layer_name
            and len(module_names_in_full_layer_name) == 4
        ):
            joint_layer_names_before_stitch.append(layer_name)
    return joint_layer_names_before_stitch