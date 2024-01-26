import albumentations as A
import albumentations.pytorch as pytorch
from torchvision import transforms

from vision_mtl.cfg import cfg

norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)
height, width = (cfg.data.height, cfg.data.width)

cityscapes_train_transform = A.Compose(
    [
        A.Resize(height=height, width=width),
        # A.Normalize(mean=norm_mean, std=norm_std),
        pytorch.ToTensorV2(),
    ]
)

cityscapes_test_transform = A.Compose(
    [
        A.Resize(height=height, width=width),
        # A.Normalize(mean=norm_mean, std=norm_std),
        pytorch.ToTensorV2(),
    ]
)

nyuv2_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
