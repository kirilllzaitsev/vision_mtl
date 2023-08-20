import albumentations as A
import albumentations.pytorch as pytorch

from vision_mtl.cfg import cfg

norm_mean = (0.485, 0.456, 0.406)
norm_std = (0.229, 0.224, 0.225)
height, width = (cfg.data.height, cfg.data.width)

train_transform = A.Compose(
    [
        A.Resize(height=height, width=width),
        A.Normalize(mean=norm_mean, std=norm_std),
        pytorch.ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=height, width=width),
        A.Normalize(mean=norm_mean, std=norm_std),
        pytorch.ToTensorV2(),
    ]
)
