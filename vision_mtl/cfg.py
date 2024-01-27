"""Contains utility classes to control the pipeline."""

import os
import typing as t
from pathlib import Path

import albumentations as A
import albumentations.pytorch as pytorch
import numpy as np
from dotenv import load_dotenv
from omegaconf import MISSING
from torchvision import transforms

# sensitive data goes to the .env file not shared in the repository
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

root_dir = Path(__file__).parent


class ModelConfig:
    encoder_name: str = MISSING
    encoder_weights: str = "imagenet"


class BasicModelConfig(ModelConfig):
    encoder_name: str = "timm-mobilenetv3_large_100"


class LoggerConfig:
    api_key: str = os.environ["comet_api_key"]
    username: str = os.environ["comet_username"]
    project_name: str = "vision-mtl"


class DataConfig:
    dataset_name: str

    data_dir: str = str(root_dir / "data")
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    shuffle_train: bool = True
    train_size: float = 0.8

    height: int
    width: int

    # segmentation
    num_classes: int
    class_names: list

    # depth estimation
    max_depth: float

    # transforms
    train_transform: t.Union[A.Compose, transforms.Compose]
    test_transform: t.Union[A.Compose, transforms.Compose]


class CityscapesConfig(DataConfig):
    dataset_name: str = "cityscapes"
    data_dir: str = f"{DataConfig.data_dir}/{dataset_name}"
    benchmark_batch_path: str = f"{data_dir}/benchmark_batch.pt"

    height: int = 128
    width: int = 256

    num_classes: int = 19
    class_names: list = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "artifact",
    ]

    max_depth: float = 1.0

    batch_size: int = 8
    num_workers: int = 4

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    train_transform: A.Compose = A.Compose(
        [
            A.Resize(height=height, width=width),
            pytorch.ToTensorV2(),
        ]
    )
    test_transform: A.Compose = A.Compose(
        [
            A.Resize(height=height, width=width),
            pytorch.ToTensorV2(),
        ]
    )


class NYUv2Config(DataConfig):
    dataset_name: str = "nyuv2"
    data_dir: str = f"{DataConfig.data_dir}/{dataset_name}"

    height: int = 480
    width: int = 640

    num_classes: int = 13
    class_names: list = [
        "background",
        "bed",
        "books",
        "ceiling",
        "chair",
        "floor",
        "furniture",
        "objects",
        "painting",
        "sofa",
        "table",
        "tv",
        "wall",
        "window",
    ]

    max_depth: float = 10.0

    train_transform: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_transform: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


class VisConfig:
    rgb_palette: np.ndarray = np.array(
        [
            [63, 171, 212],
            [109, 137, 117],
            [24, 83, 42],
            [148, 77, 185],
            [122, 139, 58],
            [32, 126, 85],
            [17, 164, 215],
            [124, 39, 146],
            [161, 239, 20],
            [40, 81, 119],
            [149, 34, 38],
            [166, 224, 205],
            [134, 100, 230],
            [123, 157, 137],
            [11, 5, 225],
            [60, 84, 80],
            [173, 186, 12],
            [199, 91, 22],
            [170, 124, 184],
            [119, 102, 69],
        ]
    )


class PipelineConfig:
    model: ModelConfig = BasicModelConfig()
    logger: LoggerConfig = LoggerConfig()
    vis: VisConfig = VisConfig()
    data: DataConfig = DataConfig()

    device: str = "cuda"

    debug: bool = False
    seed: int = 11

    log_root_dir = Path(__file__).parent.parent / "lightning_logs"

    def load_data_cfg(self, ds_name: str) -> None:
        if ds_name == "cityscapes":
            self.data = CityscapesConfig()
        elif ds_name == "nyuv2":
            self.data = NYUv2Config()
        else:
            raise ValueError(
                f"Unknown dataset name: {ds_name}. Please use one of: cityscapes, nyuv2"
            )


cityscapes_data_cfg = CityscapesConfig()
nyuv2_data_cfg = NYUv2Config()

cfg = PipelineConfig()
