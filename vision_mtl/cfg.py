import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from omegaconf import MISSING

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)


@dataclass
class ModelConfig:
    encoder_name: str = MISSING
    encoder_weights: str = "imagenet"


@dataclass
class BasicModelConfig(ModelConfig):
    encoder_name: str = "timm-mobilenetv3_large_100"


@dataclass
class LoggerConfig:
    api_key: str = os.environ["comet_api_key"]
    username: str = os.environ["comet_username"]
    project_name: str = "vision-mtl"


@dataclass
class DataConfig:
    data_dir: str = os.environ["data_base_dir"]
    benchmark_batch_path: str = f"{data_dir}/benchmark_batch.pt"
    num_classes: int = 19
    height: int = 128
    width: int = 256

    class_names: list = field(
        default_factory=lambda: [
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
    )

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    val_split: float = 0.1
    test_split: float = 0.1
    img_size: int = 224
    grayscale: bool = False
    normalize: bool = True
    augment: bool = True


@dataclass
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


@dataclass
class PipelineConfig:
    model: ModelConfig = BasicModelConfig()
    data: DataConfig = DataConfig()
    logger: LoggerConfig = LoggerConfig()
    vis: VisConfig = VisConfig()

    # device: str = "cpu"
    device: str = "cuda"

    debug: bool = False
    seed: int = 11

    log_root_dir = Path(__file__).parent.parent / "lightning_logs"


cfg = PipelineConfig()
