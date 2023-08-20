import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import MISSING

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
# load_dotenv("/mnt/wext/projects/vision_mtl/vision_mtl/.env")


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


@dataclass
class DataConfig:
    data_dir: str = os.environ["data_base_dir"]
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
class Config:
    model: ModelConfig = BasicModelConfig()
    data: DataConfig = DataConfig()
    logger: LoggerConfig = LoggerConfig()
    debug: bool = False
    seed: int = 11


cfg = Config()
