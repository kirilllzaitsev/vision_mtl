# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class ModelConfig:
    encoder_name: str = MISSING
    encoder_weights: str = "imagenet"


@dataclass
class BasicModelConfig(ModelConfig):
    encoder_name: str = "timm-mobilenetv3_large_100"


@dataclass
class cfg:
    model: ModelConfig
    debug: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=cfg)
cs.store(group="model", name="base_basic_model", node=BasicModelConfig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(conf: cfg) -> None:
    print(OmegaConf.to_yaml(conf))


if __name__ == "__main__":
    my_app()
