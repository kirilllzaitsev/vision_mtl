import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from vision_mtl.cfg import cfg


class PhotopicVisionDataset(Dataset):
    def __init__(
        self,
        stage: str,
        data_base_dir: str = cfg.data.data_dir,
        transforms: Any = None,
    ):
        self.data_base_dir = data_base_dir
        self.transforms = transforms
        self.stage = stage
        self.paths = self.parse_paths()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx) -> tuple:
        data_path, labels_paths = self.imgs[idx], self.labels[idx]
        mask_path, depth_path = labels_paths["mask"], labels_paths["depth"]
        image = cv2.imread(data_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            transformed_depth = self.transforms(image=image, mask=depth)
            image, mask, depth = (
                transformed["image"],
                transformed["mask"],
                transformed_depth["mask"],
            )
        return image, mask, depth

    def parse_paths(self) -> dict:
        dict_paths = {"image": [], "labels": {"mask": [], "depth": []}}
        base_dir = f"{self.data_base_dir}/{self.stage}"
        for dir_name, _, filenames in os.walk(base_dir):
            for filename in filenames:
                name = filename.split(".")[0]
                dict_paths["image"].append(f"{base_dir}/image/{name}.npy")
                dict_paths["labels"]["mask"].append(f"{base_dir}/label/{name}.npy")
                dict_paths["labels"]["depth"].append(f"{base_dir}/depth/{name}.npy")

        return dict_paths
