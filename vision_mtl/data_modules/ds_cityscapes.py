import os
import typing as t

import numpy as np
import torch
from torch.utils.data import Dataset

from vision_mtl.cfg import cityscapes_data_cfg as data_cfg


class CityscapesDataset(Dataset):
    def __init__(
        self,
        stage: str,
        data_base_dir: str = data_cfg.data_dir,
        transforms: t.Any = data_cfg.train_transform,
        max_depth: float = data_cfg.max_depth,
    ):
        self.data_base_dir = data_base_dir
        self.transforms = transforms
        self.stage = stage
        self.paths = self.parse_paths()
        self.max_depth = max_depth

    def __len__(self) -> int:
        return len(self.paths["img"])

    def __getitem__(self, idx) -> dict:
        data_path, mask_path, depth_path = (
            self.paths["img"][idx],
            self.paths["mask"][idx],
            self.paths["depth"][idx],
        )
        img = np.load(data_path)
        assert img.max() <= 1.0
        mask = np.load(mask_path)
        mask[mask == -1] = data_cfg.num_classes - 1
        depth = np.load(depth_path)
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            transformed_depth = self.transforms(image=img, mask=depth)
            img, mask, depth = (
                transformed["image"],
                transformed["mask"],
                transformed_depth["mask"],
            )

            mask = mask.long()
        else:
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            depth = torch.from_numpy(depth)

        img = img.float()
        mask = mask.long()
        depth = depth.float()

        # normalize depth
        if depth.max() > 1.0:
            depth /= self.max_depth

        sample = {"img": img, "mask": mask, "depth": depth}
        return sample

    def parse_paths(self) -> dict:
        base_dir = f"{self.data_base_dir}/{self.stage}"
        dir_name_to_key = {
            "image": "img",
            "label": "mask",
            "depth": "depth",
        }
        dict_paths = {v: [] for v in dir_name_to_key.values()}
        for k, v in dir_name_to_key.items():
            filenames = sorted(glob.glob(f"{base_dir}/{k}/*.npy"))
            for filename in filenames:
                dict_paths[v].append(filename)

        assert (
            len(dict_paths["img"])
            == len(dict_paths["mask"])
            == len(dict_paths["depth"])
        )

        return dict_paths
