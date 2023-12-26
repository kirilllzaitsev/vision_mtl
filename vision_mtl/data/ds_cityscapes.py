import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vision_mtl.cfg import cfg


class CityscapesDataset(Dataset):
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
        return len(self.paths)

    def __getitem__(self, idx) -> dict:
        data_path, mask_path, depth_path = (
            self.paths["img"][idx],
            self.paths["mask"][idx],
            self.paths["depth"][idx],
        )
        img = np.load(data_path)
        assert img.max() <= 1.0
        mask = np.load(mask_path)
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

        sample = {"img": img, "mask": mask, "depth": depth}
        return sample

    def parse_paths(self) -> dict:
        dict_paths = {"img": [], "mask": [], "depth": []}
        base_dir = f"{self.data_base_dir}/{self.stage}"
        for dir_name, _, filenames in os.walk(base_dir):
            for filename in filenames:
                name = filename.split(".")[0]
                dict_paths["img"].append(f"{base_dir}/image/{name}.npy")
                dict_paths["mask"].append(f"{base_dir}/label/{name}.npy")
                dict_paths["depth"].append(f"{base_dir}/depth/{name}.npy")

        return dict_paths
