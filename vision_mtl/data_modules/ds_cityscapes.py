import glob
import typing as t

import numpy as np
import torch

from vision_mtl.cfg import cityscapes_data_cfg as data_cfg
from vision_mtl.data_modules.common_ds import MTLDataset


class CityscapesDataset(MTLDataset):
    benchmark_idxs: list[int] = [955, 2279, 1878, 2325]

    def __init__(
        self,
        stage: str,
        data_base_dir: str = data_cfg.data_dir,
        transforms: t.Any = data_cfg.train_transform,
        max_depth: float = data_cfg.max_depth,
    ):
        super().__init__(
            stage=stage,
            data_base_dir=data_base_dir,
            max_depth=max_depth,
            train_transform=transforms,
            test_transform=transforms,
        )
        self.paths = self.parse_paths()

    def __len__(self) -> int:
        return len(self.paths["img"])

    def __getitem__(self, idx) -> dict:
        raw_sample = self.load_raw_sample(idx)
        sample = self.prepare_sample(raw_sample, self.transform)

        return sample

    def prepare_sample(self, raw_sample: dict, transforms: t.Any = None) -> dict:
        img, mask, depth = raw_sample["img"], raw_sample["mask"], raw_sample["depth"]

        mask[mask == -1] = data_cfg.num_classes - 1
        if transforms:
            transformed = transforms(image=img, mask=mask)
            transformed_depth = transforms(image=img, mask=depth)
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

        self.normalize_depth(depth)
        return {
            "img": img,
            "mask": mask,
            "depth": depth,
        }

    def load_raw_sample(self, idx):
        data_path, mask_path, depth_path = (
            self.paths["img"][idx],
            self.paths["mask"][idx],
            self.paths["depth"][idx],
        )
        img = np.load(data_path)
        assert img.max() <= 1.0
        mask = np.load(mask_path)
        depth = np.load(depth_path)
        return {
            "img": img,
            "mask": mask,
            "depth": depth,
        }

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
