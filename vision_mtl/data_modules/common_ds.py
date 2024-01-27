import typing as t

import torch
from torch.utils.data import Dataset


class MTLDataset(Dataset):
    benchmark_idxs: t.Optional[list[int]] = None

    def __init__(
        self,
        stage: str,
        data_base_dir: str,
        max_depth: float,
        train_transform: t.Any = None,
        test_transform: t.Any = None,
    ):
        self.data_base_dir = data_base_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.stage = stage
        self.max_depth = max_depth
        self.transform = train_transform if stage == "train" else test_transform

    def load_raw_sample(self, idx):
        raise NotImplementedError

    def prepare_sample(self, raw_sample: dict, transforms: t.Any = None) -> dict:
        raise NotImplementedError

    def load_benchmark_batch(self) -> t.Optional[dict]:
        if self.benchmark_idxs is None:
            return None

        batch = {
            "img": [],
            "mask": [],
            "depth": [],
        }
        for idx in self.benchmark_idxs:
            raw_sample = self.load_raw_sample(idx)
            sample = self.prepare_sample(raw_sample, self.test_transform)
            for k in batch.keys():
                batch[k].append(sample[k])
        return {k: torch.stack(v, dim=0) for k, v in batch.items()}

    def normalize_depth(self, depth):
        if depth.max() > 1.0:
            depth /= self.max_depth
        return depth
