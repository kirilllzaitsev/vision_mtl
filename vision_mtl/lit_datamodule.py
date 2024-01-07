from typing import Union

import albumentations as A
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from vision_mtl.data_modules.ds_cityscapes import CityscapesDataset
from vision_mtl.data_modules.transforms import test_transform, train_transform


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_base_dir: str,
        train_transform: Union[T.Compose, A.Compose] = train_transform,
        test_transform: Union[T.Compose, A.Compose] = test_transform,
        train_size: float = 0.8,
        batch_size: int = 4,
        num_workers: int = 0,
        do_overfit: bool = False,
    ):
        super().__init__()
        self.data_base_dir = data_base_dir
        self.do_overfit = do_overfit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.data_train: CityscapesDataset = None
        self.data_val: CityscapesDataset = None
        self.data_test: CityscapesDataset = None
        self.data_predict: CityscapesDataset = None
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.save_hyperparameters()

    def setup(self, stage: str = None) -> None:
        data_train = CityscapesDataset(
            stage="train",
            data_base_dir=self.data_base_dir,
            transforms=self.train_transform,
        )
        self.benchmark_batch = data_train.load_benchmark_batch()
        if stage == "fit" or stage is None:
            if self.do_overfit:
                self.data_val = self.data_train = torch.utils.data.Subset(
                    data_train,
                    range(0, self.batch_size),
                )
            else:
                train_len = int(len(data_train) * self.train_size)

                self.data_train, self.data_val = torch.utils.data.random_split(
                    data_train,
                    [
                        train_len,
                        len(data_train) - train_len,
                    ],
                )
                self.data_val.transforms = self.test_transform
        if stage == "test" or stage is None:
            if self.do_overfit:
                self.data_test = self.data_train
            else:
                self.data_test = CityscapesDataset(
                    stage="val",
                    data_base_dir=self.data_base_dir,
                    transforms=self.test_transform,
                )
        if stage == "predict" or stage is None:
            if self.do_overfit:
                self.data_predict = self.data_train
            else:
                self.data_predict = CityscapesDataset(
                    stage="val",
                    data_base_dir=self.data_base_dir,
                    transforms=self.test_transform,
                )

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
