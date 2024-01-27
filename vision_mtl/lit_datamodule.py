import typing as t
from typing import Union

import albumentations as A
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from vision_mtl.cfg import cfg
from vision_mtl.data_modules.ds_cityscapes import CityscapesDataset
from vision_mtl.data_modules.nyuv2 import NYUv2


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        train_transform: t.Optional[Union[T.Compose, A.Compose]] = None,
        test_transform: t.Optional[Union[T.Compose, A.Compose]] = None,
        train_size: float = cfg.data.train_size,
        batch_size: int = cfg.data.batch_size,
        num_workers: int = cfg.data.num_workers,
        shuffle_train: bool = cfg.data.shuffle_train,
        do_overfit: bool = False,
    ):
        super().__init__()
        self.do_overfit = do_overfit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.shuffle_train = shuffle_train
        self.data_train: t.Union[CityscapesDataset, NYUv2] = None
        self.data_val: t.Union[CityscapesDataset, NYUv2] = None
        self.data_test: t.Union[CityscapesDataset, NYUv2] = None
        self.data_predict: t.Union[CityscapesDataset, NYUv2] = None
        self.benchmark_batch = None
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dataset_name = dataset_name
        self.save_hyperparameters()

    def setup(self, stage: t.Optional[str] = None) -> None:
        if self.dataset_name == "cityscapes":
            ds_cls = CityscapesDataset
        else:
            ds_cls = NYUv2

        data_train = ds_cls(
            stage="train",
            transforms=self.train_transform,
        )
        try:
            self.benchmark_batch = data_train.load_benchmark_batch()
        except Exception as e:
            print("Failed to load benchmark batch: ", e)
            self.benchmark_batch = None
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

        val_stage_name = "val" if self.dataset_name == "cityscapes" else "test"
        if stage == "test" or stage is None:
            if self.do_overfit:
                self.data_test = self.data_train
            else:
                self.data_test = ds_cls(
                    stage=val_stage_name,
                    transforms=self.test_transform,
                )
        if stage == "predict" or stage is None:
            if self.do_overfit:
                self.data_predict = self.data_train
            else:
                self.data_predict = ds_cls(
                    stage=val_stage_name,
                    transforms=self.test_transform,
                )

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
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
