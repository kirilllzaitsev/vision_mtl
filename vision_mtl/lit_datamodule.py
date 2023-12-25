from typing import Union

import albumentations as A
import albumentations.pytorch as pytorch
import lightning.pytorch as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from vision_mtl.data.cityscapes import PhotopicVisionDataset
from vision_mtl.data.transforms import test_transform, train_transform


class PhotopicVisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_base_dir: str,
        train_transform: Union[T.Compose, A.Compose] = train_transform,
        test_transform: Union[T.Compose, A.Compose] = test_transform,
        train_size: float = 0.8,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_base_dir = data_base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None
        self.save_hyperparameters()

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            data_train = PhotopicVisionDataset(
                stage="train",
                data_base_dir=self.data_base_dir,
                transforms=self.train_transform,
            )

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
            self.data_test = PhotopicVisionDataset(
                stage="val",
                data_base_dir=self.data_base_dir,
                transforms=self.test_transform,
            )
        if stage == "predict" or stage is None:
            self.data_predict = PhotopicVisionDataset(
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
            shuffle=True,
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
