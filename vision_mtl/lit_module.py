import typing as t
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MaxMetric, MeanMetric

from vision_mtl.cfg import cfg
from vision_mtl.losses import SILogLoss
from vision_mtl.models.basic_model import BasicMTLModel


class LightningPhotopicVisionModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optim_dict: dict = None,
        lr: float = None,
        num_classes: int = cfg.data.num_classes,
        device: str = cfg.device,
    ):
        self.example_input_array = torch.Tensor(1, 3, 128, 256)
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.num_classes = num_classes
        self.model = model
        self.segm_criterion = nn.CrossEntropyLoss()
        self.depth_criterion = SILogLoss()
        self.optim_dict = optim_dict

        self.step_outputs = {
            k: {
                "loss": [],
                "accuracy": [],
                "jaccard_index": [],
                "fbeta_score": [],
            }
            for k in ["train", "val", "test"]
        }

        self.metrics = {
            "accuracy": Accuracy(
                threshold=0.5,
                num_classes=num_classes,
                ignore_index=None,
                average="micro",
            ).to(device),
            "jaccard_index": JaccardIndex(
                threshold=0.5,
                num_classes=num_classes,
                ignore_index=None,
            ).to(device),
            "fbeta_score": FBetaScore(
                beta=1.0,
                threshold=0.5,
                num_classes=num_classes,
                average="micro",
                ignore_index=None,
                mdmc_average="global",
            ).to(device),
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        img, gt_mask, gt_depth = batch["img"], batch["mask"], batch["depth"]

        # assert img.ndim == 4
        # assert img.max() <= 3 and img.min() >= -3
        # assert gt_mask.ndim == 3
        # assert gt_mask.max() <= 22 and gt_mask.min() >= 0

        out = self.forward(img)
        segm_logits = out["segm"]
        depth_logits = out["depth"]

        segm_pred = F.softmax(input=segm_logits, dim=1)
        segm_predictions = torch.argmax(segm_pred, dim=1)

        segm_validity_mask = gt_mask != -1
        segm_validity_mask_with_channels = segm_validity_mask.unsqueeze(1).repeat(
            1, 19, 1, 1
        )
        segm_pred = segm_pred[segm_validity_mask_with_channels].reshape(-1, 19)
        segm_predictions = segm_predictions[segm_validity_mask]
        gt_mask = gt_mask[segm_validity_mask]

        loss_segm = self.segm_criterion(segm_pred, gt_mask)

        depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
        loss_depth = self.depth_criterion(depth_predictions, gt_depth)

        loss = loss_segm + loss_depth
        # loss = loss_segm
        # loss = loss_depth
        # loss = F.mse_loss(depth_predictions, gt_depth) + F.cross_entropy(
        #     segm_pred, gt_mask
        # )

        accuracy = self.metrics["accuracy"](segm_predictions, gt_mask)
        jaccard_index = self.metrics["jaccard_index"](segm_predictions, gt_mask)
        fbeta_score = self.metrics["fbeta_score"](segm_predictions, gt_mask)

        self.step_outputs[stage]["loss"].append(loss)
        self.step_outputs[stage]["accuracy"].append(accuracy)
        self.step_outputs[stage]["jaccard_index"].append(jaccard_index)
        self.step_outputs[stage]["fbeta_score"].append(fbeta_score)
        for k, v in self.step_outputs[stage].items():
            self.log(
                f"{stage}_{k}",
                v[-1],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def shared_epoch_end(self, stage: Any):
        step_results = self.step_outputs[stage]
        loss = torch.mean(torch.tensor([loss for loss in step_results["loss"]]))

        accuracy = torch.mean(
            torch.tensor([accuracy for accuracy in step_results["accuracy"]])
        )

        jaccard_index = torch.mean(
            torch.tensor(
                [jaccard_index for jaccard_index in step_results["jaccard_index"]]
            )
        )

        fbeta_score = torch.mean(
            torch.tensor([fbeta_score for fbeta_score in step_results["fbeta_score"]])
        )

        for key in step_results.keys():
            step_results[key].clear()

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": accuracy,
            f"{stage}_jaccard_index": jaccard_index,
            f"{stage}_fbeta_score": fbeta_score,
        }
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics

    def training_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="train")

    def on_train_epoch_end(self):
        metrics = self.shared_epoch_end(stage="train")
        return metrics

    def validation_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.shared_epoch_end(stage="val")
        return metrics

    def test_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="test")

    def on_test_epoch_end(self):
        metrics = self.shared_epoch_end(stage="test")
        return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        img = batch["img"]

        out = self.forward(img)

        segm_logits = out["segm"]
        depth_logits = out["depth"]

        segm_pred = F.softmax(input=segm_logits, dim=1)
        segm_predictions = torch.argmax(segm_pred, dim=1)
        depth_predictions = F.sigmoid(input=depth_logits).squeeze(1)
        preds = {"segm": segm_predictions, "depth": depth_predictions}
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=5, factor=0.95, verbose=True
            ),
            "interval": "epoch",
            # "monitor": "val_loss",
            "monitor": "train_loss",
        }

        optimization_dictionary = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
        return self.optim_dict if self.optim_dict else optimization_dictionary

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     if isinstance(batch, dict):
    #         for key in batch.keys():
    #             batch[key] = batch[key].to(device)
    #     else:
    #         batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    #     return batch

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                grads = v
                name = k
                self.logger.experiment.add_histogram(
                    tag=name, values=grads, global_step=self.trainer.global_step
                )


if __name__ == "__main__":
    model = BasicMTLModel(segm_classes=19)
    module = LightningPhotopicVisionModule(model).to(cfg.device)
    print(module)
    batch_size = 1
    sample_batch = {
        "img": torch.randn(batch_size, 3, 128, 256).to(cfg.device),
        "mask": torch.randint(0, 19, (batch_size, 128, 256)).to(cfg.device),
        "depth": torch.randn(batch_size, 128, 256).to(cfg.device),
    }
    module.training_step(sample_batch, 0)
    # print(out["segm"].shape)
