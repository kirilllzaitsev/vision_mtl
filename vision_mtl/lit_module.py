import typing as t
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MeanAbsoluteError

from vision_mtl.cfg import cfg
from vision_mtl.losses import SILogLoss
from vision_mtl.models.basic_model import BasicMTLModel
from vision_mtl.utils.pipeline_utils import summarize_epoch_metrics


class MTLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optim_dict: t.Optional[dict] = None,
        lr: t.Optional[float] = None,
        device: str = cfg.device,
        loss_segm_weight: float = 1.0,
        loss_depth_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.num_classes = num_classes
        self.model = model
        self.segm_criterion = nn.CrossEntropyLoss()
        self.depth_criterion = SILogLoss()
        self.optim_dict = optim_dict
        self.loss_segm_weight = loss_segm_weight
        self.loss_depth_weight = loss_depth_weight

        self.step_outputs = {
            k: {
                "loss": [],
                "accuracy": [],
                "jaccard_index": [],
                "fbeta_score": [],
                "mae": [],
            }
            for k in ["train", "val", "test", "predict"]
        }

        self.metrics = {
            "accuracy": Accuracy(
                threshold=0.5,
                num_classes=num_classes,
                ignore_index=None,
                average="micro",
            ).to(device),
            "fbeta_score": FBetaScore(
                beta=1.0,
                threshold=0.5,
                num_classes=num_classes,
                average="weighted",
                ignore_index=None,
                mdmc_average="global",
            ).to(device),
            "jaccard_index": JaccardIndex(
                threshold=0.5,
                num_classes=num_classes,
                ignore_index=None,
            ).to(device),
            "mae": MeanAbsoluteError().to(device),
        }
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)

    def shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        img, gt_mask, gt_depth = batch["img"], batch["mask"], batch["depth"]

        raw_out = self(img)
        out = self.postprocess_raw_out(raw_out)

        all_losses = self.calc_losses(gt_mask, gt_depth, out)

        all_metrics = self.calc_metrics(gt_mask, gt_depth, out)

        self.update_step_stats(stage, all_losses, all_metrics)
        for k, v in self.step_outputs[stage].items():
            self.log(
                f"{stage}_{k}",
                v[-1],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return all_losses["loss"]

    def update_step_stats(
        self, stage: str, all_losses: dict, all_metrics: dict
    ) -> None:
        self.step_outputs[stage]["loss"].append(all_losses["loss"])
        self.step_outputs[stage]["accuracy"].append(all_metrics["accuracy"])
        self.step_outputs[stage]["jaccard_index"].append(all_metrics["jaccard_index"])
        self.step_outputs[stage]["fbeta_score"].append(all_metrics["fbeta_score"])
        self.step_outputs[stage]["mae"].append(all_metrics["mae"])

    def calc_metrics(
        self, gt_mask: torch.Tensor, gt_depth: torch.Tensor, out: dict
    ) -> dict:
        accuracy = self.metrics["accuracy"](out["segm_predictions"], gt_mask)
        jaccard_index = self.metrics["jaccard_index"](out["segm_predictions"], gt_mask)
        fbeta_score = self.metrics["fbeta_score"](out["segm_predictions"], gt_mask)
        mae = self.metrics["mae"](out["depth_predictions"], gt_depth)
        return {
            "accuracy": accuracy,
            "jaccard_index": jaccard_index,
            "fbeta_score": fbeta_score,
            "mae": mae,
        }

    def calc_losses(
        self, gt_mask: torch.Tensor, gt_depth: torch.Tensor, out: dict
    ) -> dict:
        loss_segm = self.segm_criterion(out["segm_logits"], gt_mask)
        loss_depth = self.depth_criterion(out["depth_predictions"], gt_depth)

        loss = self.loss_segm_weight * loss_segm + self.loss_depth_weight * loss_depth
        return {
            "loss": loss,
            "loss_segm": loss_segm,
            "loss_depth": loss_depth,
        }

    def postprocess_raw_out(self, out: dict) -> dict:
        segm_logits = out["segm"]
        depth_logits = out["depth"]

        segm_pred_probs = F.softmax(input=segm_logits, dim=1)
        segm_predictions = torch.argmax(segm_pred_probs, dim=1)
        depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
        return {
            "segm_logits": segm_logits,
            "segm_predictions": segm_predictions,
            "depth_predictions": depth_predictions,
        }

    def training_step(self, batch: dict, batch_idx: Any):
        return self.shared_step(batch=batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: Any):
        return self.shared_step(batch=batch, stage="val")

    def test_step(self, batch: dict, batch_idx: Any):
        return self.shared_step(batch=batch, stage="test")

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        img = batch["img"]

        raw_out = self(img)

        out = self.postprocess_raw_out(raw_out)

        if "mask" in batch and "depth" in batch:
            gt_mask, gt_depth = batch["mask"], batch["depth"]
            all_losses = self.calc_losses(gt_mask, gt_depth, out)
            all_metrics = self.calc_metrics(gt_mask, gt_depth, out)
            self.update_step_stats("predict", all_losses, all_metrics)

        preds = {"segm": out["segm_predictions"], "depth": out["depth_predictions"]}
        return preds

    def shared_epoch_end(self, stage: Any):
        step_results = self.step_outputs[stage]
        metrics = summarize_epoch_metrics(step_results, metric_name_prefix=stage)
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics

    def on_train_epoch_end(self):
        metrics = self.shared_epoch_end(stage="train")
        return metrics

    def on_validation_epoch_end(self):
        metrics = self.shared_epoch_end(stage="val")
        return metrics

    def on_test_epoch_end(self):
        metrics = self.shared_epoch_end(stage="test")
        return metrics

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

    def transfer_batch_to_device(
        self, batch: dict, device: t.Union[str, torch.device], dataloader_idx: int = 0
    ):
        if isinstance(batch, dict):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

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

    def parameters(self, recurse: bool = True):
        for p in self.model.parameters():
            yield p

    def to(self, *args: Any, **kwargs: Any):
        self = super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        return self


if __name__ == "__main__":
    model = BasicMTLModel(segm_classes=19)
    module = MTLModule(model, num_classes=19).to(cfg.device)
    print(module)
    batch_size = 1
    sample_batch = {
        "img": torch.randn(batch_size, 3, 128, 256).to(cfg.device),
        "mask": torch.randint(0, 19, (batch_size, 128, 256)).to(cfg.device),
        "depth": torch.randn(batch_size, 128, 256).to(cfg.device),
    }
    module.training_step(sample_batch, 0)
