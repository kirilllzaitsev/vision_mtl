import torch

from vision_mtl.cfg import cfg
from vision_mtl.inference_utils import get_segm_preds
from vision_mtl.lit_datamodule import CityscapesDataModule
from vision_mtl.loss_utils import calc_loss
from vision_mtl.pipeline_utils import build_model, summarize_epoch_metrics
from vision_mtl.utils import parse_args
from vision_mtl.vis_utils import plot_preds

# torch.set_float32_matmul_precision("medium")

args = parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming the necessary imports for metrics like Accuracy, JaccardIndex, FBetaScore, and SILogLoss
from torchmetrics import Accuracy, FBetaScore, JaccardIndex

from vision_mtl.losses import SILogLoss


# Define the training loop
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    num_classes=cfg.data.num_classes,
):
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.95, verbose=True
    )
    segm_criterion = nn.CrossEntropyLoss()
    depth_criterion = SILogLoss()

    step_outputs = {
        k: {
            "loss": [],
            "accuracy": [],
            "jaccard_index": [],
            "fbeta_score": [],
        }
        for k in ["train", "val", "test"]
    }

    metrics = {
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

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger("lightning_logs", name="prototyping")

    global_step = 0
    for epoch in range(num_epochs):
        print(f"### Epoch {epoch+1}/{num_epochs} ###")
        model.train()
        train_loss = 0
        stage = "train"
        for batch in train_loader:
            img, gt_mask, gt_depth = (
                batch["img"].to(device),
                batch["mask"].to(device),
                batch["depth"].to(device),
            )
            optimizer.zero_grad()
            out = model(img)
            # Loss calculation and backpropagation
            segm_validity_mask = gt_mask != -1
            loss = calc_loss(
                out,
                gt_mask,
                gt_depth,
                segm_criterion,
                depth_criterion,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Update train_accuracy, train_jaccard_index, train_fbeta_score using metrics
            _, segm_predictions_classes = get_segm_preds(
                segm_validity_mask, out["segm"]
            )
            gt_mask_valid = gt_mask[segm_validity_mask]
            accuracy = metrics["accuracy"](segm_predictions_classes, gt_mask_valid)
            jaccard_index = metrics["jaccard_index"](
                segm_predictions_classes, gt_mask_valid
            )
            fbeta_score = metrics["fbeta_score"](
                segm_predictions_classes, gt_mask_valid
            )

            step_outputs[stage]["loss"].append(loss)
            step_outputs[stage]["accuracy"].append(accuracy)
            step_outputs[stage]["jaccard_index"].append(jaccard_index)
            step_outputs[stage]["fbeta_score"].append(fbeta_score)
            for k, v in step_outputs[stage].items():
                print(f"{stage}_{k}", v[-1])
                logger.log_metrics({f"{stage}/{k}": v[-1]}, step=global_step)

            global_step += 1

        train_epoch_metrics = summarize_epoch_metrics(step_outputs, stage)
        for k, v in train_epoch_metrics.items():
            print(f"{stage}/epoch_{k}", v)
        logger.log_metrics(train_epoch_metrics, step=epoch)

        stage = "val"
        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                img, gt_mask, gt_depth = (
                    batch["img"].to(device),
                    batch["mask"].to(device),
                    batch["depth"].to(device),
                )
                out = model(img)
                # Loss calculation
                segm_validity_mask = gt_mask != -1
                loss = calc_loss(
                    out,
                    gt_mask,
                    gt_depth,
                    segm_criterion,
                    depth_criterion,
                )

                val_loss += loss.item()

                _, segm_predictions_classes = get_segm_preds(
                    segm_validity_mask, out["segm"]
                )
                gt_mask_valid = gt_mask[segm_validity_mask]
                accuracy = metrics["accuracy"](segm_predictions_classes, gt_mask_valid)
                jaccard_index = metrics["jaccard_index"](
                    segm_predictions_classes, gt_mask_valid
                )
                fbeta_score = metrics["fbeta_score"](
                    segm_predictions_classes, gt_mask_valid
                )

                step_outputs[stage]["loss"].append(loss)
                step_outputs[stage]["accuracy"].append(accuracy)
                step_outputs[stage]["jaccard_index"].append(jaccard_index)
                step_outputs[stage]["fbeta_score"].append(fbeta_score)
                for k, v in step_outputs[stage].items():
                    print(f"{stage}/{k}", v[-1])

        val_epoch_metrics = summarize_epoch_metrics(step_outputs, stage)
        for k, v in val_epoch_metrics.items():
            print(f"{stage}/epoch_{k}", v)

        scheduler.step(val_loss)


model = build_model(args)

# Datamodule
datamodule = CityscapesDataModule(
    data_base_dir=cfg.data.data_dir,
    batch_size=args.batch_size,
    do_overfit=args.do_overfit,
)
datamodule.setup()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_model(
    model,
    datamodule.train_dataloader(),
    datamodule.val_dataloader(),
    optimizer,
    args.num_epochs,
    cfg.device,
)


def predict_step(self, batch):
    img = batch["img"]

    out = self.forward(img)

    segm_logits = out["segm"]
    depth_logits = out["depth"]

    segm_pred = F.softmax(input=segm_logits, dim=1)
    segm_predictions = torch.argmax(segm_pred, dim=1)
    depth_predictions = F.sigmoid(input=depth_logits).squeeze(1)
    preds = {"segm": segm_predictions, "depth": depth_predictions}
    return preds


preds = []
for pred_Batch in datamodule.predict_dataloader():
    for k, v in pred_Batch.items():
        pred_Batch[k] = v.to(cfg.device)
    preds.append(predict_step(model, pred_Batch))

plot_preds(args.batch_size, datamodule.predict_dataloader(), preds)
