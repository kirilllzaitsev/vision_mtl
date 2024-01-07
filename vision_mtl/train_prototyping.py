
import torch

from vision_mtl.cfg import cfg
from vision_mtl.lit_datamodule import PhotopicVisionDataModule
from vision_mtl.models.basic_model import BasicMTLModel
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


def calc_loss(
    out, gt_mask, gt_depth, segm_criterion, depth_criterion
):
    segm_logits = out["segm"]
    depth_logits = out["depth"]

    # segm_pred_probs, _ = get_segm_preds(segm_validity_mask, segm_logits)
    # gt_mask = gt_mask[segm_validity_mask]

    loss_segm = segm_criterion(segm_logits, gt_mask)

    depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
    loss_depth = depth_criterion(depth_predictions, gt_depth)
    # loss_depth = 0

    loss = loss_segm + loss_depth
    return loss


def get_segm_preds(segm_validity_mask, segm_logits):
    segm_pred_probs = F.softmax(input=segm_logits, dim=1)
    segm_predictions_classes = torch.argmax(segm_pred_probs, dim=1)

    segm_validity_mask_with_channels = segm_validity_mask.unsqueeze(1).repeat(
        1, 19, 1, 1
    )
    segm_pred_probs = segm_pred_probs[segm_validity_mask_with_channels].reshape(-1, 19)
    segm_predictions_classes = segm_predictions_classes[segm_validity_mask]
    return segm_pred_probs, segm_predictions_classes


def summarize_epoch_metrics(step_outputs, stage):
    step_results = step_outputs[stage]
    loss = torch.mean(torch.tensor([loss for loss in step_results["loss"]]))

    accuracy = torch.mean(
        torch.tensor([accuracy for accuracy in step_results["accuracy"]])
    )

    jaccard_index = torch.mean(
        torch.tensor([jaccard_index for jaccard_index in step_results["jaccard_index"]])
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
    return metrics


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

    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger("lightning_logs", name="prototyping")

    global_step = 0
    for epoch in range(num_epochs):
        print(f"### Epoch {epoch+1}/{num_epochs} ###")
        model.train()
        train_loss, train_accuracy, train_jaccard_index, train_fbeta_score = 0, 0, 0, 0
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
            jaccard_index = metrics["jaccard_index"](segm_predictions_classes, gt_mask_valid)
            fbeta_score = metrics["fbeta_score"](segm_predictions_classes, gt_mask_valid)

            step_outputs[stage]["loss"].append(loss)
            step_outputs[stage]["accuracy"].append(accuracy)
            step_outputs[stage]["jaccard_index"].append(jaccard_index)
            step_outputs[stage]["fbeta_score"].append(fbeta_score)
            for k, v in step_outputs[stage].items():
                print(f"{stage}_{k}", v[-1])
                logger.log_metrics({f"{stage}_{k}": v[-1]}, step=global_step)
            
            global_step += 1

        train_epoch_metrics = summarize_epoch_metrics(step_outputs, stage)
        for k, v in train_epoch_metrics.items():
            print(f"{stage}_epoch_{k}", v)
        logger.log_metrics(train_epoch_metrics, step=epoch)
        continue
        stage = "val"
        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, val_jaccard_index, val_fbeta_score = 0, 0, 0, 0
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
                # Update val_accuracy, val_jaccard_index, val_fbeta_score using metrics

                _, segm_predictions_classes = get_segm_preds(
                    segm_validity_mask, out["segm"]
                )
                gt_mask_valid = gt_mask[segm_validity_mask]
                accuracy = metrics["accuracy"](segm_predictions_classes, gt_mask_valid)
                jaccard_index = metrics["jaccard_index"](segm_predictions_classes, gt_mask_valid)
                fbeta_score = metrics["fbeta_score"](segm_predictions_classes, gt_mask_valid)

                step_outputs[stage]["loss"].append(loss)
                step_outputs[stage]["accuracy"].append(accuracy)
                step_outputs[stage]["jaccard_index"].append(jaccard_index)
                step_outputs[stage]["fbeta_score"].append(fbeta_score)
                for k, v in step_outputs[stage].items():
                    print(f"{stage}_{k}", v[-1])
        
        val_epoch_metrics = summarize_epoch_metrics(step_outputs, stage)
        for k, v in val_epoch_metrics.items():
            print(f"{stage}_epoch_{k}", v)

        scheduler.step(val_loss)

        # print(
        #     f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}"
        # )
        # Print other metrics as needed


# Model initialization

if args.model_name == "basic":
    # basic
    model = BasicMTLModel(decoder_first_channel=540, num_decoder_layers=5)
elif args.model_name == "mtan":
    # MTAN
    map_tasks_to_num_channels = {
        "depth": 1,
        "segm": cfg.data.num_classes,
    }
    model = MTANMiniUnet(
        in_channels=3,
        map_tasks_to_num_channels=map_tasks_to_num_channels,
        task_subnets_hidden_channels=128,
        encoder_first_channel=32,
        encoder_num_channels=4,
    )
elif args.model_name == "csnet":
    # cross-stitch
    backbone_params = dict(
        encoder_name="timm-mobilenetv3_large_100",
        encoder_weights="imagenet",
        decoder_first_channel=256,
        num_decoder_layers=5,
    )
    models = {
        "depth": get_model_with_dense_preds(
            segm_classes=1, activation=None, backbone_params=backbone_params
        ),
        "segm": get_model_with_dense_preds(
            segm_classes=cfg.data.num_classes,
            activation=None,
            backbone_params=backbone_params,
        ),
    }
    model = CSNet(models, channels_wise_stitching=True)
else:
    raise NotImplementedError(f"Unknown model name: {args.model_name}")

# Datamodule
datamodule = PhotopicVisionDataModule(
    data_base_dir=cfg.data.data_dir,
    batch_size=args.batch_size,
    do_overfit=args.do_overfit,
)
datamodule.setup()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Assuming train_loader and val_loader are defined
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
    for k,v in pred_Batch.items():
        pred_Batch[k] = v.to(cfg.device)
    preds.append(predict_step(model, pred_Batch))

plot_preds(args.batch_size, datamodule.predict_dataloader(), preds)