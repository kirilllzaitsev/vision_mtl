import argparse
import os
import re

import torch
import yaml

from vision_mtl.cfg import cfg
from vision_mtl.models.basic_model import BasicMTLModel
from vision_mtl.models.cross_stitch_model import CSNet, get_model_with_dense_preds
from vision_mtl.models.mtan_model import MTANMiniUnet


def build_model(args):
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
    return model


def summarize_epoch_metrics(step_results):
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
        "loss": loss,
        "accuracy": accuracy,
        "jaccard_index": jaccard_index,
        "fbeta_score": fbeta_score,
    }
    return metrics


def print_metrics(prefix, train_epoch_metrics):
    metrics_str = ""
    for k, v in train_epoch_metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() > 1:
                value = v[-1]
            else:
                value = v.item()
        else:
            value = v[-1]
        print(f"{prefix}/{k}: {value:.3f}\t", end="")
        metrics_str += f"{k}: {value:.3f}\t"
    return metrics_str


def save_ckpt(
    module, optimizer, scheduler, epoch, save_path_model, save_path_session, exp=None
):
    torch.save(
        {
            "model": module.state_dict(),
        },
        save_path_model,
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        save_path_session,
    )
    if exp:
        log_ckpt_to_exp(exp, save_path_model, "ckpt")
        log_ckpt_to_exp(exp, save_path_session, "ckpt")
    print(f"Saved model to {save_path_model}")


def log_params_to_exp(experiment, params: dict, prefix: str):
    experiment.log_parameters({f"{prefix}/{str(k)}": v for k, v in params.items()})


def log_ckpt_to_exp(experiment, ckpt_path: str, model_name):
    experiment.log_model(model_name, ckpt_path, overwrite=False)


def log_args(args, save_path, exp=None):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    with open(save_path, "w") as f:
        yaml.dump(
            {"args": args},
            f,
            default_flow_style=False,
        )
    if exp:
        exp.log_asset(save_path)


def load_args(load_path):
    with open(load_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)["args"]
    return argparse.Namespace(**args)


def load_ckpt(ckpt_dir, epoch=None):
    session_ckpt = load_ckpt_session(ckpt_dir)
    session_ckpt, model_ckpt = load_ckpt_model(ckpt_dir, epoch=epoch)
    return session_ckpt, model_ckpt


def load_ckpt_model(ckpt_dir, epoch=None):
    artifact_name_regex = r"model_(\d+).pt"
    if epoch is not None:
        artifact_name = f"model_{epoch}.pt"
    else:
        available_model_ckpts = [
            f for f in os.listdir(ckpt_dir) if re.match(artifact_name_regex, f)
        ]
        if len(available_model_ckpts) == 0:
            raise ValueError("No model ckpt found")
        artifact_name = sorted(
            available_model_ckpts,
            key=lambda x: int(re.match(artifact_name_regex, x).group(1)),
        )[-1]
    model_ckpt_path = os.path.join(ckpt_dir, artifact_name)
    print(f"Loading model from {model_ckpt_path}")
    model_ckpt = torch.load(model_ckpt_path)
    return model_ckpt


def load_ckpt_session(ckpt_dir):
    session_ckpt_path = os.path.join(ckpt_dir, "session.pt")
    session_ckpt = torch.load(session_ckpt_path)
    return session_ckpt
