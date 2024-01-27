import argparse
import glob
import numbers
import os
import re
import typing as t

import comet_ml
import torch
import yaml

from vision_mtl.cfg import DataConfig, cfg, cityscapes_data_cfg, nyuv2_data_cfg
from vision_mtl.models.basic_model import BasicMTLModel
from vision_mtl.models.cross_stitch_model import CSNet
from vision_mtl.models.model_utils import get_model_with_dense_preds
from vision_mtl.models.mtan_model import MTANMiniUnet


def build_model(
    args: argparse.Namespace, data_cfg: DataConfig
) -> t.Union[BasicMTLModel, MTANMiniUnet, CSNet]:
    """Instantiate a model based on the args. The models are aligned with each other in terms of the number of parameters."""

    if args.model_name == "basic":
        # basic
        model = BasicMTLModel(
            segm_classes=data_cfg.num_classes,
            decoder_first_channel=540,
            num_decoder_layers=5,
            encoder_weights=getattr(
                args,
                "backbone_weights",
                "imagenet",
            ),
        )
    elif args.model_name == "mtan":
        # MTAN
        map_tasks_to_num_channels = {
            "depth": 1,
            "segm": data_cfg.num_classes,
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
                segm_classes=data_cfg.num_classes,
                activation=None,
                backbone_params=backbone_params,
            ),
        }
        model = CSNet(
            models,
            channel_wise_stitching=getattr(args, "channel_wise_stitching", True),
        )
    else:
        raise NotImplementedError(f"Unknown model name: {args.model_name}")
    return model


def summarize_epoch_metrics(
    step_results: dict, metric_name_prefix: t.Optional[str] = None
) -> dict:
    """Average the metrics in step_results and return them as a dict."""

    if metric_name_prefix is None:
        metric_name_prefix = ""
    else:
        metric_name_prefix += "/"
    metrics = {
        f"{metric_name_prefix}{k}": torch.mean(
            torch.tensor([v for v in step_results[k]])
        ).item()
        for k in step_results.keys()
    }
    for key in step_results.keys():
        step_results[key].clear()
    return metrics


def print_metrics(prefix: str, train_epoch_metrics: dict) -> str:
    """Assemble a string out of the metrics and print it."""

    metrics_str = ""
    for k, v in train_epoch_metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() > 1:
                value = v[-1]
            else:
                value = v.item()
        else:
            if isinstance(v, numbers.Number):
                value = v
            else:
                value = v[-1]
        print(f"{prefix}/{k}: {value:.3f} ")
        metrics_str += f"{k}: {value:.3f} "
    return metrics_str


def save_ckpt(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: t.Any,
    epoch: int,
    save_path_model: str,
    save_path_session: str,
    exp: t.Optional[comet_ml.Experiment] = None,
) -> None:
    """Save model and session checkpoints locally and to Comet."""

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


def log_params_to_exp(
    experiment: comet_ml.Experiment, params: dict, prefix: str
) -> None:
    experiment.log_parameters({f"{prefix}/{str(k)}": v for k, v in params.items()})


def log_ckpt_to_exp(
    experiment: comet_ml.Experiment, ckpt_path: str, model_name: str
) -> None:
    experiment.log_model(model_name, ckpt_path, overwrite=False)


def log_args(
    args: argparse.Namespace,
    save_path: str,
    exp: t.Optional[comet_ml.Experiment] = None,
) -> None:
    if isinstance(args, argparse.Namespace):
        args_map = vars(args)
    else:
        args_map = args
    with open(save_path, "w") as f:
        yaml.dump(
            {"args": args_map},
            f,
            default_flow_style=False,
        )
    if exp:
        exp.log_asset(save_path)


def load_args(load_path: str) -> argparse.Namespace:
    with open(load_path, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)["args"]
    return argparse.Namespace(**args)


def load_ckpt(ckpt_dir: str, epoch: t.Optional[int] = None) -> t.Tuple:
    """Load a checkpoint from a directory.
    Returns:
        A tuple of (session_ckpt, model_ckpt), where session_ckpt refers to the optimizer and scheduler state, and model_ckpt refers to the model state.
    """
    session_ckpt = load_ckpt_session(ckpt_dir)
    session_ckpt, model_ckpt = load_ckpt_model(ckpt_dir, epoch=epoch)
    return session_ckpt, model_ckpt


def load_ckpt_model(
    ckpt_dir: str,
    epoch: t.Optional[int] = None,
    artifact_name_regex: str = r"model_(\d+).pt",
) -> t.Any:
    """Load a model checkpoint from a directory based on the epoch number. The epoch number is parsed from the artifact name using the artifact_name_regex."""
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


def load_ckpt_session(ckpt_dir: str, filename="session.pt") -> t.Any:
    session_ckpt_path = os.path.join(ckpt_dir, filename)
    session_ckpt = torch.load(session_ckpt_path)
    return session_ckpt


def create_tracking_exp(
    args: argparse.Namespace,
    exp_disabled: bool = True,
    force_disabled: bool = False,
    project_name: str = cfg.logger.project_name,
) -> comet_ml.Experiment:
    """Creates an experiment on Comet and logs all the code in the current directory."""

    exp_init_args = dict(
        api_key=cfg.logger.api_key,
        auto_output_logging="simple",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_env_details=True,
        log_env_host=False,
        log_env_gpu=True,
        log_env_cpu=True,
        log_code=False,
        disabled=getattr(args, "exp_disabled", exp_disabled) or force_disabled,
    )
    if getattr(args, "resume_exp", False):
        from comet_ml.api import API

        api = API(api_key=cfg.logger.api_key)
        exp_api = api.get(f"{cfg.logger.username}/{project_name}/{args.exp_name}")
        experiment = comet_ml.ExistingExperiment(
            **exp_init_args, experiment_key=exp_api.id
        )
    else:
        experiment = comet_ml.Experiment(**exp_init_args, project_name=project_name)

    for code_file in glob.glob("./*.py"):
        experiment.log_code(code_file)

    print(
        f'Please leave a note about the experiment at {experiment._get_experiment_url(tab="notes")}'
    )

    return experiment


def fetch_data_cfg(dS_name: str) -> DataConfig:
    if dS_name == "cityscapes":
        return cityscapes_data_cfg
    elif dS_name == "nyuv2":
        return nyuv2_data_cfg
    else:
        raise ValueError(f"Unknown dataset name {dS_name}")
