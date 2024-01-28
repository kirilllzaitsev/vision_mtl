import argparse
from functools import reduce

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("pipe")
    pipe_args.add_argument("--do_overfit", action="store_true")
    pipe_args.add_argument("--do_optimize", action="store_true")
    pipe_args.add_argument("--do_plot_preds", action="store_true")
    pipe_args.add_argument("--do_show_preds", action="store_true")
    pipe_args.add_argument("--exp_disabled", action="store_true")
    pipe_args.add_argument("--ckpt_dir")
    pipe_args.add_argument("--run_name")
    pipe_args.add_argument("--device", default="cuda:0")
    pipe_args.add_argument("--exp_tags", nargs="*", default=[])

    model_args = parser.add_argument_group("model")
    model_args.add_argument(
        "--model_name", choices=["basic", "mtan", "csnet"], default="basic"
    )
    model_args.add_argument("--backbone_weights", choices=["imagenet"])
    model_args.add_argument("--channel_wise_stitching", action="store_true")

    data_args = parser.add_argument_group("data")
    data_args.add_argument(
        "--dataset_name", choices=["cityscapes", "nyuv2"], default="cityscapes"
    )
    data_args.add_argument("--batch_size", type=int, default=1)
    data_args.add_argument("--num_workers", type=int, default=0)

    optuna_args = parser.add_argument_group("opt")
    optuna_args.add_argument("--n_trials", type=int, default=7)
    optuna_args.add_argument("--n_jobs", type=int, default=2)

    trainer_args = parser.add_argument_group("trainer")
    trainer_args.add_argument("--lr", type=float, default=5e-3)
    trainer_args.add_argument("--loss_segm_weight", type=float, default=1)
    trainer_args.add_argument("--loss_depth_weight", type=float, default=1)
    trainer_args.add_argument("--num_epochs", type=int, default=10)
    trainer_args.add_argument("--val_epoch_freq", type=int, default=1)
    trainer_args.add_argument("--save_epoch_freq", type=int, default=10)

    args, _ = parser.parse_known_args()
    return args


def get_module_by_name(module: torch.nn.Module, access_string: str) -> torch.nn.Module:
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def print_sample_stats(sample: dict) -> None:
    for k in sample:
        print(k)
        print(f"{sample[k].shape=}")
        print(f"{sample[k].min()=} {sample[k].max()=}")
        print(f" {sample[k].median()=} {sample[k].dtype=}")
        print("-" * 10)
