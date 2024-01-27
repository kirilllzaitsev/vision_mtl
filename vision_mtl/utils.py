import argparse
from functools import reduce

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument("--do_optimize", action="store_true")
    parser.add_argument("--do_plot_preds", action="store_true")
    parser.add_argument("--do_show_preds", action="store_true")
    parser.add_argument("--exp_disabled", action="store_true")
    parser.add_argument("--channel_wise_stitching", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_epoch_freq", type=int, default=1)
    parser.add_argument("--save_epoch_freq", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=7)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument(
        "--model_name",
        choices=[
            "basic",
            "mtan",
            "csnet",
        ],
        default="basic",
    )
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--run_name")
    parser.add_argument(
        "--dataset_name", choices=["cityscapes", "nyuv2"], default="cityscapes"
    )
    parser.add_argument("--backbone_weights", choices=["imagenet"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--loss_segm_weight", type=float, default=1)
    parser.add_argument("--loss_depth_weight", type=float, default=1)
    parser.add_argument("--exp_tags", nargs="*", default=[])
    args, _ = parser.parse_known_args()
    return args


def get_module_by_name(module: torch.nn.Module, access_string: str) -> torch.nn.Module:
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
