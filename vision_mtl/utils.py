import argparse
from functools import reduce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument("--do_optimize", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--model_name",
        choices=[
            "basic",
            "mtan",
            "csnet",
        ],
        default="basic",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    args, _ = parser.parse_known_args()
    return args


def get_module_by_name(module, access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
