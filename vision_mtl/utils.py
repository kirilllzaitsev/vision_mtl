import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_overfit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    args, _ = parser.parse_known_args()
    return args