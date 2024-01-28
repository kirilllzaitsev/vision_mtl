"""This module trains the Lightning module in a standard way, i.e., with a vanilla PyTorch training loop.
The reason for this is that the Lightning module does not get optimized withe the PyTorch Lightning Trainer."""

import argparse
import copy
import functools
import os
from collections import defaultdict

import optuna
from optuna.trial import TrialState

if os.environ.get("DISPLAY") != ":0":
    import matplotlib

    matplotlib.use("Agg")

import typing as t

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from vision_mtl.cfg import DataConfig, cfg
from vision_mtl.lit_datamodule import MTLDataModule
from vision_mtl.lit_module import MTLModule
from vision_mtl.utils.pipeline_utils import (
    build_model,
    create_tracking_exp,
    fetch_data_cfg,
    load_ckpt_model,
    log_args,
    log_params_to_exp,
    print_metrics,
    save_ckpt,
    summarize_epoch_metrics,
)
from vision_mtl.utils.utils import parse_args
from vision_mtl.utils.vis_utils import plot_preds


def run_pipe(
    args: argparse.Namespace,
    module: MTLModule,
    datamodule: MTLDataModule,
    num_epochs: int,
    device: t.Union[str, torch.device],
    exp: comet_ml.Experiment,
    logger: TensorBoardLogger,
) -> t.Dict[str, t.Dict[str, list]]:
    """Run the training loop for num_epochs and return metrics from training and validation epochs."""

    global_step = 0
    val_step = 0

    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.9, verbose=True
    )

    module.to(device)

    benchmark_batch = datamodule.benchmark_batch
    if benchmark_batch is not None:
        benchmark_batch = module.transfer_batch_to_device(benchmark_batch, device, 0)
    else:
        print("A batch for benchmarking is not found.")

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    epoch_metrics = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }

    for epoch in epoch_pbar:
        print(f"### Epoch {epoch+1}/{num_epochs} ###")
        stage = "train"
        print(f"---{stage.upper()}---")

        train_loss = 0

        train_pbar = tqdm(
            datamodule.train_dataloader(), desc="Train Batches", leave=False
        )
        for batch in train_pbar:
            optimizer.zero_grad()
            batch = module.transfer_batch_to_device(batch, device, 0)
            loss = module.training_step(batch, batch_idx=0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for k, v in module.step_outputs[stage].items():
                logger.log_metrics({f"step/{stage}/{k}": v[-1]}, step=global_step)
                if exp:
                    exp.log_metric(f"step/{stage}/{k}", v[-1], step=global_step)
            pbar_postfix = print_metrics(f"{stage}", module.step_outputs[stage])
            train_pbar.set_postfix_str(pbar_postfix)
            train_pbar.update()

            global_step += 1

        train_epoch_metrics = module.on_train_epoch_end()
        for k, v in train_epoch_metrics.items():
            epoch_metrics[stage][k].append(v)
        pbar_postfix = print_metrics("epoch", train_epoch_metrics)
        epoch_pbar.set_postfix_str(pbar_postfix)
        logger.log_metrics(
            {f"epoch/{k}": v for k, v in train_epoch_metrics.items()},
            step=epoch,
        )
        if exp:
            exp.log_metrics(
                {f"epoch/{k}": v for k, v in train_epoch_metrics.items()},
                step=epoch,
            )

        if ((epoch + 1) % args.val_epoch_freq) == 0:
            stage = "val"
            print(f"---{stage.upper()}---")

            with torch.no_grad():
                if benchmark_batch is not None:
                    benchmark_preds = module.predict_step(benchmark_batch, 0, 0)
                    fig = plot_preds(
                        batch_size=4,
                        inputs_batch=benchmark_batch,
                        preds_batch=benchmark_preds,
                    )
                    exp.log_figure("preds", fig)
                    if args.do_show_preds:
                        plt.show()
                    plt.close()

                val_loss = 0

                val_pbar = tqdm(
                    datamodule.val_dataloader(), desc="Val Batches", leave=False
                )
                for batch in val_pbar:
                    batch = module.transfer_batch_to_device(batch, device, 0)
                    loss = module.validation_step(batch, batch_idx=0)

                    val_loss += loss.item()
                    for k, v in module.step_outputs[stage].items():
                        logger.log_metrics({f"step/{stage}/{k}": v[-1]}, step=val_step)
                        if exp:
                            exp.log_metric(f"step/{stage}/{k}", v[-1], step=val_step)
                    pbar_postfix = print_metrics(f"{stage}", module.step_outputs[stage])
                    val_pbar.set_postfix_str(pbar_postfix)
                    val_pbar.update()

                    val_step += 1

            val_epoch_metrics = module.on_validation_epoch_end()
            for k, v in val_epoch_metrics.items():
                epoch_metrics[stage][k].append(v)
            pbar_postfix = print_metrics(f"epoch/{stage}", val_epoch_metrics)
            epoch_pbar.set_postfix_str(pbar_postfix)

            logger.log_metrics(
                {f"epoch/{k}": v for k, v in val_epoch_metrics.items()},
                step=epoch,
            )
            if exp:
                exp.log_metrics(
                    {f"epoch/{k}": v for k, v in val_epoch_metrics.items()},
                    step=epoch,
                )

            scheduler.step(val_loss)

        if (epoch + 1) % args.save_epoch_freq == 0 or epoch == args.num_epochs - 1:
            save_path_model = os.path.join(logger.log_dir, f"model_{epoch}.pt")
            save_path_session = os.path.join(logger.log_dir, f"session_{epoch}.pt")
            save_ckpt(
                module=module,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                save_path_model=save_path_model,
                save_path_session=save_path_session,
                exp=exp,
            )

    exp.end()

    return epoch_metrics


@torch.no_grad()
def predict(
    predict_dataloader: torch.utils.data.DataLoader,
    module: MTLModule,
    device: t.Union[str, torch.device],
    do_plot_preds: bool = False,
    exp: t.Optional[comet_ml.Experiment] = None,
    do_show_preds: bool = False,
):
    """Predict on the predict_dataloader and return the predictions and metrics."""

    preds = []
    module.eval()
    module.to(device)
    for pred_batch in tqdm(predict_dataloader, desc="Predict Batches"):
        pred_batch = module.transfer_batch_to_device(pred_batch, device, 0)
        batch_preds = module.predict_step(pred_batch, 0, 0)
        preds.append(batch_preds)
        if do_plot_preds:
            if isinstance(pred_batch, dict):
                batch_size = pred_batch["img"].shape[0]
            else:
                batch_size = len(pred_batch)
            fig = plot_preds(batch_size, pred_batch, batch_preds)
            if exp:
                exp.log_figure("preds", fig)
            if do_show_preds:
                plt.show()
            plt.close()
    predict_metrics = module.on_predict_epoch_end()
    return preds, predict_metrics


def main():
    args = parse_args()
    cfg.update_fields_with_args(args)

    data_cfg = fetch_data_cfg(args.dataset_name)
    main_components = create_main_components(init_model, args, data_cfg)
    datamodule = main_components["datamodule"]
    module = main_components["module"]

    if args.do_optimize:
        from vision_mtl.hyperparam_tuning import run_study

        optimal_params = run_study(args, data_cfg)
        update_args(args, optimal_params)
        args.exp_tags += ["best_trial"]

    tools = create_tools(args)
    exp = tools["exp"]
    logger = tools["logger"]

    main_components = create_main_components(init_model, args, data_cfg)
    datamodule = main_components["datamodule"]
    module = main_components["module"]

    run_pipe(
        args,
        module,
        datamodule,
        args.num_epochs,
        cfg.device,
        exp=exp,
        logger=logger,
    )

    preds, predict_metrics = predict(
        datamodule.predict_dataloader(),
        module,
        cfg.device,
        args.do_plot_preds,
        exp=exp,
        do_show_preds=args.do_show_preds,
    )
    torch.save(preds, os.path.join(logger.log_dir, "preds.pt"))

    print_metrics("predict", predict_metrics)
    if exp:
        exp.log_metrics(
            {f"epoch/{k}": v for k, v in predict_metrics.items()},
            step=args.num_epochs,
        )

    exp.end()


if __name__ == "__main__":
    main()
