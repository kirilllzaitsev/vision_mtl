"""This module trains the Lightning module in a standard way, i.e., with a vanilla PyTorch training loop.
The reason for this is that the Lightning module does not get optimized withe the PyTorch Lightning Trainer."""

import copy
import functools
import os
from collections import defaultdict

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from vision_mtl.cfg import cfg
from vision_mtl.lit_datamodule import CityscapesDataModule
from vision_mtl.lit_module import MTLModule
from vision_mtl.pipeline_utils import (
    build_model,
    create_tracking_exp,
    load_ckpt_model,
    log_args,
    log_params_to_exp,
    print_metrics,
    save_ckpt,
    summarize_epoch_metrics,
)
from vision_mtl.utils import parse_args
from vision_mtl.vis_utils import plot_preds


def run_pipe(
    args,
    module: MTLModule,
    datamodule: CityscapesDataModule,
    num_epochs,
    device,
    exp: comet_ml.Experiment,
    logger,
):
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
                logger.log_metrics({f"{stage}/{k}": v[-1]}, step=global_step)
            pbar_postfix = print_metrics(f"{stage}", module.step_outputs[stage])
            train_pbar.set_postfix_str(pbar_postfix)
            train_pbar.update()

            global_step += 1

        train_epoch_metrics = summarize_epoch_metrics(module.step_outputs[stage])
        for k, v in train_epoch_metrics.items():
            epoch_metrics[stage][k].append(v)
        pbar_postfix = print_metrics(f"epoch/{stage}", train_epoch_metrics)
        epoch_pbar.set_postfix_str(pbar_postfix)
        logger.log_metrics(
            {f"epoch/{stage}/{k}": v for k, v in train_epoch_metrics.items()},
            step=epoch,
        )

        if ((epoch + 1) % args.val_epoch_freq) == 0:
            stage = "val"
            print(f"---{stage.upper()}---")

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

            with torch.no_grad():
                val_pbar = tqdm(
                    datamodule.val_dataloader(), desc="Val Batches", leave=False
                )
                for batch in val_pbar:
                    batch = module.transfer_batch_to_device(batch, device, 0)
                    loss = module.validation_step(batch, batch_idx=0)

                    val_loss += loss.item()
                    for k, v in module.step_outputs[stage].items():
                        logger.log_metrics({f"{stage}/{k}": v[-1]}, step=val_step)
                    pbar_postfix = print_metrics(f"{stage}", module.step_outputs[stage])
                    val_pbar.set_postfix_str(pbar_postfix)
                    val_pbar.update()

                    val_step += 1

            val_epoch_metrics = summarize_epoch_metrics(module.step_outputs[stage])
            for k, v in val_epoch_metrics.items():
                epoch_metrics[stage][k].append(v)
            pbar_postfix = print_metrics(f"epoch/{stage}", val_epoch_metrics)
            epoch_pbar.set_postfix_str(pbar_postfix)

            logger.log_metrics(
                {f"epoch/{stage}/{k}": v for k, v in val_epoch_metrics.items()},
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
    predict_dataloader,
    module,
    device,
    do_plot_preds=False,
    exp=None,
    do_show_preds=False,
):
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
    stage = "predict"
    predict_metrics = summarize_epoch_metrics(module.step_outputs[stage])
    return preds, predict_metrics


def init_model(args):
    model = build_model(args)
    module = MTLModule(model=model, lr=args.lr)
    if args.ckpt_dir:
        module.load_state_dict(load_ckpt_model(args.ckpt_dir)["model"])
    return module


def optuna_objective(trial, args):
    param_keys = ["loss_segm_weight", "loss_depth_weight"]
    loss_weights = {k: trial.suggest_float(k, 0.0, 1.0) for k in param_keys}

    args = update_args(args, loss_weights)

    main_components = create_main_components(init_model, args)
    datamodule = main_components["datamodule"]
    module = main_components["module"]

    tools = create_tools(args)
    exp = tools["exp"]
    logger = tools["logger"]

    exp.add_tags([f"trial_{trial.number}"])

    fit_metrics = run_pipe(
        args,
        module,
        datamodule,
        args.num_epochs,
        cfg.device,
        exp=exp,
        logger=logger,
    )

    return np.mean(fit_metrics["val"]["accuracy"])


def run_study(args):
    import optuna
    from optuna.trial import TrialState

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    args = copy.deepcopy(args)
    args.num_epochs = 3

    objective = functools.partial(optuna_objective, args=args)
    study.optimize(objective, n_trials=args.n_trials, timeout=None, n_jobs=args.n_jobs)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("Trail with : \n")
    print("=========================================")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


def create_tools(args):
    exp = create_tracking_exp(args)
    if not args.exp_disabled:
        args.run_name = exp.name
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    extra_tags = [args.model_name]
    exp.add_tags(extra_tags + args.exp_tags)

    log_subdir_name = f"training-{args.model_name}"
    if args.run_name:
        log_subdir_name += f"/{args.run_name}"
    logger = TensorBoardLogger(cfg.log_root_dir, name=log_subdir_name)
    os.makedirs(logger.log_dir, exist_ok=True)
    log_args(args, f"{logger.log_dir}/train_args.yaml", exp=exp)
    return {
        "exp": exp,
        "logger": logger,
    }


def create_main_components(init_model, args):
    datamodule = CityscapesDataModule(
        data_base_dir=cfg.data.data_dir,
        batch_size=args.batch_size,
        do_overfit=args.do_overfit,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    module = init_model(args)
    return {
        "datamodule": datamodule,
        "module": module,
    }


def update_args(args, optimal_params):
    for k, v in optimal_params.items():
        assert hasattr(args, k)
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = parse_args()

    main_components = create_main_components(init_model, args)
    datamodule = main_components["datamodule"]
    module = main_components["module"]

    if args.do_optimize:
        optimal_params = run_study(args)
        update_args(args, optimal_params)
        args.exp_tags += ["best_trial"]

    tools = create_tools(args)
    exp = tools["exp"]
    logger = tools["logger"]

    main_components = create_main_components(init_model, args)
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
