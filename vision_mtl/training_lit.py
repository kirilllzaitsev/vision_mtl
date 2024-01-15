"""This module trains the Lightning module in a standard way, i.e., with a vanilla PyTorch training loop.
The reason for this is that the Lightning module does not get optimized withe the PyTorch Lightning Trainer."""

import os

import matplotlib.pyplot as plt
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


def train_model(
    args,
    module: MTLModule,
    train_loader,
    val_loader,
    num_epochs,
    device,
    benchmark_batch=None,
):
    exp = create_tracking_exp(args)
    if not args.exp_disabled:
        args.run_name = exp.name
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    extra_tags = []
    exp.add_tags(extra_tags)

    log_subdir_name = f"training-{args.model_name}"
    if args.run_name:
        log_subdir_name += f"/{args.run_name}"
    logger = TensorBoardLogger(cfg.log_root_dir, name=log_subdir_name)
    os.makedirs(logger.log_dir, exist_ok=True)
    log_args(args, f"{logger.log_dir}/train_args.yaml", exp=exp)

    global_step = 0
    val_step = 0

    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.95, verbose=True
    )

    module.to(device)

    if benchmark_batch is not None:
        benchmark_batch = module.transfer_batch_to_device(benchmark_batch, device, 0)

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")

    for epoch in epoch_pbar:
        print(f"### Epoch {epoch+1}/{num_epochs} ###")
        stage = "train"
        print(f"---{stage.upper()}---")

        train_loss = 0

        train_pbar = tqdm(train_loader, desc="Train Batches")
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
        pbar_postfix = print_metrics(f"epoch/{stage}", train_epoch_metrics)
        epoch_pbar.set_postfix_str(pbar_postfix)
        logger.log_metrics(train_epoch_metrics, step=epoch)

        if (epoch + 1 % args.val_epoch_freq) == 0:
            stage = "val"
            print(f"---{stage.upper()}---")

            if benchmark_batch is not None:
                benchmark_preds = module.predict_step(benchmark_batch, 0, 0)
                fig = plot_preds(
                    batch_size=4,
                    inputs_batch=benchmark_batch,
                    preds_batch=benchmark_preds,
                )
                exp.log_figure("benchmark_preds", fig)
                plt.show()
                plt.close()

            val_loss = 0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Val Batches")
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
            pbar_postfix = print_metrics(f"epoch/{stage}", val_epoch_metrics)
            epoch_pbar.set_postfix_str(pbar_postfix)

            logger.log_metrics(val_epoch_metrics, step=epoch)

            scheduler.step(val_loss)

        if (epoch + 1) % args.save_epoch_freq == 0 or epoch == args.num_epochs - 1:
            save_path_model = os.path.join(logger.log_dir, f"model_{epoch}.pt")
            save_path_session = os.path.join(logger.log_dir, "session.pt")
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


def predict(predict_dataloader, module, batch_size, device, do_plot_preds=False):
    preds = []
    module.to(device)
    for pred_batch in tqdm(predict_dataloader, desc="Predict Batches"):
        pred_batch = module.transfer_batch_to_device(pred_batch, device, 0)
        batch_preds = module.predict_step(pred_batch, 0, 0)
        preds.append(batch_preds)
        if do_plot_preds:
            plot_preds(batch_size, pred_batch, batch_preds)
    return preds


def init_model(args):
    model = build_model(args)
    module = MTLModule(model=model, lr=args.lr)
    if args.ckpt_dir:
        module.load_state_dict(load_ckpt_model(args.ckpt_dir)["model"])
    return module

if __name__ == "__main__":
    args = parse_args()

    model = build_model(args)

    datamodule = CityscapesDataModule(
        data_base_dir=cfg.data.data_dir,
        batch_size=args.batch_size,
        do_overfit=args.do_overfit,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    module = MTLModule(model=model, lr=args.lr)
    if args.ckpt_dir:
        module.load_state_dict(load_ckpt_model(args.ckpt_dir)["model"])
    train_model(
        args,
        module,
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        args.num_epochs,
        cfg.device,
        benchmark_batch=datamodule.benchmark_batch,
    )

    preds = predict(
        datamodule.predict_dataloader(),
        module,
        args.batch_size,
        cfg.device,
        args.do_plot_preds,
    )
