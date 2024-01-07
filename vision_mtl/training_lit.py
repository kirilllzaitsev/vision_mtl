import torch
from pytorch_lightning.loggers import TensorBoardLogger

from vision_mtl.cfg import cfg
from vision_mtl.lit_datamodule import CityscapesDataModule
from vision_mtl.lit_module import LightningPhotopicVisionModule
from vision_mtl.pipeline_utils import build_model, summarize_epoch_metrics
from vision_mtl.utils import parse_args
from vision_mtl.vis_utils import plot_preds


def train_model(
    module: LightningPhotopicVisionModule,
    train_loader,
    val_loader,
    num_epochs,
    device,
    num_classes=cfg.data.num_classes,
):
    logger = TensorBoardLogger("lightning_logs", name="prototyping-lit-module")

    global_step = 0

    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.95, verbose=True
    )

    for epoch in range(num_epochs):
        print(f"### Epoch {epoch+1}/{num_epochs} ###")
        train_loss = 0
        stage = "train"
        for batch in train_loader:
            optimizer.zero_grad()
            batch = module.transfer_batch_to_device(batch, device, 0)
            loss = module.training_step(batch, batch_idx=0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for k, v in module.step_outputs[stage].items():
                print(f"{stage}/{k}", v[-1])
                logger.log_metrics({f"{stage}/{k}": v[-1]}, step=global_step)

            global_step += 1

        train_epoch_metrics = summarize_epoch_metrics(module.step_outputs, stage)
        for k, v in train_epoch_metrics.items():
            print(f"{stage}/epoch_{k}", v)
        logger.log_metrics(train_epoch_metrics, step=epoch)

        val_loss = 0
        stage = "val"

        with torch.no_grad():
            for batch in val_loader:
                optimizer.zero_grad()
                batch = module.transfer_batch_to_device(batch, device, 0)
                loss = module.validation_step(batch, batch_idx=0)

                loss.backward()
                optimizer.step()

                val_loss += loss.item()
                for k, v in module.step_outputs[stage].items():
                    print(f"{stage}_{k}", v[-1])
                    logger.log_metrics({f"{stage}/{k}": v[-1]}, step=global_step)

                global_step += 1

        val_epoch_metrics = summarize_epoch_metrics(module.step_outputs, stage)
        for k, v in val_epoch_metrics.items():
            print(f"{stage}/epoch_{k}", v)
        logger.log_metrics(val_epoch_metrics, step=epoch)

        scheduler.step(val_loss)


if __name__ == "__main__":
    args = parse_args()

    model = build_model(args)

    datamodule = CityscapesDataModule(
        data_base_dir=cfg.data.data_dir,
        batch_size=args.batch_size,
        do_overfit=args.do_overfit,
    )
    datamodule.setup()

    module = LightningPhotopicVisionModule(model=model, lr=args.lr).to(cfg.device)
    train_model(
        module,
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        args.num_epochs,
        cfg.device,
    )

    preds = []
    for pred_Batch in datamodule.predict_dataloader():
        pred_Batch = module.transfer_batch_to_device(pred_Batch, cfg.device, 0)
        preds.append(module.predict_step(pred_Batch, 0, 0))

    plot_preds(args.batch_size, datamodule.predict_dataloader(), preds)
