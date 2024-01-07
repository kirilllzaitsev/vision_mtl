import typing as t

import pytorch_lightning as pl

from vision_mtl.cfg import cfg
from vision_mtl.lit_datamodule import CityscapesDataModule
from vision_mtl.lit_module import LightningPhotopicVisionModule
from vision_mtl.models.basic_model import BasicMTLModel


def main(
    callbacks: list,
    model,
    logger,
) -> None:
    # Trainer
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="auto",
        strategy="auto",
        devices="auto",
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=80,
        min_epochs=35,
        log_every_n_steps=1,
    )

    # Datamodule
    datamodule = CityscapesDataModule(
        data_base_dir=cfg.data.data_dir,
    )

    # LightningModule
    lightning_model = LightningPhotopicVisionModule(
        model=model,
        lr=3e-4,
    )

    # Start training
    trainer.fit(model=lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    main(
        callbacks=[],
        model=LightningPhotopicVisionModule(
            model=BasicMTLModel(),
        ),
        logger=None,
    )
