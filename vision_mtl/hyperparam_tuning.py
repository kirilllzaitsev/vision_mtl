import argparse
import copy
import functools
import typing as t

import numpy as np
import optuna
from optuna.trial import TrialState

from vision_mtl.cfg import DataConfig, cfg
from vision_mtl.training_lit import run_pipe
from vision_mtl.utils.pipeline_utils import (
    create_main_components,
    create_tools,
    init_model,
)
from vision_mtl.utils.utils import update_args


def optuna_objective(
    trial: optuna.Trial, args: argparse.Namespace, data_cfg: DataConfig
) -> float:
    """Optuna objective function that is evaluated at each trial. Returns the validation accuracy."""
    param_keys = ["loss_segm_weight", "loss_depth_weight"]
    loss_weights = {k: trial.suggest_float(k, 0.0, 1.0) for k in param_keys}

    args = update_args(args, loss_weights)

    main_components = create_main_components(init_model, args, data_cfg)
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

    exp.end()

    return np.mean(fit_metrics["val"]["val/accuracy"])


def run_study(args: argparse.Namespace, data_cfg: DataConfig) -> t.Dict[str, float]:
    """Run optuna study to find best hyperparameters. In this case, it picks those that maximize the validation accuracy."""

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    args = copy.deepcopy(args)
    args.num_epochs = 3

    objective = functools.partial(optuna_objective, args=args, data_cfg=data_cfg)
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
