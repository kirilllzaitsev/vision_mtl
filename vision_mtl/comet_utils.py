import os
import re
import typing as t
from pathlib import Path

from comet_ml.api import API

from vision_mtl.cfg import cfg

model_to_exp_name = {
    "basic_non_pretrained": {
        "cityscapes": "supreme_dog_1028",
        "nyuv2": "primary_heron_824",
    },
    "basic": {
        "cityscapes": "institutional_termite_879",
        "nyuv2": "stingy_grain_193",
    },
    "basic_tuned": {
        "cityscapes": "amber_guan_4288",
        "nyuv2": "legitimate_piranha_6638",
    },
    "csnet": {
        "cityscapes": "sensitive_berm_7799",
        "nyuv2": "historic_shrub_7311",
    },
    "mtan": {
        "cityscapes": "white_spice_5973",
        "nyuv2": "valid_preservative_4767",
    },
}


def get_latest_ckpt_epoch(
    exp_name: str,
    model_name_regex: str = r"model_(\d+)\.pt*",
    project_name: str = "vision-mtl",
) -> int:
    """Get the latest checkpoint epoch number from a Comet experiment by applying the model_name_regex to the experiment's assets."""

    api = API(api_key=cfg.logger.api_key)
    exp_api = api.get(f"{cfg.logger.username}/{project_name}/{exp_name}")
    ckpt_epochs = [
        int(re.match(model_name_regex, x["fileName"]).group(1))
        for x in exp_api.get_asset_list(asset_type="all")
        if re.match(model_name_regex, x["fileName"])
    ]
    return max(ckpt_epochs)


def load_artifacts_from_comet(
    exp_name: str,
    local_artifacts_dir: str,
    model_artifact_name: str,
    args_name_no_ext: str = "train_args",
    session_artifact_name: str = None,
    project_name: str = "vision-mtl",
    api: t.Optional[API] = None,
    epoch: t.Optional[int] = None,
) -> dict:
    """Load model, args, and session checkpoints from a Comet experiment."""

    include_session = session_artifact_name is not None
    args_file_path = os.path.join(local_artifacts_dir, f"{args_name_no_ext}.yaml")

    args_not_exist = not os.path.exists(args_file_path)

    if api is None:
        api = API(api_key=cfg.logger.api_key)
    exp_uri = f"{cfg.logger.username}/{project_name}/{exp_name}"
    exp_api = api.get(exp_uri)
    os.makedirs(local_artifacts_dir, exist_ok=True)
    if args_not_exist:
        try:
            asset_id = [
                x
                for x in exp_api.get_asset_list(asset_type="all")
                if f"{args_name_no_ext}" in x["fileName"]
            ][0]["assetId"]
            api.download_experiment_asset(
                exp_api.id,
                asset_id,
                args_file_path,
            )
        except IndexError:
            print(f"No args found with name {args_name_no_ext}")
            args_file_path = None
    if epoch is None:
        epoch = get_latest_ckpt_epoch(exp_name, project_name=project_name)
    model_checkpoint_path = os.path.join(
        local_artifacts_dir, f"{model_artifact_name}_{epoch}.pt"
    )
    weights_not_exist = not os.path.exists(model_checkpoint_path)
    if weights_not_exist:
        asset_id = [
            x
            for x in exp_api.get_asset_list(asset_type="all")
            if f"{model_artifact_name}_{epoch}.pt" in x["fileName"]
        ][0]["assetId"]
        api.download_experiment_asset(
            exp_api.id,
            asset_id,
            model_checkpoint_path,
        )
    results = {
        "checkpoint_path": model_checkpoint_path,
        "args_path": args_file_path,
    }
    if include_session:
        session_checkpoint_path = os.path.join(
            local_artifacts_dir, f"{session_artifact_name}.pt"
        )
        session_not_exist = not os.path.exists(session_checkpoint_path)
        if session_not_exist:
            try:
                asset_id = [
                    x
                    for x in exp_api.get_asset_list(asset_type="all")
                    if f"{session_artifact_name}" in x["fileName"]
                ][0]["assetId"]
                api.download_experiment_asset(
                    exp_api.id,
                    asset_id,
                    session_checkpoint_path,
                )
            except IndexError:
                print(f"No session found with name {session_artifact_name}")
                session_checkpoint_path = None
        if include_session:
            results["session_path"] = session_checkpoint_path
    return results
