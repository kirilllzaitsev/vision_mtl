import os
import re
from pathlib import Path

from comet_ml.api import API

root_dir = Path(__file__).parent.parent


def get_latest_ckpt_epoch(
    exp_name,
    model_name_regex=r"model_(\d+)\.pt*",
    project_name="vision-mtl",
):
    api = API(api_key=os.environ["COMET_API_KEY"])
    exp_api = api.get(f"kirilllzaitsev/{project_name}/{exp_name}")
    ckpt_epochs = [
        int(re.match(model_name_regex, x["fileName"]).group(1))
        for x in exp_api.get_asset_list(asset_type="all")
        if re.match(model_name_regex, x["fileName"])
    ]
    return max(ckpt_epochs)


def load_artifacts_from_comet(
    exp_name,
    local_artifacts_dir,
    model_artifact_name,
    args_name_no_ext="train_args",
    session_artifact_name=None,
    project_name="vision-mtl",
    api=None,
    epoch=None,
):
    include_session = session_artifact_name is not None
    args_file_path = os.path.join(local_artifacts_dir, f"{args_name_no_ext}.yaml")

    args_not_exist = not os.path.exists(args_file_path)

    if api is None:
        api = API(api_key=os.environ["COMET_API_KEY"])
    exp_uri = f"kirilllzaitsev/{project_name}/{exp_name}"
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
