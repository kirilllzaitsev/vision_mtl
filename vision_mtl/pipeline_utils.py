import torch
from vision_mtl.cfg import cfg
from vision_mtl.models.basic_model import BasicMTLModel
from vision_mtl.models.cross_stitch_model import CSNet, get_model_with_dense_preds
from vision_mtl.models.mtan_model import MTANMiniUnet


def build_model(args):
    if args.model_name == "basic":
        # basic
        model = BasicMTLModel(decoder_first_channel=540, num_decoder_layers=5)
    elif args.model_name == "mtan":
        # MTAN
        map_tasks_to_num_channels = {
            "depth": 1,
            "segm": cfg.data.num_classes,
        }
        model = MTANMiniUnet(
            in_channels=3,
            map_tasks_to_num_channels=map_tasks_to_num_channels,
            task_subnets_hidden_channels=128,
            encoder_first_channel=32,
            encoder_num_channels=4,
        )
    elif args.model_name == "csnet":
        # cross-stitch
        backbone_params = dict(
            encoder_name="timm-mobilenetv3_large_100",
            encoder_weights="imagenet",
            decoder_first_channel=256,
            num_decoder_layers=5,
        )
        models = {
            "depth": get_model_with_dense_preds(
                segm_classes=1, activation=None, backbone_params=backbone_params
            ),
            "segm": get_model_with_dense_preds(
                segm_classes=cfg.data.num_classes,
                activation=None,
                backbone_params=backbone_params,
            ),
        }
        model = CSNet(models, channels_wise_stitching=True)
    else:
        raise NotImplementedError(f"Unknown model name: {args.model_name}")
    return model


def summarize_epoch_metrics(step_outputs, stage):
    step_results = step_outputs[stage]
    loss = torch.mean(torch.tensor([loss for loss in step_results["loss"]]))

    accuracy = torch.mean(
        torch.tensor([accuracy for accuracy in step_results["accuracy"]])
    )

    jaccard_index = torch.mean(
        torch.tensor([jaccard_index for jaccard_index in step_results["jaccard_index"]])
    )

    fbeta_score = torch.mean(
        torch.tensor([fbeta_score for fbeta_score in step_results["fbeta_score"]])
    )

    for key in step_results.keys():
        step_results[key].clear()

    metrics = {
        f"{stage}/loss": loss,
        f"{stage}/accuracy": accuracy,
        f"{stage}/jaccard_index": jaccard_index,
        f"{stage}/fbeta_score": fbeta_score,
    }
    return metrics
