import numbers
import typing as t

import torch
from torch.nn import functional as F


def calc_loss(
    out: dict,
    gt_mask: torch.Tensor,
    gt_depth: torch.Tensor,
    segm_criterion: torch.nn.Module,
    depth_criterion: torch.nn.Module,
) -> torch.Tensor:
    segm_logits = out["segm"]
    depth_logits = out["depth"]

    loss_segm = segm_criterion(segm_logits, gt_mask)

    depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
    loss_depth = depth_criterion(depth_predictions, gt_depth)

    loss = loss_segm + loss_depth
    return loss


def summarize_epoch_metrics(
    step_results: dict, metric_name_prefix: t.Optional[str] = None
) -> dict:
    """Average the metrics in step_results and return them as a dict."""

    if metric_name_prefix is None:
        metric_name_prefix = ""
    else:
        metric_name_prefix += "/"
    metrics = {
        f"{metric_name_prefix}{k}": torch.mean(
            torch.tensor([v for v in step_results[k]])
        ).item()
        for k in step_results.keys()
    }
    for key in step_results.keys():
        step_results[key].clear()
    return metrics


def print_metrics(prefix: str, train_epoch_metrics: dict) -> str:
    """Assemble a string out of the metrics and print it."""

    metrics_str = ""
    for k, v in train_epoch_metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() > 1:
                value = v[-1]
            else:
                value = v.item()
        else:
            if isinstance(v, numbers.Number):
                value = v
            else:
                value = v[-1]
        print(f"{prefix}/{k}: {value:.3f} ")
        metrics_str += f"{k}: {value:.3f} "
    return metrics_str
