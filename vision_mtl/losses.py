import typing as t

import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """Based on the AdaBins implementation at https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py"""

    def __init__(self, min_depth: float = 1e-3):
        super().__init__()
        self.min_depth = min_depth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: t.Optional[torch.Tensor] = None,
        interpolate: bool = True,
        min_depth: t.Optional[float] = None,
    ) -> torch.Tensor:
        if min_depth is None:
            min_depth = self.min_depth
        if interpolate:
            pred = nn.functional.interpolate(
                pred, target.shape[-2:], mode="bilinear", align_corners=True
            )

        if mask is None:
            mask = target > min_depth

        pred = pred[mask]
        target = target[mask]
        g = torch.log(pred) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
