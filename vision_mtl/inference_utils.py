import torch
import torch.nn.functional as F


def get_segm_preds(
    segm_validity_mask: torch.Tensor, segm_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get segmentation predictions for desired areas defined by the mask based on logits."""

    segm_pred_probs = F.softmax(input=segm_logits, dim=1)
    segm_predictions_classes = torch.argmax(segm_pred_probs, dim=1)

    segm_validity_mask_with_channels = segm_validity_mask.unsqueeze(1).repeat(
        1, 19, 1, 1
    )
    segm_pred_probs = segm_pred_probs[segm_validity_mask_with_channels].reshape(-1, 19)
    segm_predictions_classes = segm_predictions_classes[segm_validity_mask]
    return segm_pred_probs, segm_predictions_classes
