from torch.nn import functional as F

from vision_mtl.train_prototyping import get_segm_preds


def calc_loss(
    out, gt_mask, segm_validity_mask, gt_depth, segm_criterion, depth_criterion
):
    segm_logits = out["segm"]
    depth_logits = out["depth"]

    segm_pred, _ = get_segm_preds(segm_validity_mask, segm_logits)
    gt_mask = gt_mask[segm_validity_mask]

    loss_segm = segm_criterion(segm_pred, gt_mask)

    depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
    loss_depth = depth_criterion(depth_predictions, gt_depth)

    loss = loss_segm + loss_depth
    return loss
