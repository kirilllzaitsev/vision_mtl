from torch.nn import functional as F


def calc_loss(out, gt_mask, gt_depth, segm_criterion, depth_criterion):
    segm_logits = out["segm"]
    depth_logits = out["depth"]

    loss_segm = segm_criterion(segm_logits, gt_mask)

    depth_predictions = F.sigmoid(input=depth_logits).permute(0, 2, 3, 1)
    loss_depth = depth_criterion(depth_predictions, gt_depth)

    loss = loss_segm + loss_depth
    return loss
