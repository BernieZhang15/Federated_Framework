import torch.nn.functional as F
import torch.nn as nn


class CrossEntropy2d(nn.Module):
    def __init__(self):
        super(CrossEntropy2d, self).__init__()

    def forward(self, seg_preds, seg_targets, weight=None):

        n, c, h, w = seg_preds.size()

        # Calculate segmentation loss
        seg_inputs = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_inputs = seg_inputs.view(-1, c)

        seg_targets = seg_targets.view(-1)

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.cross_entropy(seg_inputs, seg_targets, weight)
        return seg_loss




