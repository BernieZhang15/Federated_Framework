import torch.nn.functional as F
import torch.nn as nn


class kl_divergence(nn.Module):
    def __init__(self):
        super(kl_divergence, self).__init__()

    def forward(self, seg_preds, seg_targets):

        # Calculate segmentation loss
        seg_preds = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_targets = seg_targets.transpose(1, 2).transpose(2, 3).contiguous()

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.kl_div(seg_preds.log_softmax(dim=-1), seg_targets.softmax(dim=-1), reduction='mean')

        return seg_loss
