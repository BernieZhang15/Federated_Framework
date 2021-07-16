import torch.nn as nn
import torch
import numpy as np
from loss_utils.cross_entropy_loss import CrossEntropy2d


class TverskyCrossEntropyDiceWeightedLoss(nn.Module):
    def __init__(self, num_class, device):
        super(TverskyCrossEntropyDiceWeightedLoss, self).__init__()
        self.num_class = num_class
        self.device = device
        self.crossEntropy = CrossEntropy2d()

    def tversky_loss(self, pred, target, alpha=0.5, beta=0.5):
        target_oh = torch.eye(self.num_class)[target.squeeze(1)]  # convert to one_hot coding
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        probs = torch.nn.functional.softmax(pred, dim=1)
        target_oh = target_oh.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        inter = torch.sum(probs * target_oh, dims)
        fps = torch.sum(probs * (1 - target_oh), dims)
        fns = torch.sum((1 - probs) * target_oh, dims)
        t = (inter / (inter + (alpha * fps) + (beta * fns))).mean()
        return 1 - t

    def class_dice(self, pred, target, epsilon=1e-6):
        pred_class = torch.argmax(pred, 1)
        dices = np.ones(self.num_class)
        for c in range(self.num_class):
            p = (pred_class == c)
            t = (target == c)
            inter = (p * t).sum().float()
            union = p.sum() + t.sum() + epsilon
            d = 2 * inter / union
            dices[c] = 1 - d
        return torch.from_numpy(dices).float()

    def forward(self, pred, target, cross_entropy_weight=0.5, tversky_weight=0.5):
        if cross_entropy_weight + tversky_weight != 1:
            raise ValueError('Cross Entropy weight and Tversky weight should sum to 1')
        ce_loss = self.crossEntropy(pred, target, self.class_dice(pred, target).to(self.device))
        tv_loss = self.tversky_loss(pred, target)
        loss = (cross_entropy_weight * ce_loss) + (tversky_weight * tv_loss)
        return loss
