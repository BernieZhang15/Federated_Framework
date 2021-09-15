import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """

    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        top = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        bottom = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth
        loss = 1 - top / bottom

        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, num_class, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.device = "cuda"

    def forward(self, predict, target):
        total_loss = 0
        dice = BinaryDiceLoss(**self.kwargs)
        predict = F.softmax(predict, dim=1)
        target = torch.unsqueeze(target, dim=1)
        target = make_one_hot(target, self.num_class).to(self.device)

        for i in range(self.num_class):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/self.num_class

if __name__=="__main__":
    input = torch.randn(3,2,3,3)
    target = torch.randint(0, 2, (3,3,3))
    dice = DiceLoss()
    result = dice(input, target)