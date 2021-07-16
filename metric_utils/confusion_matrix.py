import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.num_class = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        conf = np.bincount(self.num_class * target.astype(np.int64) + predicted, minlength=self.num_class ** 2) \
            .reshape(self.num_class, self.num_class)
        self.conf += conf

    def value(self):
        return self.conf
