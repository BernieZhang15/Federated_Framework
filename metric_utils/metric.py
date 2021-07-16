import numpy as np
from metric_utils.confusion_matrix import ConfusionMatrix


class Metric:
    def __init__(self, num_class):
        self.num_class = num_class
        self.conf_matrix = ConfusionMatrix(num_class)
        self.pixel_accurary = []

    def reset(self):
        self.conf_matrix.reset()
        self.pixel_accurary.clear()

    def add_confusion_matrix(self, predicted, target):
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        self.conf_matrix.add(predicted.view(-1), target.view(-1))

    def iou_value(self):
        conf_matrix = self.conf_matrix.value()
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)

    def add_pixel_accuracy(self, pred, target):
        self.pixel_accurary.append(np.mean((pred == target).data.cpu().numpy()))

    def accuracy_value(self):
        avg_acc = np.mean(self.pixel_accurary)
        return avg_acc
