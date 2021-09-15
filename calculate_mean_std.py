import os
import torch
import numpy as np
from data_utils.data_module import read_label2, read_image, CustomImageDataset


if __name__ == "__main__":

    print("Loading AIS data..")

    train_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Central')
    train_img = read_image(os.path.join(train_path, 'image'))
    train_labels = read_label2(os.path.join(train_path, 'label'))


    print("AIS data loading finished")

    train_loader = torch.utils.data.DataLoader(CustomImageDataset(train_img, train_labels), batch_size=1)

    sum_r = 0
    sum_g = 0
    sum_b = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.squeeze(0)
        sum_r += x[0, :, :].mean()
        sum_g += x[1, :, :].mean()
        sum_b += x[2, :, :].mean()

    avg_r = sum_r / len(train_loader)
    avg_g = sum_g / len(train_loader)
    avg_b = sum_b / len(train_loader)

    print(avg_r, avg_b, avg_b)

    num = len(train_loader) * 512 * 512
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.squeeze(0)
        sum_r += np.sum((x[0, :, :] - avg_r).numpy()**2)
        sum_g += np.sum((x[1, :, :] - avg_g).numpy()**2)
        sum_b += np.sum((x[2, :, :] - avg_b).numpy()**2)

    avg_r = sum_r / num
    avg_g = sum_g / num
    avg_b = sum_b / num

    print(avg_r, avg_b, avg_b)





