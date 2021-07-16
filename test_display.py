import os
import cv2
import torch
import numpy as np
from neural_nets import SegUNet
from torch.utils.data.dataset import Dataset
from data_utils.data_module import read_label, read_image, CustomImageDataset

if __name__ == "__main__":

    print("Loading AIS data..")

    test_path = os.path.join('..', 'Data', 'AIS_Data', 'Val')
    test_img = read_image(os.path.join(test_path, 'image'))
    test_labels = read_label(os.path.join(test_path, 'label'))


    def convert_backward(image):
        y = np.zeros((512, 512, 3))
        index1 = torch.where(image == 0)
        y[index1[1], index1[2]] = [0, 0, 255]
        index2 = torch.where(image == 1)
        y[index2[1], index2[2]] = [255, 255, 255]
        index3 = torch.where(image == 2)
        y[index3[1], index3[2]] = [255, 0, 0]
        return y

    def convert_backward2(image):
        y = np.zeros((512, 512, 3))
        index1 = torch.where(image == 0)
        y[index1[1], index1[2]] = [0, 0, 0]
        index2 = torch.where(image == 1)
        y[index2[1], index2[2]] = [255, 255, 255]
        return y

    print('[INFO] Test and display the performance')

    device = 'cuda'
    model = SegUNet().to(device)
    path = os.path.join("result", "distributed", "model_checkpoint_2.pth.tar")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(CustomImageDataset(test_img, test_labels),
                                                  batch_size=1, shuffle=False)

    test_seg_scores = []

    for i, (img, label) in enumerate(test_dataloader):
        img = img.to(device)
        label = label.to(device).long()

        print('[PROGRESS] Processing images: %i of %i  ' % (i + 1, len(test_dataloader)), end='\r')

        seg_outputs = model(img)

        _, test_preds = torch.max(seg_outputs, 1)

        test_seg_scores.append(np.mean((test_preds == label).data.cpu().numpy()))

        # convert back to image
        test_preds = convert_backward(test_preds.cpu())
        cv2.imwrite(os.path.join('result', 'Distributed_Image', 'predict_{}.jpg'.format(i)), test_preds)
        label = convert_backward(label.cpu())
        cv2.imwrite(os.path.join('result', 'Distributed_Image', 'gt_{}.jpg'.format(i)), label)

    global_acc = np.mean(test_seg_scores)

    print('\r[INFO] Testing is completed')
    print("[INFO] Test Seg_Glob_Acc : %.3f" % global_acc)
