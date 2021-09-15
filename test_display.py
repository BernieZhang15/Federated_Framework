import os
import cv2
import torch
import numpy as np
from neural_nets import SegUNet
from torch.utils.data.dataset import Dataset
from data_utils.data_module import read_label, read_image, CustomImageDataset, convert_backward2

if __name__ == "__main__":

    print("Loading Aerial Dataset..")

    test_path = os.path.join('..', 'Data', 'Aerial Dataset', 'Val', 'chicago')
    test_img = read_image(os.path.join(test_path, 'image'))
    test_labels = read_label(os.path.join(test_path, 'label'))


    print('[INFO] Test and display the performance')

    device = 'cuda'
    model = SegUNet().to(device)
    path = os.path.join("result", "distributed", "model_fedavg_5_80.pth.tar")
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
        test_preds = convert_backward2(test_preds.cpu())
        cv2.imwrite(os.path.join('result', 'Distributed_Image', 'predict_{}.jpg'.format(i)), test_preds)
        label = convert_backward2(label.cpu())
        cv2.imwrite(os.path.join('result', 'Distributed_Image', 'gt_{}.jpg'.format(i)), label)

    global_acc = np.mean(test_seg_scores)

    print('\r[INFO] Testing is completed')
    print("[INFO] Test Seg_Glob_Acc : %.3f" % global_acc)
