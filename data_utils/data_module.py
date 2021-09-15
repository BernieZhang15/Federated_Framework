import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset

def read_image(path):
    img_sets = []
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4323, 0.3882, 0.3882), (0.0421, 0.0348, 0.0348))])
    files = os.listdir(path)
    for file in files:
        img = Image.open(os.path.join(path, file))
        img = np.array(transform(img))
        img_sets.append(img)
    return img_sets

def read_label(path):
    img_sets = []
    transform = transforms.Compose([transforms.Grayscale(1)])
    files = os.listdir(path)
    for file in files:
        img = Image.open(os.path.join(path, file))
        img = transform(img)
        img = np.array(img)
        img = convert(img)
        img_sets.append(img)
    return img_sets

def convert(image):
    img_h, img_w = image.shape
    img = np.zeros((img_h, img_w))
    index1 = np.where(image == 29) # blue
    img[index1[0], index1[1]] = 0
    index2 = np.where(image == 76) # red
    img[index2[0], index2[1]] = 1
    index3 = np.where(image == 255) # white
    img[index3[0], index3[1]] = 2
    return img

def read_label2(path):
    img_sets = []
    files = os.listdir(path)
    transform = transforms.Compose([transforms.Grayscale(1)])
    for file in files:
        img = Image.open(os.path.join(path, file))
        img = transform(img)
        img = np.array(img)
        img = convert2(img)
        img_sets.append(img)
    return img_sets

def convert_backward(image):
    y = np.zeros((512, 512, 3))
    index1 = torch.where(image == 0)
    y[index1[1], index1[2]] = [0, 0, 255]
    index2 = torch.where(image == 1)
    y[index2[1], index2[2]] = [255, 0, 0]
    index3 = torch.where(image == 2)
    y[index3[1], index3[2]] = [255, 255, 255]
    return y

def convert2(image):
    img_h, img_w = image.shape
    img = np.zeros((img_h, img_w))
    # black
    index1 = np.where(image == 0)
    img[index1[0], index1[1]] = 0
    # white
    index2 = np.where(image == 255)
    img[index2[0], index2[1]] = 1
    return img

def convert_backward2(image):
    y = np.zeros((512, 512, 3))
    index1 = torch.where(image == 0)
    y[index1[1], index1[2]] = [0, 0, 0]
    index2 = torch.where(image == 1)
    y[index2[1], index2[2]] = [255, 255, 255]
    return y


class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
          assert len(inputs) == len(labels)
          self.inputs = torch.tensor(inputs)
          self.labels = torch.tensor(labels).long()
          self.transforms = transforms

    def __getitem__(self, index):
          img, label = self.inputs[index], self.labels[index]

          if self.transforms is not None:
            img = self.transforms(img)

          return (img, label)

    def __len__(self):
          return len(self.inputs)


if __name__ == "__main__":
    train_path = os.path.join('..', '..', 'Data', 'Aerial Dataset', 'Train', "austin")
    img_sets = read_image(os.path.join(train_path, 'image'))
    label_sets = read_label2(os.path.join(train_path, 'label'))
