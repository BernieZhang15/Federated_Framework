import os
import cv2
import numpy as np
import tifffile as tiff


def start_point(size, split_size, overlap = 0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


if __name__ == '__main__':
    split_height = 512
    split_width = 512

    overlap = 0.00
    name = 'split'
    suffix = "png"

    image_dir = "D:\\Data\AerialImageDataset\\test\\images"
    # label_dir = "D:\\Data\AerialImageDataset\\train\\gt"

    dest_dir = "D:\\pythonProject\\Data\\Aerial Dataset\\Test"

    cities1 = ["austin", "chicago", "kitsap", "vienna", "tyrol-w"]
    cities2 = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
    for city in cities2:
        count = 0
        image_src = os.path.join(image_dir, city)
        # label_src = os.path.join(label_dir, city)
        image_dest = os.path.join(dest_dir, city, "image")
        # label_dest = os.path.join(dest_dir, city, "label")
        for i in range(1, 37):
            image = image_src + str(i) + ".tif"
            # label = label_src + str(i) + ".tif"

            # if os.path.exists(image) and os.path.exists(label):
            if os.path.exists(image):
                img = cv2.imread(image)
                # label = cv2.imread(label)
            else:
                continue

            img_h, img_w, _ = img.shape
            x_points = start_point(img_h, split_height, overlap)
            y_points = start_point(img_w, split_width, overlap)

            for j in x_points:
                for k in y_points:
                    split1 = img[j: j + split_height, k: k + split_width, :]
                    if not os.path.exists(image_dest):
                        os.makedirs(image_dest)
                    cv2.imwrite(image_dest +'\\{}_{}.{}'.format(name, count, suffix), split1)

                    # split2 = label[j: j + split_height, k: k + split_width,:]
                    # if not os.path.exists(label_dest):
                    #     os.makedirs(label_dest)
                    # cv2.imwrite(label_dest + '\\{}_{}.{}'.format(name, count, suffix), split2)
                    count += 1
