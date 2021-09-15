import os
import cv2
import random
import numpy as np

if __name__ == '__main__':
    source_path = "D:\\pythonProject\\Data\\AIS_Data\\Train"
    train_path = "C:\\Users\\berniezhang\\Desktop\\AIS_Data\\Train"
    val_path = "C:\\Users\\berniezhang\\Desktop\\AIS_Data\\Val"
    central_path = "C:\\Users\\berniezhang\\Desktop\\AIS_Data\\Central\\"

    cities = ["berlin", "chicago", "paris", "potsdam", "zurich"]
    city_dict = {"berlin": 2550, "chicago": 9030, "paris": 11319, "potsdam": 539, "zurich": 6895, "tokyo": 35}

    for city in cities:
        print("Processing data from {}".format(city))
        img_index = random.sample(range(city_dict[city]), 20)
        for i, num in enumerate(img_index):
            img_path = os.path.join(source_path, city, "image\\split_{}.png".format(num))
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                continue
            img_dest = os.path.join(train_path, city, "image")
            if not os.path.exists(img_dest):
                os.makedirs(img_dest)
            cv2.imwrite(os.path.join(img_dest, "split_{}.png".format(i + 1)), img)

            label_path = os.path.join(source_path, city, "label\\split_{}.png".format(num))
            if os.path.exists(img_path):
                label = cv2.imread(label_path)
            else:
                continue
            label_dest = os.path.join(train_path, city, "label")
            if not os.path.exists(label_dest):
                os.makedirs(label_dest)
            cv2.imwrite(os.path.join(label_dest, "split_{}.png".format(i + 1)), label)

        val_indexs = np.delete(range(city_dict[city]), img_index)
        val_index = np.random.choice(val_indexs, 20)
        for i, num in enumerate(val_index):
            img_path = os.path.join(source_path, city, "image\\split_{}.png".format(num))
            label_path = os.path.join(source_path, city, "label\\split_{}.png".format(num))
            if os.path.exists(img_path) and os.path.exists(label_path):
                img = cv2.imread(img_path)
                label = cv2.imread(label_path)
            else:
                continue
            img_dest = os.path.join(val_path, city, "image")
            label_dest = os.path.join(val_path, city, "label")
            if not os.path.exists(img_dest):
                os.makedirs(img_dest)
                os.makedirs(label_dest)
            cv2.imwrite(os.path.join(img_dest, "split_{}.png".format(i + 1)), img)
            cv2.imwrite(os.path.join(label_dest, "split_{}.png".format(i + 1)), label)

    count = 121
    for city in cities:
        img_path = os.path.join(train_path, city, "image")
        label_path = os.path.join(train_path, city, "label")
        for i in range(1, 21):
            source_img = os.path.join(img_path, "split_{}.png".format(i))
            source_label = os.path.join(label_path, "split_{}.png".format(i))
            if os.path.exists(source_img):
                img = cv2.imread(source_img)
                label = cv2.imread(source_label)
            else:
                continue
            if not os.path.exists(central_path + "image"):
                os.makedirs(central_path + "image")
                os.makedirs(central_path + "label")
            cv2.imwrite(central_path + "image\\split_" + str(count) + ".png", img)
            cv2.imwrite(central_path + "label\\split_" + str(count) + ".png", label)

            count += 1