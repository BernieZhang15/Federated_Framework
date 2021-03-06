import os
import cv2
import random
import numpy as np

if __name__ == '__main__':
    source_path = "D:\\pythonProject\\Data\\Aerial Dataset\\Train"
    train_path = "C:\\Users\\berniezhang\\Desktop\\Aerial Dataset\\Train"
    val_path = "C:\\Users\\berniezhang\\Desktop\\Aerial Dataset\\Val"
    central_path = "C:\\Users\\berniezhang\\Desktop\\Aerial Dataset\\Central\\"

    cities = ["chicago", "austin", "kitsap", "tyrol-w", "vienna"]
    # cities2 = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]

    for city in cities:
        print("Processing data from {}".format(city))
        # img_index = random.sample(range(3600), 10)
        img_index = []
        if city == 'austin':
            img_index = range(41, 51)
        elif city == "chicago":
            img_index = range(231, 241)
        elif city == "kitsap":
            img_index = range(1841, 1851)
        elif city == "tyrol-w":
            img_index = range(2051, 2061)
        elif city == "vienna":
            img_index = range(261, 271)

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

        val_indexs = np.delete(range(3600), img_index)
        val_index = np.random.choice(val_indexs, 10)
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

    count = 1
    for city in cities:
        img_path = os.path.join(train_path, city, "image")
        label_path = os.path.join(train_path, city, "label")
        for i in range(1, 11):
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