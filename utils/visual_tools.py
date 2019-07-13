# coding:utf-8
# 2019.06.22 李波四川大学614实验室

import torch
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


# Visualize the segmentation and count the proportion of each of the flue-cured tobacco, corn, barley rice, and other regions
def labelVisualiztion(img_ori, img_label):
    img = img_ori.copy()
    label_class = ['other', 'tobacco', 'corn', 'barleyrice']
    # 使用lambda来定义简单的函数
    counter = defaultdict(lambda: 0)
    lbl_pixel = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]).astype(np.uint8)

    for j in range(1, 4, 1):  # 4 pixels
        for i in range(3):
            img[:, :, i] = np.where(img_label[:, :, i] == j, lbl_pixel[j, i], img[:, :, i])

    total = img_label.shape[0] * img_label.shape[1]
    # print("total:", total)
    for k, cls in enumerate(label_class):
        mask = (img_label[:, :, 0] == k)
        arr_new = img_label[:, :, 0][mask]
        count_tmp = arr_new.size
        # print("count{}:".format(str(k)),count_tmp)
        counter[cls] = count_tmp

        # Display the statistics of the three area distributions on the image
        cv2.putText(img, '{}:{:.3f}'.format(cls, count_tmp / total), (20, 30 + k * 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

    return img


# Img_concatenate is the image stitching function
def img_concatenate(image1, image2):
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    # print(image1.dtype)
    if c1 != c2:
        print("channels NOT match, cannot merge")
        return
    elif c1 == c2:
        if h1 == h2:  # same height
            tmp = np.zeros((h1, 10, c1)) + 255  # ,np.uint8)
            tmp = tmp.astype(np.uint8)
            image1 = np.concatenate((image1, tmp), axis=1)
            image3 = np.hstack([image1, image2])
            return image3
        elif w1 == w2:  # same width
            pass
        else:
            print("shape dismatch！,can not merge!")
            return


# Process all images and save
def processedImgSaving(label_path, img_path, saved_path):
    cnt = 0
    label_imgs = sorted(os.listdir(label_path), key=lambda x: int(x.split(".")[0].split("_")[-1]))
    origin_imgs = sorted(os.listdir(img_path), key=lambda x: int(x.split(".")[0].split("_")[-1]))

    for label, img in zip(label_imgs, origin_imgs):
        cnt += 1
        # if cnt>396 and cnt <408:

        label_name = os.path.join(label_path, label)
        img_name = os.path.join(img_path, img)

        img_label = cv2.imread(label_name)
        img_origin = cv2.imread(str(img_name))

        # label visualiztion
        img_concat = labelVisualiztion(img_origin, img_label)
        image = img_concatenate(img_concat, img_origin)
        img = "visual_" + img
        img_paths = os.path.join(saved_path, img)

        print("saving image :{}".format(img_paths))
        cv2.imwrite(img_paths, image)
        # cv2.imshow(img_name, image)
        # cv2.waitKey(0)


# Analysis and statistical training samples
def statisticalAnalysis(label_path):
    label_imgs = sorted(os.listdir(label_path), key=lambda x: int(x.split(".")[0].split("_")[-1]))

    label_class = ['other', 'tobacco', 'corn', 'barleyrice']

    tobacco_area = 0
    corn_area = 0
    barley_area = 0
    other_area = 0

    tobacco_num = 0
    corn_num = 0
    barley_num = 0
    other_num = 0

    total_num = len(label_imgs)
    print("total area assess:", total_num * 500 * 500)
    total_area = 0
    cnt = 0
    for label in label_imgs:
        cnt += 1
        # if cnt>397 and cnt<409:
        label_img_path = os.path.join(label_path, label)
        label_img = cv2.imread(label_img_path)
        label_unique_pixel = np.unique(label_img)
        # print("unique pixel:",label_unique_pixel)

        label_imgb = label_img[:, :, 0]

        tobacco_area += label_imgb[label_imgb == 1].size
        corn_area += label_imgb[label_imgb == 2].size
        barley_area += label_imgb[label_imgb == 3].size
        other_area += label_imgb[label_imgb == 0].size
        total_area += tobacco_area + corn_area + barley_area + other_area

        tobacco_num += (np.array(np.where(label_unique_pixel == 1))).size
        corn_num += label_unique_pixel[label_unique_pixel == 2].size
        barley_num += (np.array(np.where(label_unique_pixel == 3))).size
        other_num += (np.array(np.where(label_unique_pixel == 0))).size

    class_num_proportion = [other_num, tobacco_num, corn_num, barley_num]

    class_area_proportion = [other_area, tobacco_area, corn_area, barley_area]

    # 返回柱状图数据
    return class_num_proportion, class_area_proportion, total_num, total_area


def barchart(data_list, tik_labels, title="dataset1 area proportion", xlabel="class", ylabel="number",
             color=('black', 'b', 'g', 'r')):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.bar(range(len(data_list)), data_list, tick_label=tik_labels,
            color=color)
    for x, y in enumerate(data_list):
        plt.text(x, y + 80, '%s' % y, ha='center', va='bottom')
    plt.savefig('{}_bar.jpg'.format(title))
    plt.show()


def piechart(sizes, tik_labels, explode=(0, 0, 0, 0.2), title="dataset1 area proportion"):
    # 设置绘图对象的大小
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    # explode = (0, 0.1, 0,0)  # 0.1表示将Hogs那一块凸显出来
    ax1.pie(sizes, explode=explode, labels=tik_labels, autopct='%1.1f%%', shadow=False,
            startangle=90)  # startangle表示饼图的起始角度
    ax1.axis('equal')  # 加入这行代码即可！
    plt.savefig('{}_pie.jpg'.format(title))
    plt.show()


def statisticalCharts(label_path1, n):
    class_num_proportion, class_area_proportion, total_num, total_area = statisticalAnalysis(label_path1)

    tik_labels_nums = ['other_num', 'tobacco_num', 'corn_num', 'barley_num']

    tik_labels_area = ['other_area', 'tobacco_area', 'corn_area', 'barley_area']

    barchart(class_num_proportion, tik_labels_nums, "dataset{} nums proportion".format(n), "class", "number")
    barchart(class_area_proportion, tik_labels_area, "dataset{} area proportion".format(n), "class", "area")

    piechart(class_num_proportion, tik_labels_nums, title="dataset{} nums proportion".format(n))
    piechart(class_area_proportion, tik_labels_area, title="dataset{} area proportion".format(n))


def anALl(label_path):
    class_num_proportion1, class_area_proportion1, total_num1, total_area1 = statisticalAnalysis(label_path.format(1))
    class_num_proportion2, class_area_proportion2, total_num2, total_area2 = statisticalAnalysis(label_path.format(2))

    num1 = np.array(class_num_proportion1)
    area1 = np.array(class_area_proportion1)

    num2 = np.array(class_num_proportion2)
    area2 = np.array(class_area_proportion2)

    num = num1 + num2
    area = area1 + area2

    tik_labels_nums = ['other_num', 'tobacco_num', 'corn_num', 'barley_num']

    tik_labels_area = ['other_area', 'tobacco_area', 'corn_area', 'barley_area']

    piechart(num, tik_labels_nums, title="all nums proportion")
    piechart(area, tik_labels_area, title="all area proportion")


if __name__ == "__main__":
    label_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/image3_predict/predict_2019-07-12 14:31:37"
    # label_path2 = "/media/lb/学习/数据集/data/train_img_label_split{}"
    img_path = "/home/liuq/wind/agriculture/231717_LQ2/data/test_img3_split"
    # img_path2 = "train_img_split{}"
    saved_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/image3_predict/vis"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # saved_path2 = "./train_img_visualization{}"

    processedImgSaving(label_path, img_path, saved_path)
    # statisticalCharts(label_path.format(i),i)

    anALl(label_path)

