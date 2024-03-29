# # coding=utf-8
#
# import cv2
# from PIL import Image
# import numpy as np
# import os
#
# Image.MAX_IMAGE_PIXELS = 100000000000
#
#
# def visualize(src_img, dst_img):
#
#     # anno_map = Image.open('../data/test/predict_2019-07-08 20:14:48/image_4_predict.png')
#     anno_map = Image.open(src_img)
#     anno_map = np.asarray(anno_map)
#     print(anno_map.shape)
#
#     B = anno_map.copy()   # 蓝色通道
#     B[B == 1] = 255
#     B[B == 2] = 0
#     B[B == 3] = 0
#
#
#     G = anno_map.copy()   # 绿色通道
#     G[G == 1] = 0
#     G[G == 2] = 255
#     G[G == 3] = 0
#
#
#     R = anno_map.copy()   # 红色通道
#     R[R == 1] = 0
#     R[R == 2] = 0
#     R[R == 3] = 255
#
#
#     anno_vis = np.dstack((B, G, R))
#     anno_vis = cv2.resize(anno_vis, None, fx=0.1, fy=0.1)
#     cv2.imwrite(dst_img, anno_vis)
#
#     print("visulization of {} is done".format(src_img))
#     # cv2.imwrite('../data/vis/image_4_predict.png', anno_vis)
#
#
# if __name__ == "__main__":
#
#     src_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/predict_2019-07-12 10:18:32"
#     dst_path = os.path.join('/home/liuq/wind/agriculture/231717_LQ2/model_train', 'vis', src_path.split('/')[-1])
#     if not os.path.exists(dst_path):
#         os.mkdir(dst_path)
#
#     for file in os.listdir(src_path):
#         src_img = os.path.join(src_path, file)
#         dst_img = os.path.join(dst_path, file)
#         visualize(src_img, dst_img)


# coding=utf-8

import cv2
from PIL import Image
import numpy as np
import os

Image.MAX_IMAGE_PIXELS = 100000000000


def visualize(src_img, dst_img):

    # anno_map = Image.open('../data/test/predict_2019-07-08 20:14:48/image_4_predict.png')
    anno_map = Image.open(src_img)
    anno_map = np.asarray(anno_map)
    print(anno_map.shape)

    B = anno_map.copy()   # 蓝色通道
    B[B == 1] = 255
    B[B == 2] = 0
    B[B == 3] = 0


    G = anno_map.copy()   # 绿色通道
    G[G == 1] = 0
    G[G == 2] = 255
    G[G == 3] = 0


    R = anno_map.copy()   # 红色通道
    R[R == 1] = 0
    R[R == 2] = 0
    R[R == 3] = 255


    anno_vis = np.dstack((B, G, R))
    anno_vis = cv2.resize(anno_vis, None, fx=0.1, fy=0.1)
    cv2.imwrite(dst_img, anno_vis)

    print("visulization of {} is done".format(src_img))
    # cv2.imwrite('../data/vis/image_4_predict.png', anno_vis)


if __name__ == "__main__":

    src_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/image3_predict/xx"
    dst_path = os.path.join('/home/liuq/wind/agriculture/231717_LQ2/model_train/image3_predict', 'vis', src_path.split('/')[-1])
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for file in os.listdir(src_path):
        src_img = os.path.join(src_path, file)
        dst_img = os.path.join(dst_path, file)
        visualize(src_img, dst_img)
