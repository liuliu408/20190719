# coding:utf-8
# 2019.06.20 刘强 四川大学614实验室

import os
import numpy as np
from PIL import Image
import time
import copy

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 切割图片（大图切分成数个小图），
# 每行的最后一张分割图的大小还要加上剩余的部分的列像素
# 每列的最后一张分割图的大小还要加上剩余的部分的行像素
def split_image(number, size, ratio = 0.5):
    timestart = time.time()
    print('开始处理图片切割, 请稍候...')
    # --------------------------------------------------------------------------
    imagepath = '/home/liuq/wind/agriculture/origin_data/train_data/image_{}.png'.format(number)
    n = os.path.basename(imagepath)[:-4]
    labelname = '/home/liuq/wind/agriculture/origin_data/train_data/' + n + '_label.png'

    Image.MAX_IMAGE_PIXELS = 100000000000
    img = Image.open(imagepath)
    img_label = Image.open(labelname)

    Width, Height = img.size  # 得到图片的尺寸：宽、高像素
    print(imagepath)          # 显示图片路径
    print('原图信息: %s * %s, %s, %s' % (Width, Height, img.format, img.mode))
    print('分割像素：{}*{}'.format(size, size))

    rownum = Height // size
    colnum = Width // size
    print('分割行列：{} * {}'.format(colnum, rownum))

    nullthresh = size * size * ratio

    num = 0
    for r in range(rownum):      # 行循环
        for c in range(colnum):  # 列循环
            box = (c * size, r * size, (c + 1) * size, (r + 1) * size)  # 左上角，右下角坐标
            img1 = img.crop(box)
            img1 = np.asanyarray(img1)

            # 目的是找出图片里面的空白区域！
            if (img1[:, :, 3] == 0).sum() > nullthresh:   # 切片后，利用广播特性，将第3个通道的每一个像素点依次求和！
                continue
            num = num + 1

            print('c={} r={}'.format(c, r))
            dst_path = '/home/liuq/wind/agriculture/data{}/'.format(size)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            data_path = os.path.join(dst_path, n + '_{}.png'.format(num))

            # _img1 = np.zeros((size, size, 3)).astype(np.int32)
            # _img1[:, :, :] = img1[:, :, :3]                       # 只取前面三个通道数据：RGB，由于训练只需要3通道RGB图
            # _img1 = Image.fromarray(np.uint8(_img1))
            # _img1.save(data_path)
            # 下面2行 和 上面4行 代码等效
            img1 = Image.fromarray(img1).convert('RGB')             # 只取前面三个通道数据：RGB
            img1.save(data_path)

            dst_path = '/home/liuq/wind/agriculture/label{}/'.format(size)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            label_path = os.path.join(dst_path, n + '_{}.png'.format(num))

            _img_label = img_label.crop(box)
            _img_label = np.asanyarray(_img_label)

            _img_label = Image.fromarray(_img_label).convert('L')   # 转为灰度图
            _img_label.save(label_path)

    print('图片切割完毕，共生成 %s 张小图片。' % num)
    print("切割耗时：%s" % (time.time() - timestart))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    print("----数据预处理任务开始------")
    start = time.time()
    # image_1.png ：47161 * 50141, PNG, RGBA
    # image_2.png ：77470 * 46050, PNG, RGBA

    split_image(1, 1024, 0.4)   # 图片切割完毕，共生成 92 * 97=8924 张小图, 耗时：409.236145734787
    split_image(2, 1024, 0.4)   # 图片切割完毕，共生成 150*88=13439 张小图, 耗时：688.1491384506226
