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
def split_image(number, size):
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

    num = 0
    for r in range(rownum):      # 行循环
        for c in range(colnum):  # 列循环
            num = num + 1
            box = (c * size, r * size, (c + 1) * size, (r + 1) * size)  # 左上角，右下角坐标
            print('c={} r={}'.format(c, r))

            dst_path = '/home/liuq/wind/agriculture/data{}/'.format(size)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            data_path = os.path.join(dst_path, n + '_{}.png'.format(num))
            img.crop(box).save(data_path)

            dst_path = '/home/liuq/wind/agriculture/label{}/'.format(size)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            label_path = os.path.join(dst_path, n + '_{}.png'.format(num))
            img_label.crop(box).save(label_path)

    print('图片切割完毕，共生成 %s 张小图片。' % num)
    print("切割耗时：%s" % (time.time() - timestart))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 重新命名
def rename(imgPath):
    filelist = os.listdir(imgPath)
    filelist = sorted(filelist, key=lambda x: int(x.split('.')[0]))

    counter = 1  # 首张图片为1号
    for item in filelist:
        extimag = item.split('.')[-1]  # 得到图片后缀
        src = os.path.join(os.path.abspath(imgPath), item)
        dst = os.path.join(os.path.abspath(imgPath), 'image' + '_' + str(counter) + '.' + extimag)
        os.rename(src, dst)
        print('converting %s to %s ...' % (src, dst))
        # print('filename={} file_counter={}'.format(item, counter))
        # print("filename=%s file_counter=%s" % (item, counter))
        counter = counter + 1
    print("重命名图片总计=%s 张" % (counter - 1))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 找出白色像素点高于一半的图片，并返回一个列表
def filter_Image(FilePath):
    list = []
    image_list = os.listdir(FilePath)
    # image_list = sorted(image_list , key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for k in image_list:
        img = Image.open(FilePath + '/' + k)
        img_array = img.load()
        count = 0
        flag = 0
        x, y = 1024, 1024
        for i in range(0, 1024):
            for j in range(0, 1024):
                if (img_array[i, j] == (0, 0, 0, 0)):  # RGBA
                    count += 1
                    if count > 524288:  # 占1/2就保存
                        list.append(k)
                        flag = 1
                        break
            if flag == 1:  # 跳出双层循环
                break
    return list


# 6. 将图片的列表转为对应的标签的列表 如image_1_1.png ---》 image_1_label_1.png
def imgList_To_Label(imgList):
    # 此处必须是深拷贝，直接赋值会影响两个list，切记
    labelList = copy.deepcopy(imgList)
    n = len(labelList)
    for i in range(0, n):
        labelList[i] = labelList[i].split('_')[0] + '_' + labelList[i].split('_')[1] + \
                       '_label_' + labelList[i].split('_')[2]
    return labelList


# 7. 删除列表中img 和label的图片
def delete_ImgLabel(img_path, img_list, label_path, label_list):
    for i in img_list:
        imgSubPath = os.path.join(os.getcwd(), img_path, i)
        print(imgSubPath)
        os.remove(imgSubPath)

    for j in label_list:
        labelSubpath = os.path.join(os.getcwd(), label_path, j)
        print(labelSubpath)
        os.remove(labelSubpath)


# 8 对过滤的图片重命名
def rename_Filter(imgPath, labelPath):
    img_file = os.listdir(imgPath)
    # 按照文件夹的顺序命名，不然会被打乱
    img_file.sort(key=lambda x: int(x[8:-4]))
    img_file_counter = 0
    for img in img_file:
        img_file_counter += 1
        os.rename(imgPath + '/' + img,
                  imgPath + '/' + img.split('.')[0] + '_' + str(img_file_counter) + '.' + img.split('.')[1])

    label_file = os.listdir(labelPath)
    # 按照文件夹的顺序命名，不然会被打乱
    label_file.sort(key=lambda x: int(x[14:-4]))
    label_file_counter = 0
    for label in label_file:
        label_file_counter += 1
        os.rename(labelPath + '/' + label,
                  labelPath + '/' + label.split('.')[0] + '_' + str(label_file_counter) + '.' + label.split('.')[1])

# 8 图片重命名
def Rename_file(imgPath, number):
    img_file = os.listdir(imgPath)
    # 按照文件夹的顺序命名，不然会被打乱
    img_file.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    img_file_counter = number
    for img in img_file:
        os.rename(imgPath + '/' + img, imgPath + '/' + str(img_file_counter) + '.' + img.split('.')[-1])
        img_file_counter += 1
    return img_file_counter

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    print("----数据预处理任务开始------")
    start = time.time()
    # image_1.png ：47161 * 50141, PNG, RGBA
    # image_2.png ：77470 * 46050, PNG, RGBA
    # ==========================================================================
    # 1. 裁剪图片（截图图片的一部分）--不修改原始数据
    # split_image(1, 512)   # 图片切割完毕，共生成 92 * 97=8924 张小图, 耗时：409.236145734787
    split_image(2, 512)   # 图片切割完毕，共生成 150*88=13439 张小图, 耗时：688.1491384506226

    # ==========================================================================
    # 4. 筛选图片，找到白色像素点个数高于一半的图片，并将这些图片的名字返回为一个列表
    # img_delete_list = filter_Image(FilePath=train_Path2)
    # print("stage1------------over")

    # 5. 将图片的列表转为对应的标签的列表
    # label_delete_list = imgList_To_Label(img_delete_list)
    # print("stage2------------over")

    # 6. 删除列表中img 和label的图片
    # delete_ImgLabel(train_Path2, img_delete_list, train_label_Path2, label_delete_list)
    # print("stage3------------over")

    # 7. 对过滤的图片重命名
    # rename_Filter(train_Path2, train_label_Path2)
    # print("stage4------------over")
    # print(time.time() - start)

    # 8. 重新命名
    # num = Rename_file(train_Path1, 1)
    # Rename_file(train_label_Path1, 1)
    #
    # Rename_file(train_Path2, num)
    # Rename_file(train_label_Path2, num)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
