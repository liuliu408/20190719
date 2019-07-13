# coding:utf-8
# 2019.06.20 刘强 四川大学614实验室
import os
import time
import copy
import numpy as np
from PIL import Image
import random

# 数据路径
train_Path = '/home/bobo/614/Tianchi/Agriculture_data/modified_data1024_1024/train_img/'
train_lablePath = '/home/bobo/614/Tianchi/Agriculture_data/modified_data1024_1024/train_img_label/'

# 数据保存路径
save_train_Path = '/home/bobo/614/Tianchi/Agriculture_data/modified_data1024_1024/train_img_ext/'
save_train_lablePath = '/home/bobo/614/Tianchi/Agriculture_data/modified_data1024_1024/train_img_label_ext/'

# # 数据路径
# train_Path = '/home/bobo/614/liuq/data/train_img/'
# train_lablePath = '/home/bobo/614/liuq/data/train_img_label/'
#
# # 数据保存路径
# save_train_Path = '/home/bobo/614/liuq/data/train_img_ext/'
# save_train_lablePath = '/home/bobo/614/liuq/data/train_img_label_ext/'

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. 图片对切分组合（图1与图片2左右部分对调）Left & Right
def image_split_LR(train_Path, train_lablePath,save_train_Path, save_train_lablePath):
    if not os.path.exists(save_train_Path):
     os.makedirs(save_train_Path)
    if not os.path.exists(save_train_lablePath):
     os.makedirs(save_train_lablePath)

    trainlist1 = os.listdir(train_Path)
    imgNum = int(len(trainlist1))
    trainlist1.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    trainlist2 = copy.deepcopy(trainlist1)  # 深拷贝
    random.shuffle(trainlist2)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱

    for i in range(imgNum):
        img1 = Image.open(os.path.join(train_Path, trainlist1[i]))          # 读取图片
        img2 = Image.open(os.path.join(train_Path, trainlist2[i]))
        Width, Height = img1.size    # 得到图片的尺寸：宽、高像素
        print(os.path.join(train_Path, trainlist1[i]))
        print(os.path.join(train_Path, trainlist2[i]))
        ext = trainlist1[i].split('.')[-1]         # 得到图片格式后缀：png

        box1 = (0, 0, int(Width/2), Height)         # 左部分
        box2 = (int(Width/2), 0, Width, Height)     # 右部分
        region1 = img1.crop(box1)    # 取出图像块左部分
        region2 = img2.crop(box2)    # 取出图像块右部分
        img1.paste(region2, box2)    # 粘贴图像块
        img2.paste(region1, box1)
        img1.save(os.path.join(save_train_Path, str(i+1+imgNum) + '.' + ext))        # 保存图像
        img2.save(os.path.join(save_train_Path, str(i+1+imgNum+imgNum) + '.' + ext))

        img3 = Image.open(os.path.join(train_lablePath, trainlist1[i]))     # 读取mask
        img4 = Image.open(os.path.join(train_lablePath, trainlist2[i]))
        print(os.path.join(train_lablePath, trainlist1[i]))
        print(os.path.join(train_lablePath, trainlist2[i]))
        region3 = img3.crop(box1)    # 取出mask图像块左部分
        region4 = img4.crop(box2)    # 取出mask图像块右部分
        img3.paste(region4, box2)    # 粘贴mask图像块
        img4.paste(region3, box1)
        img3.save(os.path.join(save_train_lablePath, str(i+1+imgNum) + '.' + ext))   # 保存图像
        img4.save(os.path.join(save_train_lablePath, str(i+1+imgNum+imgNum) + '.' + ext))

# 2. 图片对切分组合（图1与图片2上下部分对调）    Up & Down
def image_split_UD(train_Path, train_lablePath,save_train_Path, save_train_lablePath):
    if not os.path.exists(save_train_Path):
     os.makedirs(save_train_Path)
    if not os.path.exists(save_train_lablePath):
     os.makedirs(save_train_lablePath)

    trainlist1 = os.listdir(train_Path)
    imgNum = int(len(trainlist1))
    trainlist1.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    trainlist2 = copy.deepcopy(trainlist1)  # 深拷贝
    random.shuffle(trainlist2)  # 注意shuffle没有返回值，该函数完成一种功能，就是对list进行排序打乱

    for i in range(imgNum):
        img1 = Image.open(os.path.join(train_Path, trainlist1[i]))          # 读取图片
        img2 = Image.open(os.path.join(train_Path, trainlist2[i]))
        Width, Height = img1.size    # 得到图片的尺寸：宽、高像素
        print(os.path.join(train_Path, trainlist1[i]))
        print(os.path.join(train_Path, trainlist2[i]))
        ext = trainlist1[i].split('.')[-1]          # 得到图片格式后缀：png

        box1 = (0, 0, Width, int(Height/2))         # 上部分
        box2 = (0, int(Height/2), Width, Height)    # 下部分

        region1 = img1.crop(box1)    # 取出图像块左部分
        region2 = img2.crop(box2)    # 取出图像块右部分
        img1.paste(region2, box2)    # 粘贴图像块
        img2.paste(region1, box1)
        img1.save(os.path.join(save_train_Path, str(i+1+imgNum) + '.' + ext))  # 保存图像
        img2.save(os.path.join(save_train_Path, str(i+1+imgNum+imgNum) + '.' + ext))

        img3 = Image.open(os.path.join(train_lablePath, trainlist1[i]))     # 读取mask
        img4 = Image.open(os.path.join(train_lablePath, trainlist2[i]))
        print(os.path.join(train_lablePath, trainlist1[i]))
        print(os.path.join(train_lablePath, trainlist2[i]))
        region3 = img3.crop(box1)    # 取出mask图像块左部分
        region4 = img4.crop(box2)    # 取出mask图像块右部分
        img3.paste(region4, box2)    # 粘贴mask图像块
        img4.paste(region3, box1)
        img3.save(os.path.join(save_train_lablePath, str(i+1+imgNum) + '.' + ext))   # 保存图像
        img4.save(os.path.join(save_train_lablePath, str(i+1+imgNum+imgNum) + '.' + ext))

# 3 图片重命名
def Rename_file(imgPath, number):
    img_file = os.listdir(imgPath)
    # 按照文件夹的顺序命名，不然会被打乱
    img_file.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    img_file_counter = number
    for img in img_file:
        os.rename(imgPath + '/' + img, imgPath + '/' + str(img_file_counter) + '.' + img.split('.')[-1])
        img_file_counter += 1
    return img_file_counter

if __name__ == "__main__":
	print( "----数据预处理任务开始------")
	start = time.time()
	image_split_LR(train_Path, train_lablePath,save_train_Path, save_train_lablePath)      #左右对换
	# image_split_UD(train_Path, train_lablePath,save_train_Path, save_train_lablePath)      #上下对换
	print("处理总耗时：%s" % (time.time()-start))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++