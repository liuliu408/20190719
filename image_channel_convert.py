# coding:utf-8
# 2019.06.20 刘强 四川大学614实验室
import os
import time
import copy
import numpy as np
from PIL import Image

imagePath = '/home/liuq/wind/agriculture/231717/data/train_img/'
imagelabelPath = '/home/liuq/wind/agriculture/231717/data/train_img_label/'

# 将图片:4通道--转--3通道
def image_channel_4_3(image_Path):
	img_file = os.listdir(image_Path)
	img_file.sort(key=lambda x: int(x.split('.')[0]))
	imgNum = int(len(img_file))

	for i in range(imgNum):
		img = Image.open(os.path.join(image_Path, img_file[i]))
		Width, Height = img.size
		print('原图信息: %s * %s, %s, %s' % (Width, Height, img.format, img.mode))

		img = np.asanyarray(img)                # 图片转变为numpy数据
		# print(img.shape)
		_img = img[:, :, :3]                    # 只取前三个通道
		# print(_img.shape)

		_img = Image.fromarray(np.uint8(_img))  # numpy转为图片格式
		_img.save(os.path.join(image_Path, img_file[i]))
		print('已经转换了%s 张' %(i+1))

		# 测试一张图片是否转换正常
		# imgx = Image.open(os.path.join(image_Path, img_file[i]))
		# Width, Height = imgx.size
		# print('处理后信息: %s * %s, %s, %s' % (Width, Height, imgx.format, imgx.mode))

# 图像灰度转换
def label_grey_covert(image_Path):
	img_file = os.listdir(image_Path)
	img_file.sort(key=lambda x: int(x.split('.')[0]))
	imgNum = int(len(img_file))

	for i in range(imgNum):
		img_label = Image.open(os.path.join(image_Path, img_file[i]))
		Width, Height = img_label.size
		print('原图信息: %s * %s, %s, %s' % (Width, Height, img_label.format, img_label.mode))

		img_label = np.asanyarray(img_label)                # 图片转变为numpy数据
		# print(img.shape)
		_img_label = Image.fromarray(img_label).convert('L')
		# print(_img.shape)
		_img_label.save(os.path.join(image_Path, img_file[i]))
		print('已经转换了%s 张图片' %(i+1))

	# 测试一张图片是否转换正常
	imgx = Image.open(os.path.join(image_Path, img_file[imgNum-1]))
	Width, Height = imgx.size
	print('处理后信息: %s * %s, %s, %s' % (Width, Height, imgx.format, imgx.mode))

if __name__ == '__main__':
	# image_channel_4_3(imagePath)   # 训练数据通道转换

	imgx = Image.open(os.path.join(imagelabelPath, "1637.png"))
	Width, Height = imgx.size
	print('处理后信息: %s * %s, %s, %s' % (Width, Height, imgx.format, imgx.mode))