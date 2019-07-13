# coding:utf-8
# 2019.06.20 刘强 四川大学614实验室
import os
from PIL import Image
import time
import copy
import numpy as np


# 切割图片（大图切分成数个小图），
# 每行的最后一张分割图的大小还要加上剩余的部分的列像素
# 每列的最后一张分割图的大小还要加上剩余的部分的行像素
# num 图片编号，如image_1.png中的1
def split_image(ImagePath, num, size,saved_path):
	timestart = time.time()
	print('开始处理图片切割, 请稍候...')
	# --------------------------------------------------------------------------
	Image.MAX_IMAGE_PIXELS = 100000000000
	img = Image.open(ImagePath)
	Width, Height = img.size          # 得到图片的尺寸：宽、高像素
	print(image_path)                 # 显示图片路径
	print('原图信息: %s * %s, %s, %s' % (Width, Height, img.format, img.mode))
	print('分割像素：{}*{}'.format(size, size))

	rownum = Height // size
	colnum = Width  // size
	print('分割行列：{} * {}'.format(colnum, rownum))

	num = 0
	for r in range(rownum):           # 行循环
		if (r == (rownum - 1)):       # 判断是否是最后1行
			rr = Height % size        # 多余的行
			print('rr={}'.format(rr))
		else:
			rr = 0

		for c in range(colnum):       # 列循环
			num = num + 1
			if (c == (colnum - 1)):   # 判断是否是最后1列
				cc = Width % size     # 多余的列
				print('cc={}'.format(cc))
			else:
				cc = 0
			box = (c * size, r * size, (c + 1) * size + cc, (r + 1) * size + rr)  # 左上角，右下脚坐标
			print('c={} r={}'.format(c, r))
			img.crop(box).save(os.path.join(saved_path, ImagePath.split('/')[-1].split('.')[0] + '_' + str(num) + '.' + ImagePath.split('.')[-1]))

	print('图片切割完毕，共生成 %s 张小图片。' % num)
	print("切割耗时：%s" % (time.time()-timestart))
	# --------------------------------------------------------------------------

# 定义图像拼接函数
def image_compose(ImagePath, width, height, saved_path):
	timestart = time.time()
	print('开始处理图片拼接, 请稍候...')
	Image_FORMAT = ['.jpg', '.JPG', '.png']  # 图片格式
	Image_SIZE = 512                         # 小图大小为512*512
	Image_Col = width // Image_SIZE          # 图片间隔，也就是合并成一张图后，一共有几列
	Image_Row = height // Image_SIZE         # 图片间隔，也就是合并成一张图后，一共有几行

	# 获取图片文件夹下的所有图片名称
	image_names = [name for name in os.listdir(ImagePath) for item in Image_FORMAT if os.path.splitext(name)[1] == item]
	image_names = sorted(image_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

	to_image = Image.new('L', (width, height))  # 创建一个新图,不要创建RGB的

	# 循环遍历，把每张图片按顺序粘贴到对应位置上
	to_image.MAX_IMAGE_PIXELS = 100000000000
	for y in range(Image_Row):
		for x in range(Image_Col):
			from_image = Image.open(os.path.join(ImagePath, image_names[Image_Col * y + x]))
			to_image.paste(from_image, ((x * Image_SIZE), (y * Image_SIZE)))

	to_image.save(saved_path)  # 保存新图
	print("图像拼接耗时：%s" % (time.time()-timestart))
	print("拼接结束")

if __name__ == "__main__":
	print( "----数据预处理任务开始------")
	start = time.time()
	# compose = True
	# ==========================================================================
	# 1. 大图切割小图---用于测试
	# image_3.png ：37241 * 19903, PNG, RGBA
	# image_4.png ：25936 * 28832, PNG, RGBA
	for i in range(3, 5):
		size = 1024  # 切割为1024*1024
		image_path = '/home/liuq/wind/agriculture/origin_data/test_data/image_{}.png'.format(i)
		dst_Path = '/home/liuq/wind/agriculture/231717_liuq/data/img{}_split_{}/'.format(i, size)

		if not os.path.exists(dst_Path):
			os.makedirs(dst_Path)

		split_image(image_path, i, 1024,dst_Path)   # 实施切割
		# image_3.png ：37241 * 19903 分成共生成 36*19=84图片切割完毕。切割耗时：214.78217244148254
		# image_4.png ：25936 * 28832 分成共生成 25*28=700图片切割完毕。切割耗时：207.34162759780884

	# ==========================================================================
	# 将测试的小图拼接为大图测试
	for i in range(3, 5):
		src_path = "/home/liuq/wind/agriculture/231717_liuq/model_train/img{}_predict_{}/".format(i,
						time.strftime("%Y-%m-%d",time.localtime()))
		dst_path = "/home/liuq/wind/agriculture/231717_liuq/model_train/predict_result_img"

		if not os.path.exists(dst_path):
			os.mkdir(dst_path)
		if i == 3:
			w, h = 37241, 19903  # image_3.png ：37241 * 19903
		elif i == 4:
			w, h = 25936, 28832  # image_4.png ：25936 * 28832
		image_compose(src_path, w, h, os.path.join(dst_path, 'image_{}_predict.png'.format(i)))
		print("完成了 test_image{} 拼接".format(i))
	print("处理总耗时：%s" % (time.time()-start))

