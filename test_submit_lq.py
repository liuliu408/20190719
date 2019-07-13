# coding=utf-8

import torch as tc
from PIL import Image
import numpy as np
import os
import cv2
import time

np.set_printoptions(threshold=np.inf)  # print not show ...
Image.MAX_IMAGE_PIXELS = 100000000000

model = tc.load('/home/liuq/wind/agriculture/231717_LQ2/model_train/model200.pth')

# 分割预测
def test(model, src_path, dst_path):
    timestart = time.time()
    model.eval()

    imagelist = os.listdir(src_path)
    imgNum = int(len(imagelist))
    imagelist.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))

    for i in range(imgNum):
        img = Image.open(os.path.join(src_path, imagelist[i]))  # 读取图片
        img = np.asanyarray(img)
        height, width, _ = img.shape   # img.shape[0]：图像的垂直尺寸（高度） img.shape[1]：图像的水平尺寸（宽度） img.shape[2]：图像的通道数
        print(img.shape)

        _img = np.zeros((height, width, 3)).astype(np.int32)
        _img[:, :, :] = img[:, :, :3]                # 只取前面三个通道数据：RGB

        _img = np.array(_img).transpose([2, 0, 1])   # 维度：C*H*W
        print(img.shape)

        # predict
        x = tc.from_numpy(_img / 255.0).float()      # 归一化数据
        x = x.unsqueeze(0).to(device)                # 第0维设置为1最后形成维度：1*C*H*W   --增加一个维度
        rr = model.forward(x)
        rr = rr.detach()[0, :, :, :].cpu()           # 表示不回传梯度
        data = tc.argmax(rr, 0).byte().numpy()       # 指定dim=0时，行的size没有了。求每一列的最大行标！

        imgx = Image.fromarray(data).convert('L')    # PIL的九种不同模式：1，L，P，RGB，RGBA，CMYK，YCbCr,I，F
        predict_path = os.path.join(os.path.join(dst_path,  str(i) + '.png'))
        imgx.save(predict_path)
    print("分割预测耗时：%s" % (time.time()-timestart))


# 定义图像拼接函数
def image_compose(ImagePath, width, height, saved_path):
    timestart = time.time()
    print('开始处理图片拼接, 请稍候...')
    Image_FORMAT = ['.jpg', '.JPG', '.png']  # 图片格式
    Image_SIZE = 1024                         # 小图大小为512*512
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

# 可视化
def visualize(src_img, dst_img):
    timestart = time.time()
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
    print("可视化耗时：%s" % (time.time()-timestart))

if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------------------
    # 预测
    use_cuda = True
    device = tc.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    for i in range(3, 5):
        src_path = "/home/liuq/wind/agriculture/231717_LQ2/data/img{}_split_1024".format(i)
        dst_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/img{}_predict_{}/".format(i,
                    time.strftime("%Y-%m-%d", time.localtime()))

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        test(model, src_path, dst_path)
        print("完成了 test_image{} 预测".format(i))
    print("预测结束")

    # --------------------------------------------------------------------------------------------------------
    # 拼接
    for i in range(3, 5):
        src_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/img{}_predict_{}/".format(i,
                    time.strftime("%Y-%m-%d", time.localtime()))
        dst_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/predict_result_img"

        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        if i == 3:
            w, h = 37241, 19903    	# image_3.png ：37241 * 19903
        elif i == 4:
            w, h = 25936, 28832     # image_4.png ：25936 * 28832

        image_compose(src_path, w, h, os.path.join(dst_path, 'image_{}_predict.png'.format(i)))
        print("完成了 test_image{} 拼接".format(i))

    print("拼接结束")

    # --------------------------------------------------------------------------------------------------------
    # 可视化
    for i in range(3, 5):
        src_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/predict_result_img/image_{}_predict.png".format(i)
        dst_path = "/home/liuq/wind/agriculture/231717_LQ2/model_train/predict_result_img/vis_image_{}_predict.png".format(i)
        visualize(src_path, dst_path)

    print("可视化结束")

