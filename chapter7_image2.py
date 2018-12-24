# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
import tensorflow as tf

"""图像处理：
1.神经网络输入节点的个数是固定的，所以将图像的像素作为输入提供给神经网络之前，需要将图像大小统一；
2.图像的翻转、亮度、对比度、饱和度和色相在很多图像识别中都不会影响识别结果，大大增加输入数据的多样性，
使训练得到的模型尽可能小的受到无关因素的影响；
3.图像处理前统一把图像的uint8转换为float32，其值由0-255变成了0.0-1.0；
4.随机截取图像上有信息含量的部分是一个提高模型健壮性(robustness)的手段；
5.图像预处理步骤：根据标注框随机截取、随机翻转、调整大小、随机调整颜色
"""


I = 0
def show(img_tensor, seconds=3):
    mat = img_tensor.eval()
    print('图像形状为：{}，矩阵为：{}'.format(mat.shape, mat))
    if mat.shape[-1] == 3:
        for w in mat:
            for c in w:
                c[0], c[2] = c[2], c[0]
    global I
    I += 1
    cv2.imshow('show%d' % I, mat)
    cv2.waitKey(seconds * 1000)


sess = tf.InteractiveSession()
# 1.读取数据并解析、转浮点
path = ['../OpenCV3/img/dog1.jpg', '../OpenCV3/img/ice&fire3.jpg']
image_raw = tf.read_file(path[0])  # 读取
png_tensor = tf.image.decode_png(image_raw, channels=3)  # 解码
float32_png_tensor = tf.image.convert_image_dtype(png_tensor, dtype=tf.float32)  # 转换，从0-255转为0.0-1.0
# show(float32_png_tensor)


# 2.缩放
resize_tensor = tf.image.resize_images(float32_png_tensor, size=(300, 300), method=0)
# show(resize_tensor)


# 3.裁剪和填充（从图像中心开始）
pad_crop_tensor = tf.image.resize_image_with_crop_or_pad(float32_png_tensor, target_height=500, target_width=500)
# show(pad_crop_tensor)

crop_pad_tensor = tf.image.resize_image_with_crop_or_pad(float32_png_tensor, target_width=1300, target_height=1300)
# show(crop_pad_tensor)

pad_tensor = tf.image.pad_to_bounding_box(float32_png_tensor, offset_height=50, offset_width=100,
                                          target_height=1000, target_width=1300)
# show(pad_tensor)

crop_tensor = tf.image.crop_to_bounding_box(float32_png_tensor, offset_height=50, offset_width=300,
                                            target_height=500, target_width=500)
# show(crop_tensor)

central_tensor = tf.image.central_crop(float32_png_tensor, central_fraction=.5)
# show(central_tensor)


# 4.图像翻转
flip_tensor = tf.image.flip_up_down(float32_png_tensor)
# show(flip_tensor)

flip_tensor = tf.image.flip_left_right(float32_png_tensor)
# show(flip_tensor)

transpose_tensor = tf.image.transpose_image(float32_png_tensor)
# show(transpose_tensor)

flip_50_tensor = tf.image.random_flip_left_right(float32_png_tensor)  # 50%的概率左右翻转
# show(flip_50_tensor)

flip_50_tensor = tf.image.random_flip_up_down(float32_png_tensor)  # 50%的概率上下翻转
# show(flip_50_tensor)


# 5.图像的亮度、对比度、饱和度和色相
# 亮度
adjust_tensor = tf.image.adjust_brightness(float32_png_tensor, delta=-.5)  # 亮度-0.5
# show(adjust_tensor)
adjust_tensor = tf.image.random_brightness(float32_png_tensor, max_delta=.8)  # 随机亮度
# show(adjust_tensor)

# 对比度
adjust_tensor = tf.image.adjust_contrast(float32_png_tensor, contrast_factor=.5)  # 减少0.5倍
# show(adjust_tensor)
adjust_tensor = tf.image.adjust_contrast(float32_png_tensor, contrast_factor=5)  # 增加5倍
# show(adjust_tensor)
adjust_tensor = tf.image.random_contrast(float32_png_tensor, lower=.5, upper=5)  # 随机对比度
# show(adjust_tensor)

# 色相
adjust_tensor = tf.image.adjust_hue(float32_png_tensor, delta=.9)  # 增加0.9倍
# show(adjust_tensor)
adjust_tensor = tf.image.random_hue(float32_png_tensor, max_delta=.5)  # [0, 0.5]范围的随机色相(最高0.5)
# show(adjust_tensor)

# 饱和度
adjust_tensor = tf.image.adjust_saturation(float32_png_tensor, saturation_factor=-5)  # 饱和度-5
# show(adjust_tensor)
adjust_tensor = tf.image.random_saturation(float32_png_tensor, lower=0, upper=10)  # 随机饱和度，lower为非负数
# show(adjust_tensor)

# 图像标准化：均值为0， 方差为1
adjust_tensor = tf.image.per_image_standardization(float32_png_tensor)
# show(adjust_tensor)


# # 6.处理标注框(需要一个batch的图像，4维)
# bounding_tensor = tf.image.resize_images(float32_png_tensor, size=(800, 800))
# bounding_tensor = tf.expand_dims(bounding_tensor, axis=0)  # 图像维度扩大到4维，实现一个batch的多图像维度
# boxes = tf.constant([[[.1, .24, .6, .69], [.31, .36, .36, .41]]])  # [Ymin, Xmin, Ymax, Xmax]，3维数据，和batch保持一致
# bounding_tensor = tf.image.draw_bounding_boxes(bounding_tensor, boxes=boxes)  # 标注图像
# for mat in bounding_tensor.eval():
#     print('图像形状为：{}，矩阵为：{}'.format(mat.shape, mat))
#     for w in mat:
#         for c in w:
#             c[0], c[2] = c[2], c[0]
#     cv2.imshow('show', mat)
#     cv2.waitKey(2000)

# 7.根据标注框随机截取图像
boxes = tf.constant([[[.1, .24, .6, .69], [.31, .36, .36, .41]]])
# 随机标注图像（至少包含40%的目标内容）
resize_tensor = tf.image.resize_images(float32_png_tensor, size=(600, 600))
begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(image_size=tf.shape(resize_tensor),
                                bounding_boxes=boxes, min_object_covered=0.4)
batched_tensor = tf.expand_dims(resize_tensor, axis=0)
bounding_tensor1 = tf.image.draw_bounding_boxes(batched_tensor, boxes=bbox_for_draw)
# 根据随机标注的图像坐标截取图像1
bounding_tensor2 = tf.slice(resize_tensor, begin=begin, size=size)
lst = [bounding_tensor1.eval()[0], bounding_tensor2.eval()]
# 根据随机标注的图像坐标截取图像2
for x, y, x1, y1 in bbox_for_draw.eval()[0]:
    x, y, x1, y1 = map(lambda x: int(x * 600), [x, y, x1, y1])
    tmp = tf.image.crop_to_bounding_box(resize_tensor, x, y, x1 - x, y1 - y)
    lst.append(tmp.eval())
for mat in lst:
    print(mat.shape)
    for w in mat:
        for c in w:
            c[0], c[2] = c[2], c[0]
    cv2.imshow('show', mat)
    cv2.waitKey(2000)

sess.close()