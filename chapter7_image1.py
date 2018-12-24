# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
import tensorflow as tf

"""图像处理函数"""
I = 0
def show(img_tensor, seconds=2):
    img_tensor = tf.image.convert_image_dtype(img_tensor, dtype=tf.uint8)
    mat = img_tensor.eval()
    print('图像形状为：{}'.format(mat.shape))
    if mat.shape[-1] == 3:
        for w in mat:
            for c in w:
                c[0], c[2] = c[2], c[0]
    global I
    I += 1
    cv2.imshow('show%d' % I, mat)
    cv2.waitKey(seconds * 1000)


sess = tf.InteractiveSession()
# 1.读取数据
path = ['../OpenCV3/img/1.jpg', '../OpenCV3/img/ice&fire3.jpg']
image_raw = tf.read_file(path[0])


# 2.图像格式转换
# 参数channels：可选值：0 1 3 4，默认为0， 一般使用0 1 3，不建议使用4
# 0：使用图像的默认通道，也就是图像是几通道的就使用几通道
# 1：使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
# 3：使用RGB三通道读取数据
# 4：使用RGBA四通道读取数据(R：红色，G：绿色，B：蓝色，A：透明度)
png_tensor = tf.image.decode_png(contents=image_raw, channels=0)
# show(png_tensor)
jpg_tensor = tf.image.decode_jpeg(contents=image_raw, channels=3)
# show(jpg_tensor)
gray_tensor = tf.image.decode_png(contents=image_raw, channels=1)
# show(gray_tensor)


# 3.图像大小重置
# images格式为：[height, width, num_channels]或者[batch, height, width, num_channels]
# API返回值和images格式一样，唯一区别是height和width变化为给定的值
# method:BILINEAR=0 线性插值，默认；NEAREST_NEIGHBOR=1 最近邻插值，失真最小；BICUBIC=2 三次插值；AREA=3 面积插值
resize_tensor = tf.image.resize_images(images=png_tensor, size=(300, 400),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# show(resize_tensor)
resize_tensor = tf.image.resize_images(jpg_tensor, size=(300, 400))
# show(resize_tensor)


# 4.图像剪切和填充
# 图片重置大小，通过图片的剪切或者填充（从中间开始计算新图片的大小）
crop_pad_tensor = tf.image.resize_image_with_crop_or_pad(png_tensor, target_height=600, target_width=800)
# show(crop_pad_tensor, seconds=5)

# 中间等比例剪切
central_crop_tensor = tf.image.central_crop(png_tensor, central_fraction=0.5)
# show(central_crop_tensor, 5)

# 填充数据（给定位置开始填充，offset_width为图片向右偏移，offset_height为向下偏移）
pad_to_bounding_box_tensor = tf.image.pad_to_bounding_box(jpg_tensor, offset_width=50, offset_height=100,
                                                          target_width=1200, target_height=900)
# show(pad_to_bounding_box_tensor, 10)

# 剪切数据（给定位置开始剪切，target_width和target_height不能超过原图像大小）
crop_to_bounding_box_tensor = tf.image.crop_to_bounding_box(png_tensor, offset_width=100, offset_height=100,
                                                            target_width=200, target_height=250)
# show(crop_to_bounding_box_tensor, 10)


# 5.翻转
flip_up_down_tensor = tf.image.flip_up_down(png_tensor)  #上下翻转
# show(flip_up_down_tensor)

flip_left_right_tensor = tf.image.flip_left_right(jpg_tensor)  # 左右翻转
# show(flip_left_right_tensor)

flip_transpose_tensor = tf.image.transpose_image(png_tensor)  # 转置
# show(flip_transpose_tensor, 10)

flip_krot90_tensor = tf.image.rot90(png_tensor, k=1)  # 旋转(90、180、270...) k*90度逆时针旋转
# show(flip_krot90_tensor)


# 6、颜色空间转换（rgb、hsv、gray） 必须将image的值转换为float32类型，不能使用unit8类型
float32_tensor = tf.image.convert_image_dtype(jpg_tensor, dtype=tf.float32)  # 图片值类型转换

# rgb -> hsv（h:图像的色彩/色度，s:图像的饱和度，v：图像的亮度）
hsv_tensor = tf.image.rgb_to_hsv(float32_tensor)  # 返回浮点
# show(hsv_tensor)

# hsv -> rgb
rgb_tensor = tf.image.hsv_to_rgb(hsv_tensor)  # 返回浮点
# show(rgb_tensor)

# rgb -> gray
grayscale_tensor = tf.image.rgb_to_grayscale(jpg_tensor)
# show(grayscale_tensor)

# 从颜色空间中提取图像的轮廓信息(图像的二值化，一般为灰度图像)
thread_tensor = tf.where(gray_tensor < 100, gray_tensor, 255 - gray_tensor + gray_tensor)
# show(thread_tensor, 10)


# 7.图像的调整
# 亮度调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# delta: 取值范围(-1, 1）之间的float类型的值，表示对于亮度的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h,s,v*delta -> rgb
adjust_brightness_tensor = tf.image.adjust_brightness(jpg_tensor, delta=-0.5)
# show(adjust_brightness_tensor)

# 色调调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# delta: 取值范围(-1,1）之间的float类型的值，表示对于色调的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h*delta,s,v -> rgb
adjust_hue_tensor = tf.image.adjust_hue(jpg_tensor, delta=0.5)
# show(adjust_hue_tensor)

# 饱和度调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# saturation_factor: 一个float类型的值，表示对于饱和度的减弱或者增强的系数值，饱和因子
# 底层执行：rgb -> hsv -> h,s*saturation_factor,v -> rgb
adjust_saturation_tensor = tf.image.adjust_saturation(jpg_tensor, saturation_factor=20)
# show(adjust_saturation_tensor)

# 对比度调整，公式：(x-mean) * contrast_factor + mean
adjust_contrast_tensor = tf.image.adjust_contrast(jpg_tensor, contrast_factor=10)  # 返回浮点
# show(adjust_contrast_tensor)

# 图像的gamma校正
# images: 要求必须是float类型的数据
# gamma：任意值，Oup = In * Gamma
adjust_gamma_tensor = tf.image.adjust_gamma(float32_tensor, gamma=10)  # 返回浮点
# show(adjust_gamma_tensor)

# 图像的归一化(x-mean)/adjusted_stddev, adjusted_stddev=max(stddev, 1.0/sqrt(image.NumElements()))
standardization_tensor = tf.image.per_image_standardization(jpg_tensor)  # 返回浮点
# standardization_tensor = tf.where(standardization_tensor < 0, -standardization_tensor, standardization_tensor)
# standardization_tensor = tf.cast(standardization_tensor * 255, tf.uint8)
# show(standardization_tensor)


# 8.噪音数据的加入
img = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)
h, w = img.shape[0], img.shape[1]
noisy_tensor = tf.cast(tf.truncated_normal((h, w, 3), mean=10, stddev=20), dtype=tf.uint8)
noisy_image_tensor = jpg_tensor + noisy_tensor
print(noisy_tensor.eval())
show(noisy_image_tensor, 5)


sess.close()

