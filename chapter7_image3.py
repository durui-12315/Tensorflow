# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""图像预处理完整样例：随机截取->随机翻转->调整图像大小->随机颜色调整"""


# 图像预处理，输入原始图像，输出符合模型训练的输入图像
def image_preprocess(image_tensor, height, width, bbox=None):
    # 没有标注框，则认为整个图像都是关注的部分
    if bbox is None:
        bbox = tf.constant([0, 0, 1, 1], dtype=tf.float32, shape=(1, 1, 4))

    # 转换图像张量类型
    if image_tensor.dtype != tf.float32:
        image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)

    # 根据标注框随机截取图像，减少需要关注的物体大小对模型的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image_tensor),
                                    bounding_boxes=bbox, min_object_covered=0.4)  # 计算随机截取坐标
    image_tensor = tf.slice(image_tensor, begin=bbox_begin, size=bbox_size)  # 根据坐标裁剪

    # 随机左右、上下、对角线翻转图像
    image_tensor = tf.image.random_flip_up_down(image_tensor)
    image_tensor = tf.image.random_flip_left_right(image_tensor)
    if np.random.randint(0, 2):
        image_tensor = tf.image.transpose_image(image_tensor)

    # 调整随机截取的图像的大小，缩放算法随机选择
    image_tensor = tf.image.resize_images(image_tensor, size=(height, width), method=np.random.randint(0, 4))
    #image_tensor = tf.reshape(image_tensor, shape=(height, width, 3))

    # 随机调整图像色彩顺序（如亮度、对比度、饱和度、色相），以降低无关因素对模型的影响
    color = np.array([tf.image.random_brightness(image_tensor, max_delta=32 / 255),
                      tf.image.random_contrast(image_tensor, lower=0.5, upper=1.5),
                      tf.image.random_hue(image_tensor, max_delta=0.2),
                      tf.image.random_saturation(image_tensor, lower=0.5, upper=1.5)])
    np.random.shuffle(color)
    for img in color:
        image_tensor = img
    return tf.clip_by_value(image_tensor, clip_value_min=0.0, clip_value_max=1.0)  # 范围外的值截断


if __name__ == '__main__':
    img_tensor = tf.read_file('../OpenCV3/img/dog1.jpg')
    img_tensor = tf.image.decode_jpeg(img_tensor, channels=3)
    boxes = tf.constant([[[.1, .24, .6, .69], [.31, .36, .36, .41]]])
    with tf.Session() as sess:
        for i in range(10):
            result = image_preprocess(img_tensor, 300, 300, bbox=boxes)
            plt.imshow(result.eval())
            plt.show()

