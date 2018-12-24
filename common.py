# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
import numpy as np


def conv2d(scope, inputs, depth, ksize=(1, 1, 3), stride=1, padding='SAME', activation_fn=tf.nn.relu,
           reuse=False, weights_initializer=None, biases_initializer=None,
           is_pool=True, pool=tf.nn.max_pool, pool_size=(2, 2), pool_stride=2, pool_padding='SAME'):
    with tf.variable_scope(scope, reuse=reuse):
        # 卷积
        if reuse:
            w, b = tf.get_variable('w'), tf.get_variable('b')
        else:
            w_init = weights_initializer if weights_initializer else tf.truncated_normal_initializer(stddev=.1)
            b_init = biases_initializer if biases_initializer else tf.truncated_normal_initializer(stddev=.1)
            w = tf.get_variable('w', shape=(ksize[0], ksize[1], ksize[2], depth), dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=(depth,), dtype=tf.float32, initializer=b_init)
        net = tf.nn.conv2d(inputs, filter=w, strides=(1, stride, stride, 1), padding=padding)
        net = activation_fn(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
        # 池化
        if is_pool:
            net = pool(net, ksize=(1, pool_size[0], pool_size[1], 1),
                       strides=(1, pool_stride, pool_stride, 1), padding=pool_padding)
            print(net.name, net.shape)
        return net


# 全连接
def fc(scope, inputs, output_num, flatten=False, activation_fn=tf.nn.relu, keep_prob=1.0,
       weights_initializer=None, biases_initializer=None, softmax=False, reuse=False,
       weights_regularizer=None, collection_name=None):
    # 矩阵转换向量(扁平化)
    if flatten:
        inputs = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2] * inputs.shape[3]))
    with tf.variable_scope(scope, reuse=reuse):
        # 全连接参数
        if not reuse:
            w_init = weights_initializer if weights_initializer else tf.truncated_normal_initializer(stddev=.1)
            b_init = biases_initializer if biases_initializer else tf.truncated_normal_initializer(stddev=.1)
            w = tf.get_variable('w', shape=(inputs.shape[1], output_num), dtype=tf.float32, initializer=w_init)
            # 加入正则项
            if weights_regularizer:
                tf.add_to_collection(collection_name, value=weights_regularizer(w))
            # 偏置项
            b = tf.get_variable('b', shape=(output_num,), dtype=tf.float32, initializer=b_init)
        else:
            w, b = tf.get_variable('w'), tf.get_variable('b')
        # 激励
        net = activation_fn(tf.matmul(inputs, w) + b)
        print(net.name, net.shape)
        # dropout
        if 0 < keep_prob < 1: net = tf.nn.dropout(net, keep_prob=keep_prob)
        # softmax
        if softmax: net = tf.nn.softmax(net)
        return net


# 图像预处理，输入原始图像，输出符合模型训练的输入图像
def image_preprocess(image_tensor, height, width, bbox=None, is_train=True):
    # 转换图像张量类型
    if image_tensor.dtype != tf.float32:
        image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)

    # 没有标注框，则认为整个图像都是关注的部分
    if bbox is None:
        bbox = tf.constant([0, 0, 1, 1], dtype=tf.float32, shape=(1, 1, 4))

    if is_train:
        # 根据标注框随机截取图像，减少需要关注的物体大小对模型的影响
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image_tensor),
                                        bounding_boxes=bbox, min_object_covered=0.4)  # 计算随机截取坐标
        image_tensor = tf.slice(image_tensor, begin=bbox_begin, size=bbox_size)  # 根据坐标裁剪

    # 调整随机截取的图像的大小，缩放算法随机选择
    image_tensor = tf.image.resize_images(image_tensor, size=(height, width), method=np.random.randint(0, 4))

    if is_train:
        # 随机左右、上下、对角线翻转图像
        image_tensor = tf.image.random_flip_up_down(image_tensor)
        image_tensor = tf.image.random_flip_left_right(image_tensor)
        if np.random.randint(0, 2):
            image_tensor = tf.image.transpose_image(image_tensor)

        # 随机调整图像色彩顺序（如亮度、对比度、饱和度、色相），以降低无关因素对模型的影响
        color = np.array([tf.image.random_brightness(image_tensor, max_delta=32 / 255),
                          tf.image.random_contrast(image_tensor, lower=0.5, upper=1.5),
                          tf.image.random_hue(image_tensor, max_delta=0.2),
                          tf.image.random_saturation(image_tensor, lower=0.5, upper=1.5)])
        np.random.shuffle(color)
        for img in color:
            image_tensor = img

    return tf.clip_by_value(image_tensor, clip_value_min=0.0, clip_value_max=1.0)  # 范围外的值截断


# import tensorflow as tf
#
# a = []
# c = tf.constant([[1,2,3],[4,5,6]], dtype=tf.float32)
# c = tf.expand_dims(c, 0)
# a.append(c)
# c = tf.constant([[3,2,1],[6,5,4]], dtype=tf.float32)
# c = tf.expand_dims(c, 0)
# a.append(c)
# a = tf.concat(a, axis=0)
# a = tf.reduce_mean(a, axis=0)
# with tf.Session() as sess:
#     print(sess.run(a))