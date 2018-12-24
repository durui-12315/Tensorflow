# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import chapter6_net
import numpy as np
import os
import cv2
import tensorflow.contrib.slim as slim
import common

"""
一、卷积网络的参数比全连接要少很多，不会随着图片的增大而增大，只和卷积核的大小有关。
二、卷积神经网络的5中结构：
    1.输入层。整个卷积神经网络的输入，一般为一张图片的像素矩阵；
    2.卷积层。将网络的每一小块进行更加深入地分析从而得到抽象度更高的特征；使通道变得更深
    3.池化层。进一步缩小最后全连接层节点的个数，减少神经网络参数的目的；加快计算速度，防止过拟合
    4.全连接层。在多轮卷积和池化之后，一般1-2层；组合进过卷积池化的图像特征
    5.softmax层。用于目标分类，得到当前样例属于不同种类的概率分布，即每个种类的置信度
    6.正则表达：输入层 -> (卷积层+ -> 池化层?)+ -> 全连接层+ -> softmax?
三、dropout可进一步提升模型的可靠性并防止过拟合；只在训练时使用；一般只在全连接层使用
四、迁移学习：
    1.用来解决标注数据和训练时间的问题；
    2.将一个问题上训练好的模型通过简单的调整使其适用于一个新的问题；
    3.一般保留一个模型之前的卷积部分（卷积的本质是图像特征提取），替换最后的全连接层，
在旧的卷积参数基础上训练新的全连接层参数；
    4.在数据量足够的情况下，迁移学习的效果不如重新训练。
"""

# # 卷积层前向传播(5*5大小，通道3，深度16)
# filter_w = tf.get_variable('w', shape=(5, 5, 3, 16), initializer=tf.truncated_normal_initializer(stddev=.1))
# filter_b = tf.get_variable('b', shape=(16,), initializer=tf.truncated_normal_initializer(stddev=.1))
# inputs = tf.ones((30, 10, 10, 3), dtype=tf.float32)
# conv = tf.nn.conv2d(inputs, filter=filter_w, strides=[1, 2, 2, 1], padding='SAME')  # 卷积计算(padding为自动)
# conv = tf.nn.bias_add(conv, filter_b)  # 添加偏置项
# actived_conv = tf.nn.relu(conv)  # 去线性化
# print(conv.shape)
#
# # 池化层(3*3大小)
# pool = tf.nn.max_pool(actived_conv, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')  # 不加padding
# print(pool.shape)

# # 测试LeNet-5
# inputs = tf.placeholder(tf.float32, shape=(None, 28 ** 2))
# labels = tf.placeholder(tf.float32, shape=(None, 10))
# outputs = chapter6_net.LeNet_5(tf.reshape(inputs, shape=(-1, 28, 28, 1)), 10)
# ema = tf.train.ExponentialMovingAverage(.99)
# acc = chapter6_net.accuracy(outputs, labels)
# with tf.Session() as sess:
#     saver = tf.train.Saver(ema.variables_to_restore())
#     saver.restore(sess, 'model/mnist_bpnn4/chapter6.ckpt')
#     mnist = input_data.read_data_sets('data/mnist', one_hot=True)
#     acc = sess.run(acc, feed_dict={inputs: mnist.test.images, labels: mnist.test.labels})
#     print('Accuracy: %.2f%%' % (acc * 100))
#     # 保存模型图为JSON文件
#     saver.export_meta_graph('graph/mnist_bpnn/chapter6.ckpt.meta.json', as_text=True)


# # Inception-v3的最后卷积并行层
# slim = tf.contrib.slim
# # 给列表中的函数设置默认参数取值，如调用slim.conv2d会自动加上stride=1,padding='VALID'参数，除非手动指定
# with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
#     net = tf.ones((100, 96, 96, 128))  # 假设为上一层输出节点矩阵
#     with tf.variable_scope('Mixed-7c'):  # 为最后卷积并行层统一一个命名空间
#         with tf.variable_scope('Branch-0'):  # 给每个并行分支不同的命名空间
#             branch_0 = slim.conv2d(net, 320, (1, 1), scope='Conv2d_0a_1x1')  # 边长1，深度320的过滤器
#         with tf.variable_scope('Branch-1'):
#             branch_1 = slim.conv2d(net, 384, (1, 1), scope='Conv2d_0a_1x1')
#             branch_1 = tf.concat([
#                 slim.conv2d(branch_1, 384, (1, 3), scope='Conv2d_0b_1x3'),
#                 slim.conv2d(branch_1, 384, (3, 1), scope='Conv2d_0c_3x1')
#             ], axis=3)  # 把列表中的卷积结果在深度的维度上合并
#         with tf.variable_scope('Branch-2'):
#             branch_2 = slim.conv2d(net, 448, (1, 1), scope='Conv2d_0a_1x1')
#             branch_2 = slim.conv2d(branch_2, 384, (3, 3), scope='Conv2d_0b_3x3')
#             branch_2 = tf.concat([
#                 slim.conv2d(branch_2, 384, (1, 3), scope='Conv2d_0c_1x3'),
#                 slim.conv2d(branch_2, 384, (3, 1), scope='Conv2d_0d_3x1')
#             ], axis=3)
#         with tf.variable_scope('Branch-3'):
#             branch_3 = slim.avg_pool2d(net, (3, 3), scope='AvgPool_0a_3x3')
#             branch_3 = slim.conv2d(branch_3, 192, (1, 1), scope='Conv2d_0b_1x1')
#         # 当前Inception模块的最后输出是由上面四个分支在深度上合并得到的
#         net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
#         print(net.shape)


# # 图片数据预处理
# i, flower = 0, 'data\\flower_photos'
# train_img, train_label, test_img, test_label, valid_img, valid_label = [], [], [], [], [], []
# for path, dirs, files in os.walk(flower):
#     if len(dirs):
#         kinds = len(dirs)  # 种类数量
#         continue
#     for file in files:
#         img = cv2.imread(os.path.join(path, file))  # 读取图片为矩阵
#         img = cv2.resize(img, dsize=(299, 299))  # 缩放
#         label = [1 if j == i else 0 for j in range(kinds)]  # 种类标签
#         n = np.random.randint(0, 100)  # 分训练、验证、测试
#         if n < 80:
#             train_img.append(img)
#             train_label.append(label)
#         elif 80 <= n < 85:
#             valid_img.append(img)
#             valid_label.append(label)
#         else:
#             test_img.append(img)
#             test_label.append(label)
#     i += 1
# # 打乱训练集，并img和label保持一致
# state = np.random.get_state()
# np.random.shuffle(train_img)
# np.random.set_state(state)
# np.random.shuffle(train_label)
# # 保存
# train_img, train_label = np.array(train_img, dtype=np.uint8), np.array(train_label, dtype=np.uint8)
# test_img, test_label = np.array(test_img, dtype=np.uint8), np.array(test_label, dtype=np.uint8)
# valid_img, valid_label = np.array(valid_img, dtype=np.uint8), np.array(valid_label, dtype=np.uint8)
# np.save(os.path.join(flower, 'train_images'), train_img)
# np.save(os.path.join(flower, 'train_labels'), train_label)
# np.save(os.path.join(flower, 'test_images'), test_img)
# np.save(os.path.join(flower, 'test_labels'), test_label)
# np.save(os.path.join(flower, 'validation_images'), valid_img)
# np.save(os.path.join(flower, 'validation_labels'), valid_label)
# print(len(train_label), len(test_label), len(valid_label))


# # 读片读取
# train_images = np.load('data\\flower_photos\\train_images.npy')
# train_labels = np.load('data\\flower_photos\\train_labels.npy')
# print(train_images.shape, train_labels.shape)

# 自定义卷积函数
l2 = tf.contrib.layers.l2_regularizer(.001)
net = (np.random.randint(0, 255, size=(2, 30, 30, 1)) / 255 + 0.01).astype('float32')
with tf.variable_scope('conv1'):
    net0 = common.conv2d(net, depth=2, ksize=(5, 5), stride=2, padding='SAME',
                 biases_initializer=tf.zeros_initializer(),
                 weights_regularizer=l2, collection_name='losses',
                 pool=tf.nn.avg_pool, pool_size=(1, 1), pool_stride=3)
    print(net0.name, net0.shape)
with tf.variable_scope('fc1'):
    net1 = common.fc(net0, output_num=10, flatten=True, keep_prob=.8, biases_initializer=tf.zeros_initializer(),
                     weights_regularizer=l2, collection_name='losses', softmax=True)
    print(net1.name, net1.shape)
print(tf.get_collection('losses'))

# slim.conv2d 卷积+偏置项+激活
net2 = slim.conv2d(net, num_outputs=2, kernel_size=(3, 3), stride=2, padding='SAME',
                  activation_fn=tf.nn.sigmoid,
                  weights_initializer=tf.truncated_normal_initializer(stddev=.1),
                  biases_initializer=tf.truncated_normal_initializer(stddev=.1))
print(net2.name, net2.shape)
net3 = slim.conv2d(net, num_outputs=2, kernel_size=(3, 3), stride=2, padding='SAME')
print(net3.name, net3.shape)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(net0))
    print(sess.run(net1))
