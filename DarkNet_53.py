# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Dark Net 53制作"""
import tensorflow as tf


def convolutional(inputs, depth, ksize, strides=1, padding='SAME', is_train=True):
    net = tf.layers.conv2d(inputs, filters=depth, kernel_size=ksize, strides=strides, padding=padding)
    net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.nn.leaky_relu(net, alpha=0.1)
    print(net.name, net.get_shape())
    return net


def residual(inputs, depth1, depth2, is_train=True):
    net = convolutional(inputs, depth=depth1, ksize=1, is_train=is_train)  # 全卷积
    net = convolutional(net, depth=depth2, ksize=3, is_train=is_train)  # size:3*3,strides:1,padding:1
    net = tf.add(inputs, net)
    print(net.get_shape())
    return net


def DarkNet53(inputs, is_train=True):
    net = convolutional(inputs, depth=32, ksize=3, is_train=is_train)
    net = convolutional(net, depth=64, ksize=3, strides=2, is_train=is_train)
    net = residual(net, 32, 64, is_train=is_train)
    net = convolutional(net, depth=128, ksize=3, strides=2, is_train=is_train)
    for i in range(2):
        net = residual(net, 64, 128, is_train=is_train)
    net = convolutional(net, depth=256, ksize=3, strides=2, is_train=is_train)
    for i in range(8):
        net = residual(net, 128, 256, is_train=is_train)
    net = convolutional(net, depth=512, ksize=3, strides=2, is_train=is_train)
    for i in range(8):
        net = residual(net, 256, 512, is_train=is_train)
    net = convolutional(net, depth=1024, ksize=3, strides=2, is_train=is_train)
    for i in range(4):
        net = residual(net, 512, 1024, is_train=is_train)
    net = tf.layers.average_pooling2d(net, pool_size=2, strides=2)
    print(net.get_shape())
    net = tf.layers.flatten(net)
    print(net.get_shape())
    net = tf.layers.dense(net, units=1000, activation=tf.nn.leaky_relu,
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
    print(net.get_shape())
    return tf.nn.softmax(net)


inputs = tf.placeholder(tf.float32, shape=(None, 416, 416, 3))
net = DarkNet53(inputs)



# # 当BN为is_train时，加入以下代码
# global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
# loss = tf.losses.get_regularization_loss()
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=global_step)

