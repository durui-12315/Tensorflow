# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""ResNet101网络结构TF实现"""

import tensorflow as tf


def conv2d(name, is_train, inputs, depth, ksize=(1, 1, 3), strides=1, padding='VALID',
           use_activation=True, activation_fn=tf.nn.relu, regularizer=None,
           weights_initializer=None, biases_initializer=None):
    with tf.variable_scope(name, reuse=not is_train):
        if not is_train:
            w, b = tf.get_variable('w'), tf.get_variable('b')
        else:
            w_init = weights_initializer if weights_initializer else tf.truncated_normal_initializer()
            b_init = biases_initializer if biases_initializer else tf.zeros_initializer()
            w = tf.get_variable('w', shape=(ksize[0], ksize[1], ksize[2], depth), dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=(depth,), dtype=tf.float32, initializer=b_init)
        net = tf.nn.conv2d(inputs, filter=w, strides=(1, strides, strides, 1), padding=padding)
        if regularizer and is_train:
            tf.add_to_collection(name='losses', value=regularizer(w))
        if use_activation:
            net = activation_fn(tf.nn.bias_add(net, b))
        else:
            net = tf.nn.bias_add(net, b)
        return net


def maxpool(name, input, ksize=3, strides=2, padding='VALID'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input, ksize=(1, ksize, ksize, 1), strides=(1, strides, strides, 1),
                              padding=padding)


def fc(scope, is_train, inputs, size, output_num, flatten=False, activation_fn=tf.nn.relu, keep_prob=1.0,
       weights_initializer=None, biases_initializer=None, softmax=False, regularizer=None):
    # 矩阵转换向量(扁平化)
    if flatten: inputs = tf.reshape(inputs, shape=(-1, size))
    with tf.variable_scope(scope, reuse=not is_train):
        # 全连接参数
        if is_train:
            w_init = weights_initializer if weights_initializer else tf.truncated_normal_initializer()
            b_init = biases_initializer if biases_initializer else tf.truncated_normal_initializer()
            w = tf.get_variable('w', shape=(size, output_num), dtype=tf.float32, initializer=w_init)
            # 加入正则项
            if regularizer: tf.add_to_collection('losses', value=regularizer(w))
            # 偏置项
            b = tf.get_variable('b', shape=(output_num,), dtype=tf.float32, initializer=b_init)
        else:
            w, b = tf.get_variable('w'), tf.get_variable('b')
        # 激励
        net = activation_fn(tf.matmul(inputs, w) + b)
        # dropout
        if 0 < keep_prob < 1: net = tf.nn.dropout(net, keep_prob=keep_prob)
        # softmax
        if softmax: net = tf.nn.softmax(net)
        return net


def Stem(inputs, is_train=True, regularizer=None):
    with tf.variable_scope('Stem'):
        net = conv2d('conv1', is_train, inputs, depth=32, ksize=(3, 3, 3), strides=2, regularizer=regularizer)
        net = conv2d('conv2', is_train, net, depth=32, ksize=(3, 3, 32), regularizer=regularizer)
        net = conv2d('conv3', is_train, net, depth=64, ksize=(3, 3, 32), padding='SAME', regularizer=regularizer)
        net = tf.concat([
            maxpool('maxpool1', net),
            conv2d('conv4', is_train, net, depth=96, ksize=(3, 3, 64), strides=2, regularizer=regularizer)
        ], axis=3, name='concat1')
        net1 = conv2d('conv5_a1', is_train, net, depth=64, ksize=(1, 1, 160), regularizer=regularizer)
        net1 = conv2d('conv5_a2', is_train, net1, depth=96, ksize=(3, 3, 64), regularizer=regularizer)
        net2 = conv2d('conv6_b1', is_train, net, depth=64, ksize=(1, 1, 160), regularizer=regularizer)
        net2 = conv2d('conv6_b2', is_train, net2, depth=64, ksize=(7, 1, 64), padding='SAME', regularizer=regularizer)
        net2 = conv2d('conv6_b3', is_train, net2, depth=64, ksize=(1, 7, 64), padding='SAME', regularizer=regularizer)
        net2 = conv2d('conv6_b4', is_train, net2, depth=96, ksize=(3, 3, 64), regularizer=regularizer)
        net = tf.concat([net1, net2], axis=3, name='concat2')
        net = tf.concat([
            conv2d('conv7', is_train, net, depth=192, ksize=(3, 3, 192), strides=2, regularizer=regularizer),
            maxpool('maxpool2', net, ksize=2)
        ], axis=3, name='concat3')
    return net


def Inception_resnet_A(scope, inputs, is_train=True, regularizer=None):
    with tf.variable_scope(scope):
        net = tf.nn.relu(inputs)
        net1 = conv2d('conv1_a1', is_train, net, depth=32, ksize=(1, 1, 384), regularizer=regularizer)
        net2 = conv2d('conv2_b1', is_train, net, depth=32, ksize=(1, 1, 384), regularizer=regularizer)
        net2 = conv2d('conv2_b2', is_train, net2, depth=32, ksize=(3, 3, 32), padding='SAME', regularizer=regularizer)
        net3 = conv2d('conv3_c1', is_train, net, depth=32, ksize=(1, 1, 384), regularizer=regularizer)
        net3 = conv2d('conv3_c2', is_train, net3, depth=48, ksize=(3, 3, 32), padding='SAME', regularizer=regularizer)
        net3 = conv2d('conv3_c3', is_train, net3, depth=64, ksize=(3, 3, 48), padding='SAME', regularizer=regularizer)
        net_ = tf.concat([net1, net2, net3], axis=3)
        net_ = conv2d('conv4', is_train, net_, depth=384, ksize=(1, 1, 128), use_activation=False, regularizer=regularizer)
        net = tf.add(net, net_)  # 跳连结构
    return net


def Reduction_A(inputs, is_train=True, regularizer=None):
    with tf.variable_scope('Reduction_A'):
        net1 = maxpool('maxpool', inputs)
        net2 = conv2d('conv1_a1', is_train, inputs, 256, ksize=(3, 3, 384), strides=2, regularizer=regularizer)
        net3 = conv2d('conv2_b1', is_train, inputs, 256, ksize=(1, 1, 384), regularizer=regularizer)
        net3 = conv2d('conv2_b2', is_train, net3, 256, ksize=(3, 3, 256), regularizer=regularizer)
        net3 = conv2d('conv2_b3', is_train, net3, 256, ksize=(3, 3, 256), strides=2, padding='SAME', regularizer=regularizer)
        net = tf.concat([net1, net2, net3], axis=3)
    return net


def Inception_resnet_B(scope, inputs, is_train=True, regularizer=None):
    with tf.variable_scope(scope):
        net1 = conv2d('conv1_a1', is_train, inputs, depth=192, ksize=(1, 1, 896), regularizer=regularizer)
        net2 = conv2d('conv2_a1', is_train, inputs, depth=128, ksize=(1, 1, 896), regularizer=regularizer)
        net2 = conv2d('conv2_a2', is_train, net2, depth=160, ksize=(1, 7, 128), padding='SAME', regularizer=regularizer)
        net2 = conv2d('conv2_a3', is_train, net2, depth=192, ksize=(7, 1, 160), padding='SAME', regularizer=regularizer)
        net = tf.concat([net1, net2], axis=3)
        net = conv2d('conv3', is_train, net, depth=896, ksize=(1, 1, 384), use_activation=False, regularizer=regularizer)
        net = tf.nn.relu(inputs + net)
    return net


def Reduction_B(inputs, is_train=True, regularizer=None):
    with tf.variable_scope('Reduction_B'):
        net1 = maxpool('maxpool', inputs)
        net2 = conv2d('conv1_a1', is_train, inputs, depth=256, ksize=(1, 1, 896), regularizer=regularizer)
        net2 = conv2d('conv1_a2', is_train, net2, depth=384, ksize=(3, 3, 256), strides=2, regularizer=regularizer)
        net3 = conv2d('conv2_b1', is_train, inputs, depth=256, ksize=(1, 1, 896), regularizer=regularizer)
        net3 = conv2d('conv2_b2', is_train, net3, depth=288, ksize=(3, 3, 256), strides=2, regularizer=regularizer)
        net4 = conv2d('conv3_c1', is_train, inputs, depth=256, ksize=(1, 1, 896), regularizer=regularizer)
        net4 = conv2d('conv3_c2', is_train, net4, depth=288, ksize=(3, 3, 256), regularizer=regularizer)
        net4 = conv2d('conv3_c3', is_train, net4, depth=320, ksize=(3, 3, 288), strides=2, padding='SAME', regularizer=regularizer)
        net = tf.concat([net1, net2, net3, net4], axis=3)
    return net


def Inception_resnet_C(scope, inputs, is_train=True, regularizer=None):
    with tf.variable_scope(scope):
        net1 = conv2d('conv1_a1', is_train, inputs, depth=192, ksize=(1, 1, 1888), regularizer=regularizer)
        net2 = conv2d('conv2_b1', is_train, inputs, depth=192, ksize=(1, 1, 1888), regularizer=regularizer)
        net2 = conv2d('conv2_b2', is_train, net2, depth=224, ksize=(1, 3, 192), padding='SAME', regularizer=regularizer)
        net2 = conv2d('conv2_b3', is_train, net2, depth=256, ksize=(3, 1, 224), padding='SAME', regularizer=regularizer)
        net = tf.concat([net1, net2], axis=3)
        net = conv2d('conv3', is_train, net, depth=1888, ksize=(1, 1, 448), use_activation=False, regularizer=regularizer)
        net = tf.nn.relu(inputs + net)
    return net


# 残差网络结构resnet101层构建，输入维度：batch * 299 * 299 * 3
def ResNet_101(inputs, is_train):
    l2 = tf.contrib.layers.l2_regularizer(0.001)
    # Stem模块
    net = Stem(inputs, is_train, regularizer=l2)
    # 5个Inception_resnet_A模块
    for i in range(1, 6):
        net = Inception_resnet_A('Inception_resnet_A%d' % i, net, is_train, regularizer=l2)
    net = tf.nn.relu(net)
    # Reduction-A模块
    net = Reduction_A(net, is_train, regularizer=l2)
    # 10个Inception_resnet_A模块
    for i in range(1, 11):
        net = Inception_resnet_B('Inception_resnet_B%d' % i, net, is_train, regularizer=l2)
    # Reduction-B模块
    net = Reduction_B(net, is_train, regularizer=l2)
    # 5个Inception_resnet_C模块
    for i in range(1, 6):
        net = Inception_resnet_C('Inception_resnet_C%d' % i, net, is_train, regularizer=l2)
    # 平均池化+dropout
    with tf.variable_scope('avgpool'):
        net = tf.nn.avg_pool(net, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='VALID')
    # 全连接
    return fc('softmax', is_train, net, size=7 * 7 * 1888, output_num=1000, flatten=True, keep_prob=0.8,
             softmax=True, regularizer=l2)


# 网络输入
inputs = tf.placeholder(tf.float32, shape=(10, 299, 299, 3), name='Input')
outputs = ResNet_101(inputs, is_train=True)
print(outputs.shape)
