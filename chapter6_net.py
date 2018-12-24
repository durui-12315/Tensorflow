# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf


# LeNet-5模型
def LeNet_5(inputs, output_size, regularizer=None):
    print(inputs.shape)
    # 第一层卷积：输入32*32*1，卷积5*5*6，步长1，填充0
    with tf.variable_scope('conv1'):
        w = tf.get_variable('w', shape=(5, 5, 1, 6), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(6,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(inputs, filter=w, strides=(1, 1, 1, 1), padding='SAME', name='conv')
        net = tf.nn.bias_add(net, b, name='bias')
        net = tf.nn.relu(net, name='active')  # nodes:28*28*6, links:nodes*(5*5+1)
        print(net.name, net.shape)
    # 第一层最大池化：输入28*28*6，池化2*2*1，步长2*2*1
    with tf.variable_scope('pool1'):
        # nodes:14*14*6, links:nodes
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='pool')
        print(net.name, net.shape)
    # 第二层卷积：输入14*14*6，卷积5*5*16，步长1，填充0
    with tf.variable_scope('conv2'):
        w = tf.get_variable('w', shape=(5, 5, 6, 16), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(16,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='VALID', name='conv')
        net = tf.nn.bias_add(net, b, name='bias')
        net = tf.nn.relu(net, name='active')  # nodes:10*10*16, links:nodes*(5*5+1)
        print(net.name, net.shape)
    # 第二层池化：输入10*10*16，池化2*2*1，步长2*2*1
    with tf.variable_scope('pool2'):
        # nodes:5*5*16, links:nodes
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='pool')
        print(net.name, net.shape)
    # 全连接转换
    net = tf.reshape(net, shape=(-1, net.shape[1] * net.shape[2] * net.shape[3]))
    print(net.name, net.shape)
    # 第三层全连接：输入400
    with tf.variable_scope('fc1'):
        w = tf.get_variable('w', shape=(net.shape[1], 120), initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:  # 只有全连接层的权重需要加入正则化
            tf.add_to_collection('losses', value=regularizer(w))
        b = tf.get_variable('b', shape=(120,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b, name='active')
        if regularizer:  # 只有在训练时使用dropout，一般在全连接层
            net = tf.nn.dropout(net, keep_prob=.8)
        print(net.name, net.shape)
    # 第四层全连接：输入120
    with tf.variable_scope('fc2'):
        w = tf.get_variable('w', shape=(net.shape[1], 84), initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:  # 只有全连接层的权重需要加入正则化
            tf.add_to_collection('losses', value=regularizer(w))
        b = tf.get_variable('b', shape=(84,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b, name='active')
        print(net.name, net.shape)
    # 第五层输出：输入84
    with tf.variable_scope('output'):
        w = tf.get_variable('w', shape=(net.shape[1], output_size),
                            initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:  # 只有全连接层的权重需要加入正则化
            tf.add_to_collection('losses', value=regularizer(w))
        b = tf.get_variable('b', shape=(output_size,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b, name='active')
        print(net.name, net.shape)
    return net


# 滑动平均模型
def average(inputs, num_updates):
    ema = tf.train.ExponentialMovingAverage(.99, num_updates=num_updates)
    average_op = ema.apply(var_list=tf.trainable_variables())
    with tf.variable_scope('conv1', reuse=True):
        net = tf.nn.conv2d(inputs, filter=ema.average(tf.get_variable('w')), strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, ema.average(tf.get_variable('b'))))
    with tf.variable_scope('pool1', reuse=True):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
    with tf.variable_scope('conv2', reuse=True):
        net = tf.nn.conv2d(net, filter=ema.average(tf.get_variable('w')), strides=(1, 1, 1, 1), padding='VALID')
        net = tf.nn.relu(tf.nn.bias_add(net, ema.average(tf.get_variable('b'))))
    with tf.variable_scope('pool2', reuse=True):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
    net = tf.reshape(net, shape=(-1, net.shape[1] * net.shape[2] * net.shape[3]))
    with tf.variable_scope('fc1', reuse=True):
        net = tf.nn.relu(tf.matmul(net, ema.average(tf.get_variable('w'))) + ema.average(tf.get_variable('b')))
    with tf.variable_scope('fc2', reuse=True):
        net = tf.nn.relu(tf.matmul(net, ema.average(tf.get_variable('w'))) + ema.average(tf.get_variable('b')))
    with tf.variable_scope('output', reuse=True):
        net = tf.nn.relu(tf.matmul(net, ema.average(tf.get_variable('w'))) + ema.average(tf.get_variable('b')))
    return average_op, net


# VGGNet模型
def VGGNet(regularizer=None):
    inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    lables = tf.placeholder(tf.float32, shape=(None, 1000))
    print(inputs.shape)
    with tf.variable_scope('layer1-conv'):
        w = tf.get_variable('w', shape=(3, 3, 3, 64), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(64,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(inputs, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer1-pool'):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        print(net.name, net.shape)
    with tf.variable_scope('layer2-conv'):
        w = tf.get_variable('w', shape=(3, 3, 64, 128), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(128,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer2-pool'):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        print(net.name, net.shape)
    with tf.variable_scope('layer3-conv1'):
        w = tf.get_variable('w', shape=(3, 3, 128, 256), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(256,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer3-conv2'):
        w = tf.get_variable('w', shape=(3, 3, 256, 256), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(256,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer3-pool'):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        print(net.name, net.shape)
    with tf.variable_scope('layer4-conv1'):
        w = tf.get_variable('w', shape=(3, 3, 256, 512), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(512,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer4-conv2'):
        w = tf.get_variable('w', shape=(3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(512,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer4-pool'):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        print(net.name, net.shape)
    with tf.variable_scope('layer5-conv1'):
        w = tf.get_variable('w', shape=(3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(512,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer5-conv2'):
        w = tf.get_variable('w', shape=(3, 3, 512, 512), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(512,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.conv2d(net, filter=w, strides=(1, 1, 1, 1), padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, b))
        print(net.name, net.shape)
    with tf.variable_scope('layer5-pool'):
        net = tf.nn.max_pool(net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        print(net.name, net.shape)
    net = tf.reshape(net, shape=(net.shape[0], net.shape[1] * net.shape[2] * net.shape[3]))  # 全连接转化
    with tf.variable_scope('layer6-fc'):
        w = tf.get_variable('w', shape=(net.shape[1], 4096), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(4096,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b)
        if regularizer:
            tf.add_to_collection('losses', value=regularizer(w))
            tf.nn.dropout(net, keep_prob=.5)
        print(net.name, net.shape)
    with tf.variable_scope('layer7-fc'):
        w = tf.get_variable('w', shape=(net.shape[1], 4096), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(4096,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b)
        if regularizer:
            tf.add_to_collection('losses', value=regularizer(w))
            tf.nn.dropout(net, keep_prob=.5)
        print(net.name, net.shape)
    with tf.variable_scope('layer8-fc'):
        w = tf.get_variable('w', shape=(net.shape[1], 1000), initializer=tf.truncated_normal_initializer(stddev=.1))
        b = tf.get_variable('b', shape=(1000,), initializer=tf.truncated_normal_initializer(stddev=.1))
        net = tf.nn.relu(tf.matmul(net, w) + b)
        if regularizer:
            tf.add_to_collection('losses', value=regularizer(w))
        print(net.name, net.shape)
    with tf.variable_scope('layer9-output'):
        net = tf.nn.softmax(net)
        print(net.name, net.shape)
    return inputs, lables, net


# 正确率
def accuracy(outputs, labels):
    acc = tf.equal(tf.argmax(outputs, axis=1), tf.argmax(labels, axis=1))
    return tf.reduce_mean(tf.cast(acc, tf.float32))
