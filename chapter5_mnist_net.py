# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 数据集
def data_set():
    return input_data.read_data_sets('data/mnist', one_hot=True)

# 模型构建
def net(regularizer=None):
    # 占位符
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    labels = tf.placeholder(tf.float32, shape=(None, 10))
    # 输入层
    with tf.variable_scope('input'):
        w = tf.get_variable('w', shape=(784, 384), initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:
            tf.add_to_collection(name='losses', value=regularizer(w))  # 加入损失函数中
        b = tf.get_variable('b', shape=(1, 384), initializer=tf.truncated_normal_initializer(stddev=.1))
    # 隐藏层1
    with tf.variable_scope('hidden1'):
        nodes = tf.nn.relu(tf.matmul(inputs, w) + b)
        w = tf.get_variable('w', shape=(384, 256), initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:
            tf.add_to_collection(name='losses', value=regularizer(w))
        b = tf.get_variable('b', shape=(1, 256), initializer=tf.truncated_normal_initializer(stddev=.1))
    # 隐藏层2
    with tf.variable_scope('hidden2'):
        nodes = tf.nn.relu(tf.matmul(nodes, w) + b)
        w = tf.get_variable('w', shape=(256, 10), initializer=tf.truncated_normal_initializer(stddev=.1))
        if regularizer:
            tf.add_to_collection(name='losses', value=regularizer(w))
        b = tf.get_variable('b', shape=(1, 10), initializer=tf.truncated_normal_initializer(stddev=.1))
    # 输出层
    return inputs, labels, tf.nn.relu(tf.matmul(nodes, w) + b)


# 滑动平均模型构建
def average_net(inputs, num_updates):
    ema = tf.train.ExponentialMovingAverage(.99, num_updates=num_updates)
    average_op = ema.apply(var_list=tf.trainable_variables())  # 没有指定trainable=False的变量
    with tf.variable_scope('input', reuse=True):  # 获取变量
        w, b = tf.get_variable('w'), tf.get_variable('b')
        average_hiddens = tf.nn.relu(tf.matmul(inputs, ema.average(w)) + ema.average(b))
    with tf.variable_scope('hidden1', reuse=True):
        w, b = tf.get_variable('w'), tf.get_variable('b')
        average_hiddens = tf.nn.relu(tf.matmul(average_hiddens, ema.average(w)) + ema.average(b))
    with tf.variable_scope('hidden2', reuse=True):
        w, b = tf.get_variable('w'), tf.get_variable('b')
        average_outputs = tf.nn.relu(tf.matmul(average_hiddens, ema.average(w)) + ema.average(b))
    return average_op, average_outputs


# 预测正确率
def accuracy(outputs, labels):
    predict = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(predict, tf.float32))
