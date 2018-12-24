# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf

# tensorflow实现BP神经网络预测mnist手写数字
# 网络参数
config = {
    'num': {'i': 28 ** 2, 'h1': 128, 'h2': 96, 'o': 10},  # 每层神经元数
    'batch': {'size': 5000, 'display': 100, 'lr-change': 1000},  # 迭代参数
    'learing-rate': 0.003,  # 学习率
    'accuracy': 0.978,  # 目标正确率
}

# 占位符
feature = tf.placeholder(dtype=tf.float32, shape=(None, config['num']['i']), name='feature')  # 特征向量
target = tf.placeholder(dtype=tf.float32, shape=(None, config['num']['o']), name='target')  # 预测结果

# 网络函数
def weights(shape, mean=0.0, stddev=0.1, dtype=tf.float32, name=None):
    return tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, name=name)

def bias(value, dtype=tf.float32, name=None):
    return tf.Variable(value, dtype=dtype, name=name)

def targets(number, n):
    t = np.ones((n, config['num']['o'])) * 0.01
    t[[i for i in range(n)], [number.astype('int')]] = 0.99
    return t

def FP(feature):
    # 输入层
    with tf.variable_scope('input') as scope:
        # feature = tf.reshape(feature, shape=(-1, config['num']['i']))  # -1未知，需动态计算
        weight = tf.Variable(weights((config['num']['i'], config['num']['h1'])), name='weight')
    # 隐层1
    with tf.variable_scope('hidden1') as scope:
        neure = tf.matmul(feature, weight) + bias(0, name='bias')
        neure = tf.sigmoid(neure, name='neure')
        weight = tf.Variable(weights((config['num']['h1'], config['num']['h2'])), name='weight')
    # 隐层2
    with tf.variable_scope('hidden2') as scope:
        neure = tf.matmul(neure, weight) + bias(0, name='bias')
        neure = tf.sigmoid(neure, name='neure')
        weight = tf.Variable(weights((config['num']['h2'], config['num']['o'])), name='weight')
    # 输出层
    with tf.variable_scope('output') as scope:
        neure = tf.matmul(neure, weight) + bias(0, name='bias')
        output = tf.sigmoid(neure, name='neure')
    return output

# 前向传播
output = FP(feature)

# 反向传播：平方和损失函数及梯度下降
loss = tf.reduce_mean(tf.square(target - output), name='loss')
train = tf.train.AdamOptimizer(learning_rate=config['learing-rate']).minimize(loss)

# 模型正确率
pred = tf.equal(tf.argmax(output, axis=1), tf.argmax(target, axis=1))  # 不能用==运算符，成了两个张量的比较
predict = tf.reduce_mean(tf.cast(pred, tf.float32), name='predict')

# 运行
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 数据集
    train_data, test_data = np.load('data/mnist/mnist_train.npy'), np.load('data/mnist/mnist_test.npy')
    idx, train_len, test_len, epoch = 0, len(train_data), len(test_data), 0
    # 模型持久化
    saver = tf.train.Saver()
    while True:
        # 模型训练
        data = train_data[idx:idx + config['batch']['size']]
        re = sess.run(train, feed_dict={
            feature: data[:, 1:],
            target: targets(data[:, 0], data[:, 0].shape[0])
        })
        idx += config['batch']['size']
        if idx >= train_len: idx = 0
        if epoch % config['batch']['display'] == 0:
            # 模型预测
            re = sess.run(predict, feed_dict={
                feature: test_data[:, 1:],
                target: targets(test_data[:, 0], test_len)
            })
            print('Epoch %d: accuracy %.2f%%, learing-rate %.6f' % (epoch, re * 100, config['learing-rate']))
            if re > config['accuracy']:
                saver.save(sess, 'model/mnist_bpnn1/')
                break
        epoch += 1
        if epoch % config['batch']['lr-change'] == 0:
            config['learing-rate'] *= 0.9 ** (epoch // config['batch']['lr-change'])
