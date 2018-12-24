# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf

"""深层神经网络：
    1.激活函数实现去线性化；
    2.多层网络实现异或运算；
    3.分类和回归是监督学习的两大类问题；
    4.交叉熵损失函数是判断输出向量和期望向量距离的常用评判函数，它刻画了两个概率分布之间的距离；
    5.不同的损失函数会对训练得到的模型产生重要影响；
    6.权重参数的初始值很大程度影响最后梯度寻优的结果；
    7.网络训练步骤：
        （1）反向传播得到损失函数对每个参数的梯度；
        （2）通过梯度和学习率，使用梯度下降更新每个参数。
    8.网络测试步骤：
        （1）前向传播得到预测值，对比与真实值差距；
        （2）通过差距得出正确率，并与正确率阈值做比较。
    9.过拟合：模型过于复杂后，记忆了训练数据的噪音而未学习训练数据中通用的趋势
    10.网络优化：
        （1）训练时使用指数衰减学习率；
        （2）损失函数加入正则化，即限制权重大小，不能拟合噪音数据；
        （3）测试数据时使用滑动平均模型。
    11.调参优化：权重和偏置项初始值、学习率衰减参数、
"""

# # 分类：交叉熵计算
# target = tf.constant([1, 0, 0], dtype=tf.float32, name='target')
# output1 = tf.constant([.5, .4, .1], name='output1')  # 目标值
# output2 = tf.constant([.8, 0, .1], name='output2')  # 输出值
# output3 = tf.constant([1.2, .1, .1], name='output2')  # 输出值
# clip1 = tf.clip_by_value(output1, clip_value_min=1e-10, clip_value_max=1)  # 值小于1e-10的为1e-10，大于1的为1
# clip2 = tf.clip_by_value(output2, clip_value_min=1e-10, clip_value_max=1)  # 值小于1e-10的为1e-10，大于1的为1
# clip3 = tf.clip_by_value(output3, clip_value_min=1e-10, clip_value_max=1)  # 值小于1e-10的为1e-10，大于1的为1
# log1 = tf.log(clip1)  # 以e为底的对数
# log2 = tf.log(clip2)
# log3 = tf.log(clip3)
# cross_entropy1 = -tf.reduce_mean(target * log1)  # 交叉熵，即目标概率分布向量与输出概率分布向量的距离
# cross_entropy2 = -tf.reduce_mean(target * log2)
# cross_entropy3 = -tf.reduce_mean(target * log3)
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# print(sess.run([clip1, clip2, clip3]))
# print(sess.run([log1, log2, log3, tf.exp(log1)]))
# print(sess.run([cross_entropy1, cross_entropy2, cross_entropy3]))
# sess.close()


# # 分类：softmax和交叉熵的损失函数
# label = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32, name='label')
# output = tf.constant([[.9, .1, 0], [.1, 1.2, .4], [.1, .1, .9]], name='output')
# softmax = tf.nn.softmax(output, axis=1)
# cross_entropy1 = -tf.reduce_sum(label * tf.log(softmax), axis=1)
# cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output)
# with tf.Session() as sess:
#     print(softmax.eval(), cross_entropy1.eval(), cross_entropy2.eval(), sep='\n')


# # 回归：平方和损失函数
# label = tf.constant([.1, .2, .3, .4, .5], dtype=tf.float32, name='label')
# output = tf.constant([.5, .4, .3, .2, .1], name='output')
# loss = tf.reduce_mean(tf.square(label - output))
# with tf.Session() as sess:
#     print(sess.run(loss))

# # 回归：greater、where
# a = tf.constant([1, 2, 3, 4])
# b = tf.constant([4, 3, 2, 1])
# g1 = a > b
# g2 = tf.greater(a, b)
# w1 = tf.where(g1, a, b)
# w2 = tf.where(g2, a, b)
# with tf.Session() as sess:
#     print(sess.run(g1), sess.run(g2), sess.run(w1), sess.run(w2), sep='\n')


# # 回归：自定义损失函数 P93
# rdm = np.random.RandomState(seed=1)
# X_train = rdm.randint(1, 10, size=(128000, 2)).astype('float32')  # 训练集输入值
# Y_train = [[x1 + x2 + rdm.rand() - 1] for x1, x2 in X_train]  # 训练集输出值，加入噪音
# X_test = rdm.randint(1, 10, size=(1280, 2))  # 测试集输入值
# Y_test = [[x1 + x2 + rdm.rand() - 1] for x1, x2 in X_test]  # 测试集输出值
# # 占位符
# x = tf.placeholder(tf.float32, shape=(None, 2), name='input')
# y = tf.placeholder(tf.float32, shape=(None, 1), name='output')
# # 前向传播：2层、relu激活
# w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1), name='w1')  # 权重
# b1 = tf.Variable(tf.random_normal((1, 3), stddev=1, seed=1), name='b1')  # 偏置项
# h = tf.nn.relu(tf.matmul(x, w1) + b1)
# w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1), name='w1')  # 权重
# o = tf.matmul(h, w2)
# # 后向传播：学习率指数衰减、自定义损失、梯度寻优
# step = tf.Variable(0)
# lr = tf.train.exponential_decay(0.005, global_step=step, decay_steps=100, decay_rate=0.9, staircase=True)
# loss = tf.reduce_mean(tf.where(y > o, (y - o) * 10, (o - y) * 5))  # 自定义损失函数
# optmizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=step)  # 梯度寻优器
# # 运行
# batch_size, batch, batch_current = 10000, 0, 0
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     while True:
#         sess.run(optmizer, feed_dict={  # 训练
#             x: X_train[batch_current:batch_current + batch_size],
#             y: Y_train[batch_current:batch_current + batch_size]
#         })
#         batch_current += batch_size
#         if batch_current >= 128000: batch_current = 0
#         if batch % 100 == 0:
#             loss_ = sess.run([loss, lr], feed_dict={x: X_test, y: Y_test})
#             print('Batch %d：loss %.4f，learing_rate %.4f' % (batch, loss_[0], loss_[1]))
#             if loss_[0] < 1.69:
#                 print(sess.run(w1), sess.run(b1), sess.run(w2), sep='\n')
#                 break
#         batch += 1


# # L1和L2正则化
# w = tf.constant([[1, -2], [-3, 4]], dtype=tf.float32)
# l1 = tf.contrib.layers.l1_regularizer(.5)(w)  # 0.5为损失系数
# l2 = tf.contrib.layers.l2_regularizer(.5)(w)
# with tf.Session() as sess:
#     print(sess.run(l1), sess.run(l2))


# # n层神经网络+正则化
# layer_dim = [5, 10, 7, 3, 1]  # 每层网络个数
# layer_len = len(layer_dim)  # 网络层数
# train_x = np.random.randint(1, 20, (100000, layer_dim[0]))
# train_y = [[x1-x2+3*(x3+x4)+x5-np.random.rand()] for x1, x2, x3, x4, x5 in train_x]
# test_x = np.random.randint(1, 20, (10000, layer_dim[0]))
# test_y = [[x1-x2+3*(x3+x4)+x5-np.random.rand()] for x1, x2, x3, x4, x5 in test_x]
# # 占位符
# x = tf.placeholder(tf.float32, shape=(None, layer_dim[0]))
# y = tf.placeholder(tf.float32, shape=(None, layer_dim[-1]))
# # 前向传播
# nodes = x  # 输入层节点
# for i in range(layer_len - 1):
#     w = tf.Variable(tf.random_normal((layer_dim[i], layer_dim[i + 1]), stddev=.8, seed=1))
#     tf.add_to_collection('losses', value=tf.contrib.layers.l2_regularizer(0.001)(w))  # L2正则项加入集合
#     b = tf.Variable(tf.random_normal((1, layer_dim[i + 1]), stddev=1, seed=1))
#     nodes = tf.nn.relu(tf.matmul(nodes, w) + b)  # 下一层节点
# # 后向传播
# tf.add_to_collection('losses', value=tf.reduce_mean(tf.square(y - nodes)))  # 均方误差加入集合
# loss = tf.add_n(tf.get_collection(key='losses'))  # 得到损失函数的所有成员，然后累加(add_n)
# gs = tf.Variable(0)
# lr = tf.train.exponential_decay(0.01, gs, 300, 0.96)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=gs)
# # 运算
# batch_size, batch, batch_now = 5000, 0, 0
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     while True:
#         sess.run(optimizer, feed_dict={  # 训练
#             x: train_x[batch_now:batch_now + batch_size],
#             y: train_y[batch_now:batch_now + batch_size]
#         })
#         batch_now += batch_size
#         if batch_now >= 100000: batch_now = 0
#         if batch % 100 == 0:  # 测试
#             loss_ = sess.run(loss, feed_dict={x: test_x, y: test_y})
#             print('Batch %d: learning_rate %.4f regular_loss %.3f' % (batch, sess.run(lr), loss_))
#             if loss_ < .149: break
#         batch += 1


# 滑动平均模型：(1)先算min(decay, (1+step)/(10+step)); (2)再算shadow_v=decay*shadow_v+(1-decay)*v
v = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, dtype=tf.uint16, trainable=False)  # 控制衰减率的变量
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)  # 滑动平均模型
maintain_averages_op = ema.apply([v])  # 定义一个更新变量的滑动平均操作的列表
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run([v, ema.average(v)]))  # v: 0, step: 0, shadow_v: 0 => [0, 0]
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v, ema.average(v)]))  # v: 10, step: 0, shadow_v: 0 => [10, 9]
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v, 20))
    sess.run(maintain_averages_op)
    print(sess.run([v, ema.average(v)]))  # v: 20, step: 10000, shadow_v: 9 => [20, 9.11]
    sess.run(maintain_averages_op)
    print(sess.run([v, ema.average(v)]))  # v: 20, step: 10000, shadow_v: 9.11 => [20, 9.2189]

