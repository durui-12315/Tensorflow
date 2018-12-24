# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""
特征向量：描述实体特征的数字组合所组成的向量
神经网络分类步骤：
    1.提取问题中实体的特征向量作为神经网络的输入；
    2.定义神经网络的结构，以及定义从输入得到输出；
    3.通过反向传播、梯度下降调整神经网络的权值；
    4.使用训练好的神经网络来训练数据。
神经网络权值初始化：一般为满足正态分布的随机数
"""

import numpy as np
import tensorflow as tf

# # 随机函数
# n1 = tf.random_normal(shape=(3, 6), mean=1, stddev=2, dtype=tf.float64, seed=77, name='rn')  # 正太分布
# n2 = tf.truncated_normal(shape=(3, 6), mean=0, stddev=1, dtype=tf.float64, seed=88, name='tn')  # 完全正太分布
# n3 = tf.random_uniform(shape=(1, 10), minval=0, maxval=10, dtype=np.float32, seed=66, name='ru')  # 均匀分布
# n4 = tf.random_gamma(shape=(1, 10), alpha=0.1, beta=1.0, dtype=tf.float32, seed=55, name='rg')  # 伽马分布
# # 常量函数
# c1 = tf.zeros(shape=(2, 5), dtype=tf.uint8, name='z')  # 零
# c2 = tf.ones(shape=(2, 5), dtype=tf.uint8, name='o')  # 一
# c3 = tf.fill(dims=(3, 8), value=8, name='f')  # 单值填充
# c4 = tf.constant(value=np.random.binomial(n=10, p=0.3, size=(2, 7)), dtype=tf.float32, name='c')  # 多值常量
# # 变量
# v1 = tf.Variable(initial_value=n2, name='v1')
# v2 = tf.Variable(initial_value=v1.initialized_value() * 2, name='v2')
# # 随机和常数生成函数
# sess = tf.InteractiveSession()
# # tf.global_variables_initializer().run()  # 初始化所有变量
# tf.initialize_all_variables().run()
# print(n1.eval(), n2.eval(), n3.eval(), n4.eval(), sep='\n')
# print(c1.eval(), c2.eval(), c3.eval(), c4.eval(), sep='\n')
# print(sess.run(v1), sess.run(v2), sep='\n')
# sess.close()


# # 前向传播1
# weights = {'i': 28 ** 2, 'h1': 196, 'h2': 128, 'o': 10}
# bias = {'i': 0.1, 'h1': 0.1, 'h2': 0.1}
# # 输入层及权值
# featrues = np.random.randint(0, 256, size=(weights['i'], 1)) / 255 * 0.98 + 0.01
# input_i = tf.constant(featrues, dtype=tf.float32)
# input_w = tf.Variable(tf.random_normal((weights['h1'], weights['i']), mean=0, stddev=1, seed=1))
# input_o = tf.matmul(input_w, input_i)  # 矩阵乘法
# input_o += bias['i']
# # 隐藏层及权值
# hidden1_w = tf.Variable(tf.random_normal((weights['h2'], weights['h1']), mean=0, stddev=1, seed=1))
# hidden1_o = tf.matmul(hidden1_w, input_o) + bias['h1']
# hidden2_w = tf.Variable(tf.random_normal((weights['o'], weights['h2']), mean=0, stddev=1, seed=1))
# output = tf.matmul(hidden2_w, hidden1_o) + bias['h2']
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# print(sess.run(output))
# sess.close()


# # 前向传播2
# weights = {'i': 28 ** 2, 'h1': 196, 'h2': 128, 'o': 10}  # 神经元结构
# bias = {'i': 0.1, 'h1': -0.1, 'h2': 0.1}  # 偏置项结构
# # 输入层
# feature = tf.placeholder(dtype=tf.float32, shape=(1, weights['i']), name='feature')  # 特征向量
# input_w = tf.Variable(tf.random_normal((weights['i'], weights['h1']), mean=0, stddev=1), name='input')  # 权值
# hidden1_n = tf.matmul(feature, input_w) + bias['i']  # 隐层1神经元
# hidden1_n = tf.sigmoid(hidden1_n)  # sigmoid激活函数
# # 隐藏层
# hidden1_w = tf.Variable(tf.random_normal((weights['h1'], weights['h2']), mean=0, stddev=1), name='hidden1')  # 权值1
# hidden2_n = tf.matmul(hidden1_n, hidden1_w) + bias['h1']  # 隐层2神经元
# hidden2_n = tf.sigmoid(hidden2_n)
# hidden2_w = tf.Variable(tf.random_normal((weights['h2'], weights['o']), mean=0, stddev=1), name='hidden2')  # 权值2
# # 输出层
# output_n = tf.matmul(hidden2_n, hidden2_w) + bias['h2']  # 输出层神经元
# output_n = tf.sigmoid(output_n)
# # 计算
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# for _ in range(6):
#     re = sess.run(output_n, feed_dict={
#        feature: np.random.randint(0, 256, size=(1, weights['i'])) / 255 * 0.98 + 0.01  # 输入特征向量
#     })
#     print(re)
# sess.close()


# 平方和损失
a = tf.Variable(tf.random_uniform((1, 3), 1, 10, seed=1))
b = tf.Variable(tf.random_uniform((1, 3), 1, 10, seed=2))
c = a - b
d = tf.square(a - b)
sum = tf.reduce_sum(d)
mean = tf.reduce_mean(d)
loss = tf.reduce_mean(tf.square(a - b))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a), sess.run(b), sess.run(c), sess.run(d), sess.run(sum), sess.run(mean), sess.run(loss), sep='\n')


