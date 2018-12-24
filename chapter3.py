# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""
tensorflow环境搭建：
1.更新conda命令以支持环境创建和切换，conda update conda
2.通过anaconda navigator创建新环境（命令行也行）
3.切换环境conda activate 环境名
4.安装numpy包及配套包，conda install numpy
5.安装tensorflow包gpu版（1.8.0对应cuda9.0），pip install tensorflow-gpu==1.8.0
"""

import numpy as np
import tensorflow as tf

# # tensorflow测试
# a = tf.constant(value=1, dtype=tf.float32, name='a')  # 常量
# b = tf.constant(value=2, dtype=tf.float32, name='b')
# n = tf.Variable(initial_value=1, dtype=tf.float32, name='n')  # 变量
# c = tf.add(a, b, name='add')  # 加法
# m = n + b;  # 同上
# k = tf.assign(n, m)  # n的赋值 => n = (m = n + b)
# with tf.device('/gpu:0'):  # gpu上运行
#     sess = tf.InteractiveSession()
#     tf.global_variables_initializer().run()  # 初始化变量
#     print(a.graph is tf.get_default_graph())
#     print(a, b, n)
#     print(c, m, k)
#     print(sess.run(a), sess.run(b), sess.run(n), sess.run(c), sess.run(m))
#     for _ in range(3): print(sess.run(k))
#     sess.close()


# # 生成新的计算图及变量
# g1, g2 = tf.Graph(), tf.Graph()
# with g1.as_default():
#     # 在g1图中定义变量v，初始值为0
#     v = tf.get_variable(name='v', shape=(3, 4), initializer=tf.zeros_initializer(dtype=tf.uint8))
#
# with g2.as_default():
#     # g2图中定义变量v，初始值1
#     v = tf.get_variable(name='v', shape=1, initializer=tf.ones_initializer(dtype=tf.uint8))
#
# with tf.Session(graph=g1) as sess:  # 在g1图中读取变量v
#     tf.global_variables_initializer().run()  # 初始化该变量
#     with tf.variable_scope('', reuse=True):  # 变量空间
#         print(sess.run(tf.get_variable('v')))  # 读取变量
#     # print(sess.run(v))  # 和第二个v重复
#
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope('', reuse=True):
#         print(sess.run(tf.get_variable('v')))


# # 张量（a、b、c、d都为张量的引用）和会话
# with tf.device('/gpu:0'):
#     a = tf.constant(value=[4, 5, 6], name='a')
#     b = tf.constant(value=[1, 2, 3], name='b')
#     c = tf.add(a, b, name='add')
#     d = b + c
#     print(c)  # Tensor("add:0", shape=(3,), dtype=uint8, device=/device:GPU:0)
#     print(d)  # Tensor("add_1:0", shape=(3,), dtype=uint8, device=/device:GPU:0)
#     # 普通会话
#     sess = tf.Session()
#     print(sess.run(d))  # 计算方式1
#     print(d.eval(session=sess))  # 计算方式2
#     with sess.as_default():  # 计算方式3：普通会话注册为默认会话
#         print(c.eval())
#     sess.close()
#     # 直接构建默认会话
#     inter_sess = tf.InteractiveSession()
#     print(d.eval())
#     print(inter_sess.run(d))
#     inter_sess.close()


# 用ConfigProto生成会话实现并行计算
# a = tf.constant(np.random.normal(-10, 10, size=(3, 10)), name='a')
# b = tf.constant(np.random.normal(-5, 5, size=(3, 10)), name='b')
# r1 = tf.add(a, b, name='add')
# r2 = tf.subtract(a, b, name='sub')
# r3 = tf.multiply(a, b, name='mul')
# r4 = tf.div(a, b, name='div')
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True);
# sess1 = tf.InteractiveSession(config=config)
# sess2 = tf.Session(config=config)
# sess3 = tf.Session(config=config)
# print(sess1.run(r1), sess1.run(r2), sess2.run(r3), sess3.run(r4))
# sess1.close(); sess2.close(); sess3.close()



