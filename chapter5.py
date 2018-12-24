# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
import chapter5_mnist_net

"""变量管理"""

# # 变量声明
# v1 = tf.Variable(tf.constant(1.0, shape=(1,)), name='v1')
# v2 = tf.get_variable('v2', shape=(1,), initializer=tf.constant_initializer(1.0))
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run([v1, v2]))


# # 命名空间
# v1 = tf.get_variable('v', shape=(1,), initializer=tf.random_normal_initializer())
# print(v1.name)
# with tf.variable_scope('foo'):
#     v2 = tf.get_variable('v', (1,), initializer=tf.truncated_normal_initializer())
#     print(v2.name)
# with tf.variable_scope('foo'):
#     with tf.variable_scope('bar'):
#         v3 = tf.get_variable('v', (1,), initializer=tf.zeros_initializer())
#         print(v3.name)
#         v3_ = tf.Variable(tf.ones(shape=(1,)), name='v1')
#         print(v3_.name)
#     v4 = tf.get_variable('v1', (2, 6), initializer=tf.random_uniform_initializer(maxval=10))
#     print(v4.name)
# with tf.variable_scope('', reuse=True):  # 参数reuse=True时，get_variable将只能获取已存在的变量，不能声明新变量
#     v5 = tf.get_variable('foo/v')
#     print(v2 == v5)
#     v6 = tf.get_variable('foo/bar/v')
#     print(v3 == v6)
#     v7 = tf.get_variable('foo/v1', shape=(2, 6))
#     print(v4 == v7)
# with tf.variable_scope('haha'):
#     haha1 = tf.Variable(0, name='v1')  # tf.Variable不能用reuse=True取值
#     haha2 = tf.Variable(0, name='v2')
#     haha3 = tf.Variable(0, name='v3')
#     print(haha1.name, haha2.name, haha3.name)


# # 模型保存与加载
# v1 = tf.Variable(tf.random_uniform((1, 10), minval=10, maxval=100))  # 保存前是1,10；保存后改为10,100
# v2 = tf.Variable(tf.random_normal((1, 10)))
# v3 = tf.add(v1, v2)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     saver = tf.train.Saver()
#     # 模型保存
#     print(sess.run(v3))
#     # saver.save(sess, 'model/test/model.ckpt')  # 文件自动判断并创建
#     # 模型读取
#     saver.restore(sess, 'model/test/model.ckpt')  # 读取v1的值是修改前的
#     print(sess.run((v3)))

# 加载mnist模型的滑动平均模型值
inputs, labels, outputs = chapter5_mnist_net.net()  # 引入模型
ema = tf.train.ExponentialMovingAverage(.99)  # 滑动平均模型
accuracy = chapter5_mnist_net.accuracy(outputs, labels)  # 正确率
with tf.Session() as sess:
    saver = tf.train.Saver(ema.variables_to_restore())  # 加载滑动平均参数
    saver.restore(sess, 'model/mnist_bpnn3/chapter5.ckpt')  # 加载模型
    mnist = chapter5_mnist_net.data_set()  # 加载数据集
    acc = sess.run(accuracy, feed_dict={inputs: mnist.test.images, labels: mnist.test.labels})
    print('Accuracy: %.2f%%' % (acc * 100))
    # 保存模型图为JSON文件
    saver.export_meta_graph('graph/mnist_bpnn/chapter5.ckpt.meta.json', as_text=True)
