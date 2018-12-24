# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Tensorboard:
    1.把神经网络的图输出到指定日志文件
    2.配置tensorboard环境变量，在控制台输入tensorboard --logdir=日志路径，然后复制其网址在谷歌浏览器中显示
    3.name_scope与variable_scope相同，区别在于前者对tf.get_variable不起作用
    4.tf.summary.scalar 标量监控数据随迭代变化的趋势
    5.tf.summary.histogram 张量分布监控数据随迭代变化的趋势
    6.tf.summary.image/audio/text 图片/音频/文本数据
"""

# # 在tensorboard中查看图
# import tensorflow as tf
#
# inputs = tf.placeholder(tf.float32, shape=(None, 300, 300, 3), name='inputs')
# labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
# with tf.variable_scope('convolution'):
#     inputs = tf.reshape(inputs, shape=(-1, 300, 300, 3))
#     net = tf.layers.conv2d(inputs, filters=16, kernel_size=3, strides=2, activation=tf.nn.relu)
#     net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
# with tf.variable_scope('fc') as scope:
#     net = tf.layers.flatten(net)
#     net = tf.layers.dense(net, units=1000, activation=tf.nn.relu)
# with tf.variable_scope('output') as scope:
#     output = tf.layers.dense(net, units=6)
# with tf.name_scope('loss') as scope:
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output))
#     train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# with tf.name_scope('accuracy') as scope:
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), labels), dtype=tf.float32))
# # 当前计算图输出到日志文件
# writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())
# writer.close()


# # 监控指标可视化
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# # 生成变量监控信息并定义生成监控日志的操作
# def variable_summaries(var, name):
#     with tf.name_scope('summaries'):
#         tf.summary.histogram(name, values=var)  # 记录张量中元素的取值分布
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean/'+name, tensor=mean)  # 生成平均值信息日志
#         stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev/'+name, tensor=stddev)  # 生成标准差信息日志
#
# # 生成一层全连接神经网络
# def dense(inputs, ipt_dim, opt_dim, layer_name, act=tf.nn.relu):
#     with tf.name_scope(layer_name):
#         # 声明权值，并生成监控日志
#         with tf.name_scope('weights'):
#             w = tf.Variable(tf.truncated_normal((ipt_dim, opt_dim), stddev=0.1))
#             variable_summaries(w, layer_name+'/weights')
#         # 声明偏置项，并生成监控日志
#         with tf.name_scope('bias'):
#             b = tf.Variable(tf.zeros((opt_dim,)))
#             variable_summaries(b, layer_name+'/bias')
#         with tf.name_scope('matmul'):
#             preact = tf.matmul(inputs, w) + b
#             tf.summary.histogram(layer_name+'/pre_activation', values=preact)  # 记录激活前的取值分布
#         activations = act(preact, name='activation')
#         tf.summary.histogram(layer_name+'/activations', activations)
#         return activations
#
# # 入口函数
# def main(_):
#     # 定义输入
#     with tf.name_scope('input'):
#         inputs = tf.placeholder(tf.float32, shape=(None, 784), name='x-input')
#         labels = tf.placeholder(tf.float32, shape=(None, 10), name='y-input')
#     # 将输入的向量还原成像素矩阵
#     with tf.name_scope('input-reshape'):
#         input_reshape = tf.reshape(inputs, shape=(-1, 28, 28, 1))
#         tf.summary.image('input', input_reshape, max_outputs=10)  # 将像素矩阵的图片信息写入日志
#     # 网络构建
#     net = dense(inputs, 784, 500, 'hidden')
#     outputs = dense(net, 500, 10, 'output', act=tf.identity)
#     # 交叉熵损失
#     with tf.name_scope('cross-entropy'):
#         cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#             labels=labels, logits=outputs))
#         tf.summary.scalar('cross entropy', tensor=cross_entropy)  # 生成交叉熵信息日志
#     with tf.name_scope('train'):
#         train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#     # 正确率：取决于输入信息
#     with tf.name_scope('accuracy'):
#         with tf.name_scope('correct_prediction'):
#             correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
#         with tf.name_scope('accuracy'):
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
#         tf.summary.scalar('accuracy', tensor=accuracy)  # 生成正确率信息日志
#     # 把所有的scalar、histogram、image统一调用
#     merged = tf.summary.merge_all()
#     # 运行
#     mnist = input_data.read_data_sets('data\\mnist', one_hot=True)
#     sess = tf.InteractiveSession()
#     writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())  # 初始化日志写句柄
#     tf.global_variables_initializer().run()
#     for i in range(3000):
#         xs, ys = mnist.train.next_batch(100)
#         summary, _ = sess.run([merged, train_step], feed_dict={inputs: xs, labels: ys})
#         # 将所有日志写入文件
#         writer.add_summary(summary, i)
#     writer.close()
#     sess.close()
#
# if __name__ == '__main__':
#     tf.app.run()


# 高维向量可视化
import numpy as np
import cv2
import tensorflow as tf
import chapter6_net
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector  # 加载生成PROJECTOR日志的帮助函数

# 雪碧图
def create_sprite_image(image):
    image = np.reshape(image * 255, newshape=(-1, 28, 28)).astype(np.uint8)
    n = int(np.ceil(np.sqrt(image.shape[0])))  # 大图像边长
    h, w = image.shape[1], image.shape[2]
    sprite_image = np.zeros((h * n, w * n), dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            cur = row * n + col
            if cur < image.shape[0]:
                sprite_image[row * h:row * h + h, col * w:col * w + w] = image[cur]
            else:
                break
    return sprite_image

# mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
# # 生成雪碧图
# sprite_image = create_sprite_image(mnist.test.images)
# cv2.imwrite('log\\mnist_sprite.jpg', sprite_image)
# # 生成标签
# with open('log\\mnist_meta.tsv', 'w') as f:
#     f.write('Index\tLabel\n')
#     for k, v in enumerate(mnist.test.labels):
#         f.write('%d\t%d\n' % (k, v))

# 模型准备
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)  # 轮数
l2 = tf.contrib.layers.l2_regularizer(0.001)  # l2正则函数
learning_rate = tf.train.exponential_decay(0.1, global_step, 300, 0.99)  # 学习率
# 模型构建
inputs = tf.placeholder(tf.float32, shape=(None, 28 ** 2), name='inputs')
labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
outputs = chapter6_net.LeNet_5(tf.reshape(inputs, shape=(-1, 28, 28, 1)), 10, regularizer=l2)
# 损失函数(交叉熵+l2正则)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels))
# loss += tf.add_n(tf.get_collection(key='losses'))
loss += tf.losses.get_regularization_loss()
# 模型训练
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
    loss, global_step=global_step)
# 滑动平均评测
average_op, average_outputs = chapter6_net.average(
    tf.reshape(inputs, shape=(-1, 28, 28, 1)), num_updates=global_step)
# 运行
mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(5000):
        batch_input, batch_label = mnist.train.next_batch(100)
        _, loss_v = sess.run([optimizer, loss], feed_dict={inputs: batch_input, labels: batch_label})  # 训练
        sess.run(average_op)  # 更新平均模型，与训练同步
        if i % 100 == 0:
            print('Batch %d: loss %.4f' % (i, loss_v))
    evaluate = sess.run(average_outputs, feed_dict={inputs: mnist.test.images})

# 可视化生成日志
y = tf.Variable(evaluate, name='final_logits')
writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())
config = projector.ProjectorConfig()  # 帮助生成日志文件
embedding = config.embeddings.add()  # 增加一个可视化的embedding效果
embedding.tensor_name = y.name  # 指定对应变量名称
embedding.metadata_path = 'mnist_meta.tsv'  # 指定对应的原始数据信息
embedding.sprite.image_path = 'mnist_sprite.jpg'  # 指定对应的图像
embedding.sprite.single_image_dim.extend([28, 28])  # 指定单张图片大小，以便从sprite图像中截取图片
projector.visualize_embeddings(writer, config=config)  # 将projector所需要内容写入日志文件
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()  # 模型保存才能运行projector
    saver.save(sess, save_path='log\\visualization', global_step=10000)
writer.close()

