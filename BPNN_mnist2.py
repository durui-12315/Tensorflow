# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
test_img = mnist.test.images
test_label = mnist.test.labels
train_sample_number = mnist.train.num_examples

# 网络参数
conf = {
    'neure-num': {'i': test_img.shape[1], 'h': 196, 'o': test_label.shape[1]},
    'batch': {'size': int(train_sample_number / 5), 'display': 100, 'lr-change': 300},
    'learning-rate': 0.17,
    'accuracy': 0.978
}

# 占位符
inputs = tf.placeholder(tf.float32, shape=(None, conf['neure-num']['i']), name='neure-i')
labels = tf.placeholder(tf.float32, shape=(None, conf['neure-num']['o']), name='labels')

# 前向传播
with tf.variable_scope('input') as scope:
    weight = tf.Variable(
        tf.truncated_normal((conf['neure-num']['i'], conf['neure-num']['h']), mean=0, stddev=0.1),
        dtype=tf.float32, name='weight-i')
with tf.variable_scope('hidden') as scope:
    neure = tf.matmul(inputs, weight) + tf.Variable(0, dtype=tf.float32, name='bias-i')
    neure = tf.nn.relu(neure, name='neure-h')  # relu激活
    weight = tf.Variable(
        tf.truncated_normal((conf['neure-num']['h'], conf['neure-num']['o']), mean=0, stddev=0.1),
        dtype=tf.float32, name='weight-h')
with tf.variable_scope('output') as scope:
    neure = tf.matmul(neure, weight) + tf.Variable(0, dtype=tf.float32, name='bias-h')
    output = tf.nn.relu(neure, name='neure-o')

# 后向传播
global_step = tf.Variable(0)
learing_rate = tf.train.exponential_decay(conf['learning-rate'], global_step, conf['batch']['lr-change'], 0.96, True)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)  # softmax损失
loss = tf.reduce_mean(loss, name='loss')
train = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss, global_step=global_step)

# 正确率
pred = tf.equal(tf.argmax(labels, axis=1), tf.argmax(output, axis=1))
predict = tf.reduce_mean(tf.cast(pred, tf.float32), name='predict')

# 运行
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    epoch, saver = 0, tf.train.Saver()
    while True:
        batch_input, batch_label = mnist.train.next_batch(conf['batch']['size'])
        sess.run(train, feed_dict={inputs: batch_input, labels: batch_label})  # 训练
        if epoch % conf['batch']['display'] == 0:
            re = sess.run(predict, feed_dict={inputs: test_img, labels: test_label})  # 预测
            print('Batch %d: accuracy %.2f%%, learing-rate %.5f' % (epoch, re * 100, sess.run(learing_rate)))
            if re > conf['accuracy']:
                saver.save(sess=sess, save_path='model/mnist_bpnn2/')
                break
        epoch += 1

    # 模型可视化输出
    writer = tf.summary.FileWriter(logdir='graph/mnist_bpnn', graph=tf.get_default_graph())
    writer.close()