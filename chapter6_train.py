# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import chapter6_net


def train(mnist):
    # 参数
    batch_size, batch_display = 100, 1000
    # 数据来源
    validation_img, validation_label, validation_len = \
        mnist.validation.images, mnist.validation.labels, mnist.validation.num_examples
    test_img, test_label, test_len = mnist.test.images, mnist.test.labels, mnist.test.num_examples
    # 模型准备
    global_step = tf.Variable(0, trainable=False, dtype=tf.int32)  # 轮数
    l2 = tf.contrib.layers.l2_regularizer(.001)  # l2正则函数
    learning_rate = tf.train.exponential_decay(.0015, global_step, batch_display*3, .96)  # 学习率
    # 模型构建
    inputs = tf.placeholder(tf.float32, shape=(None, 28 ** 2))
    labels = tf.placeholder(tf.float32, shape=(None, 10))
    outputs = chapter6_net.LeNet_5(tf.reshape(inputs, shape=(-1, 28, 28, 1)), 10, regularizer=l2)
    # 损失函数(交叉熵+l2正则)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=tf.argmax(labels, 1))
    loss += tf.add_n(tf.get_collection(key='losses'))
    # 模型训练
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    # 滑动平均模型
    average_op, average_outputs = chapter6_net.average(
        tf.reshape(inputs, shape=(-1, 28, 28, 1)), num_updates=global_step)
    # 正确率
    acc = chapter6_net.accuracy(average_outputs, labels)
    # 运行
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    while True:
        batch_input, batch_label = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={inputs: batch_input, labels: batch_label})  # 训练
        sess.run(average_op)  # 更新平均模型，与训练同步
        batch = sess.run(global_step)
        if batch % batch_display == 0:
            validation = sess.run(acc, feed_dict={inputs: validation_img, labels: validation_label})
            test = sess.run(acc, feed_dict={inputs: test_img, labels: test_label})
            print('Batch %d: learning-rate %.5f; validation %.2f%%; test %.2f%%' %
                  (batch, sess.run(learning_rate), validation * 100, test * 100))
            if validation > .992 and test > .991:
                saver.save(sess, 'model/mnist_bpnn4/chapter6.ckpt')
                break
    sess.close()


def main(argv=None):
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

