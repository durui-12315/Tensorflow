# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
import chapter5_mnist_net


# 训练函数
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
    learning_rate = tf.train.exponential_decay(.1, global_step, batch_display * 3, .96)  # 学习率

    # 模型构建
    inputs, labels, outputs = chapter5_mnist_net.net(regularizer=l2)

    # 滑动平均模型构建
    average_op, average_outputs = chapter5_mnist_net.average_net(inputs, num_updates=global_step)

    # 损失函数(交叉熵+l2正则)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=tf.argmax(labels, 1))
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection(key='losses'))

    # 模型训练
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss, global_step=global_step)

    # 正确率(滑动平均模型测试)
    acc = chapter5_mnist_net.accuracy(average_outputs, labels)  # 验证集

    # 运行
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()  # 保存
    tf.global_variables_initializer().run()
    while True:
        batch_input, batch_label = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={inputs: batch_input, labels: batch_label})  # 训练
        sess.run(average_op)  # 平均模型更新
        batch = sess.run(global_step)
        if batch % batch_display == 0:  # 测试
            validation = sess.run(acc, feed_dict={inputs: validation_img, labels: validation_label})
            test = sess.run(acc, feed_dict={inputs: test_img, labels: test_label})
            print('Batch %d: learning-rate %.5f; validation %.2f%%; test %.2f%%' %
                  (batch, sess.run(learning_rate), validation * 100, test * 100))
            if validation > .98 and test > .978:
                saver.save(sess, 'model/mnist_bpnn3/chapter5.ckpt')
                break
    sess.close()


# 主函数(argv=None不可少)
def main(argv=None):
    mnist = chapter5_mnist_net.data_set()  # 加载数据
    train(mnist)


if __name__ == '__main__':
    tf.app.run()  # 运行主函数
