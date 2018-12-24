# -*- coding:UTF-8 -*-
# !/usr/bin/python

import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""利用Deep LSTM实现房价预测"""
# 数据准备
data = []
with open('data\\boston_housing.data') as f:
    for r in f.readlines():
        data.append(re.split(' +', r))
    data = np.asfarray(data, dtype=np.float32)
X, Y = np.split(data, indices_or_sections=(13,), axis=1)
X = np.reshape(X, newshape=(-1, 1, X.shape[1]))
train_X, train_Y = X[:400], Y[:400]
test_X, test_Y = X[400:], Y[400:]
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

# 训练集队列和batch
ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).repeat()
ds = ds.shuffle(buffer_size=500).batch(batch_size=50)
train_inputs, train_labels = ds.make_one_shot_iterator().get_next()

# 模型构建
def deep_lstm(x):
    cell = tf.nn.rnn_cell.BasicLSTMCell
    dropout = tf.nn.rnn_cell.DropoutWrapper
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([cell(num_units=30) for _ in range(3)])  # 某时刻层数
    multi_output, _ = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x, dtype=np.float32)  # 展开为前馈网络
    last_output = multi_output[:, -1, :]  # 只获取最后一个时序的输出
    return tf.contrib.layers.fully_connected(last_output, 1, activation_fn=None)  # 最后全连接

# 训练
with tf.variable_scope('RNN'):
    train_outputs = deep_lstm(train_inputs)
    loss = tf.losses.mean_squared_error(labels=train_labels, predictions=train_outputs)  # 均方损失
    # optimizer = tf.contrib.layers.optimize_loss(loss, global_step, optimizer='Adam',
    #                                             learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0015).minimize(loss, global_step=global_step)

# 测试
ds = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).repeat()
ds = ds.batch(100)
test_inputs, test_labels = ds.make_one_shot_iterator().get_next()
with tf.variable_scope('RNN', reuse=True):
    test_outputs = deep_lstm(test_inputs)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(test_outputs - test_labels)))  # 均方根

# 测试数据全部预测
ds = tf.data.Dataset.from_tensor_slices(test_X).repeat(count=1).batch(test_X.size)
all_test_inputs = ds.make_one_shot_iterator().get_next()
with tf.variable_scope('RNN', reuse=True):
    all_test_outputs = deep_lstm(all_test_inputs)

# 运行
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    while True:
        sess.run(optimizer)
        step = sess.run(global_step)
        if step % 50 == 0:
            re = sess.run(rmse)
            print('Batch %d: rmse %.4f' % (step, re))
            if re < 3.5:
                all_test_outputs = np.array(sess.run(all_test_outputs)).squeeze()  # 去除单维条目
                test_Y = np.array(test_Y).squeeze()
                plt.figure()
                plt.plot(all_test_outputs, label='prediction')
                plt.plot(test_Y, label='real_price')
                plt.legend()
                plt.show()
                break



