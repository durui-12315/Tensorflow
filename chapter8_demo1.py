# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""利用Deep LSTM实现对正弦函数sin按时序取值的离散预测
   离散化概念：在给定一个区间内，通过有限个采样点去模拟一个连续函数曲线"""

# 参数
HIDDEN_SIZE = 30  # LSTM隐藏状态序列数
NUM_LAYERS = 2  # LSTM单个时刻的层数
TIMESTEPS = 10  # 时间序列长度
TRAIN_BATCH_SIZE = 120  # 训练集batch大小
TEST_BATCH_SIZE = 200  # 测试集batch大小
TRAIN_EXAMPLES = 10000  # 训练样本数
TEST_EXAMPLES = 1000  # 测试样本数
SAMPLE_GAP = 0.01  # 样本间隔数
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

# 数据生成
def generate_data(sep, length):
    x, y = [], []
    for i in range(length):  # 用i到i+TIMESTEPS-1的时间序列来预测下一时刻i+TIMESTEPS的值
        x.append([sep[i: i + TIMESTEPS]])
        y.append([sep[i + TIMESTEPS]])
    return np.array(x, np.float32), np.array(y, np.float32)

test_start = (TRAIN_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TEST_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
# 训练
train_X, trian_Y = generate_data(np.sin(np.linspace(
    0, test_start, TRAIN_EXAMPLES + TIMESTEPS, dtype=np.float32)), TRAIN_EXAMPLES)
# 测试
test_sin = np.sin(np.linspace(test_start, test_end, TEST_EXAMPLES + TIMESTEPS, dtype=np.float32))
test_X, test_Y = generate_data(test_sin, TEST_EXAMPLES)


# 数据集制作
# 训练
ds = tf.data.Dataset.from_tensor_slices((train_X, trian_Y)).repeat()
ds = ds.shuffle(buffer_size=1000).batch(batch_size=TRAIN_BATCH_SIZE)
train_inputs, train_labels = ds.make_one_shot_iterator().get_next()
# 测试
ds = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).repeat()
ds = ds.batch(TEST_BATCH_SIZE)
test_inputs, test_labels = ds.make_one_shot_iterator().get_next()


# 模型制作
def deep_lstm(inputs):
    # 定义一个时刻的深层循环网络
    dl = tf.nn.rnn_cell.MultiRNNCell(cells=
            [tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    # 按多个序列展开网络，成为前馈网络，并计算每个时刻前向传播输出结果，其维度[batch_size, time, h_size]
    multi_output, _ = tf.nn.dynamic_rnn(cell=dl, inputs=inputs, dtype=tf.float32)
    # 得到最后一时刻的lstm输出值
    last_output = multi_output[:, -1, :]
    # 对输出值再做一层全连接，输出值个数为1
    return tf.contrib.layers.fully_connected(last_output, 1, activation_fn=None)

# 训练
with tf.variable_scope('RNN'):
    train_outputs = deep_lstm(train_inputs)
    # 损失函数
    loss = tf.losses.mean_squared_error(labels=train_labels, predictions=train_outputs)
    # 优化器(global_step也可以用tf.train.get_global_step()代替)
    optimizer = tf.contrib.layers.optimize_loss(loss, global_step, optimizer='Adagrad',
                                                learning_rate=0.1)
# 测试
with tf.variable_scope('RNN', reuse=True):
    test_outputs = deep_lstm(test_inputs)
    test_outputs = np.array(test_outputs).squeeze()  # 去除单维条目
    test_labels = np.array(test_labels).squeeze()
    # 使用均方根误差rmse
    rmse = tf.sqrt(tf.reduce_mean(tf.square(test_outputs - test_labels)))
    # 注：回归问题一般用误差值来直接测试，分类问题一般用正确率来测试

# 画图
def draw():
    ds = tf.data.Dataset.from_tensor_slices(test_X).batch(TEST_EXAMPLES)
    inputs = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope('RNN', reuse=True):
        return deep_lstm(inputs)


# 运行
outputs = draw()  # 必须先执行函数生成图
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
while True:
    sess.run(optimizer)
    i = sess.run(global_step)
    if i % 100 == 0:
        acc = sess.run(rmse)
        print('Batch %d: loss %.5f' % (i, acc))
        if acc <= 0.01:  # 均方根误差达到一定阈值时显示图像并停止
            outputs = np.array(sess.run(outputs)).squeeze()
            plt.figure()
            plt.plot(outputs, label='prediction')
            plt.plot(test_sin[:-10], label='real_sin')
            plt.legend()
            plt.show()
            break
sess.close()

