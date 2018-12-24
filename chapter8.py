# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf


"""循环神经网络RNN：
    1.循环神经网络的主要用途是处理和预测序列数据，它刻画的是一个序列当前输出与之前信息的关系。
    2.循环神经网络当前的状态Ht是根据上一时刻的状态Ht-1和当前的输入Xt共同决定的；在时刻t，状态Ht-1浓缩了前面序列
X0、X1、...Xt-1的信息，用于作为输出Ot的参考。
    3.由于序列长度可以无限长，维度有限的H状态不可能将序列的全部信息都保存下来，因此模型必须学习只保留与后面任务
Ot、Ot+1...相关的最重要的信息。
    4.循环神经网络对长度为N的序列展开后，可以视为一个有N个中间层的前馈神经网络，它没有循环链接，因此可以使用反向
传播算法进行训练，而不需要任何特别的优化算法。这种训练方法为“沿时间反向传播”，它是训练循环神经网络最常见的方法。
    5.循环神经网络的状态是通过一个向量来表示的，其维度也称为该网络隐层的大小。
    6.循环神经网络的总损失是所有时刻（或部分时刻）上的损失函数的总和。
    7.理论上，该网络支持任意长度的序列，但实际训练中序列过长会导致梯度消失或爆炸，所以实际中会规定一个最大长度，
超过时会截断旧信息。
    8.长短时记忆网络(LSTM)解决了复杂语言场景中有用信息有大有小、长短不一的情况。
"""

# # RNN前向传播numpy实现
# X = [[i] for i in range(1, 6)]
# state = [.0, .0]
# Y = []
# w_cell = np.asarray([[.1, .2], [.3, .4], [.5, .6]])
# b_cell = np.asarray([.1, -.1])
# w_fc = np.asarray([[1.0], [2.0]])
# b_fc = np.asarray([.1])
# for i in X:
#     inputs = np.concatenate((state, i), axis=0)
#     state = np.tanh(np.dot(inputs, w_cell) + b_cell)  # 作为下一个时间序列的状态值
#     print(state)
#     output = np.dot(state, w_fc) + b_fc  # 全连接层
#     Y.append(output)  # 把输出值存入输出序列
# print(Y)


# # RNN前向传播TF实现
# num_X, num_state, num_output = 20, 10, 5
# X = tf.placeholder(dtype=tf.float32, shape=(1, num_X))
# state = tf.get_variable('state', shape=(1, num_state), initializer=tf.zeros_initializer())
# w_cell = tf.get_variable('w_cell', shape=(num_X + num_state, num_state),
#                          initializer=tf.truncated_normal_initializer())
# b_cell = tf.get_variable('b_cell', shape=(num_state,), initializer=tf.truncated_normal_initializer())
# inputs = tf.concat([state, X], axis=1)
# state = tf.nn.tanh(tf.matmul(inputs, w_cell) + b_cell)
# w_fc = tf.get_variable('w_fc', shape=(num_state, num_output), initializer=tf.truncated_normal_initializer())
# b_fc = tf.get_variable('b_fc', shape=(num_output,), initializer=tf.truncated_normal_initializer())
# output = tf.matmul(state, w_fc) + b_fc
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for record in np.random.randint(1, 100, size=(6, 1, num_X)):
#         o, s = sess.run([output, state], feed_dict={X: record})
#         print('Output: {}\nState: {}'.format(o, s), sep='\n')


# # 长短时记(LSTM)细胞前向传播TF实现
# def cell(pre_state, cur_x, scope, act_fn=tf.nn.sigmoid):
#     with tf.variable_scope(scope):
#         w = tf.get_variable('w', shape=(2 * n, n), initializer=tf.truncated_normal_initializer())
#         b = tf.get_variable('b', shape=(n,), initializer=tf.truncated_normal_initializer())
#         tmp = tf.concat([pre_state, cur_x], axis=1)
#         return act_fn(tf.matmul(tmp, w) + b)
#
# # 初始值
# n = 4
# h = tf.get_variable('state', shape=(1, n), initializer=tf.random_normal_initializer())
# c = tf.get_variable('current', shape=(1, n), initializer=tf.random_normal_initializer())
# x = tf.placeholder(dtype=tf.float32, shape=(1, n))
# # 输入状态值
# input_val = cell(h, x, scope='input_value', act_fn=tf.nn.tanh)
# # 输入门
# input_door = cell(h, x, scope='input_door')
# # 遗忘门
# forget_door = cell(h, x, scope='forget_door')
# # 更新当前状态
# c = input_val * input_door + c * forget_door
# # 输出门
# output_door = cell(h, x, scope='output_door')
# # 更新隐藏状态
# h = c * tf.nn.tanh(output_door)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for i, record in enumerate(np.random.randint(1, 100, size=(6, 1, n))):
#         print('时序%d：' % i, '-' * 50)
#         print('当前状态：', sess.run(c, feed_dict={x: record}))
#         print('隐藏状态：', sess.run(h, feed_dict={x: record}))


# # TF的LSTM的API实现样例
# batch_size, input_length, h_length, output_length = 100, 8, 10, 5
#
# # RNNcell之外的全连接层，得到最后的输出
# def fully_connected(h):
#     w = tf.get_variable('output_w', shape=(h_length, output_length), initializer=tf.random_normal_initializer())
#     b = tf.get_variable('output_b', shape=(output_length,), initializer=tf.random_normal_initializer())
#     return tf.matmul(h, w) + b
#
# # 均方损失
# def calc_loss(output, label):
#     return tf.reduce_mean(tf.square(label - output))
#
# x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_length))
# label = tf.placeholder(dtype=tf.float32, shape=(batch_size, output_length))
# # 定义隐藏状态长度为10的LSTM结构，所需变量会自动声明
# lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=h_length)
# # 定义当前状态c和隐藏状态h为初始为0，并定义训练样本的batch，state为包含h和c的元祖
# state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
# loss = 0.0  # 定义损失值
# for i in range(10):  # 训练RNN时需要将其展开成前馈神经网络，故定义一个有限的序列长度10
#     if i > 0:
#         tf.get_variable_scope().reuse_variables()  # 第一个时刻要声明LSTM中的变量，第二时刻后需要复用
#     # 将一个batch的样本和前一时刻的h和c状态传入LSTM，得到新的h和c状态，lstm_output为h
#     lstm_output, state = lstm(inputs=x, state=state)
#     # 将当前时刻的h传入全连接层
#     final_output = fully_connected(lstm_output)
#     # 计算当前时刻的损失值并做序列上的累加
#     loss += calc_loss(final_output, label)
# # 优化器
# optimzer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
# # 训练
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()
#     # 只训练一个batch，并输出均方损失
#     cost, opt = sess.run([loss, optimzer], feed_dict={
#         x: np.random.randint(1, 100, size=(batch_size, input_length)),
#         label: np.random.randint(30, 50, size=(batch_size, output_length))
#     })
#     print(cost)


"""双向循环神经网络(bidirectional RNN):
1.双向循环神经网络的主体结构就是两个方向相反的单向循环神经网络的结合；在每时刻t，输入同时给两个方向相反网络。
2.两个网络独立计算，各自产生某时刻的新状态值和输出，而双向循环的最终输出为两个单向循环的简单拼接；
3.两个循环神经网络除方向不同外，其余结构完全对称。
"""

"""深层循环神经网络(Deep RNN):
1.多个循环层是为了增强模型的表达能力和提取抽象信息的能力；
2.每一时刻不同层的参数可以不同，类似多层卷积。
3.使用dropout可以是模型更健壮；卷积网络一般在全连接中使用dropout，而深层循环网络会在同一时刻的层与层之间使用。
"""

# 深层神经网络 Deep RNN
batch_size, input_length, output_length, h_length = 100,10, 5, 1000

# RNN cell之外的全连接层，得到最后的输出
def fully_connected(h):
    w = tf.get_variable('output_w', shape=(h_length, output_length), initializer=tf.random_normal_initializer())
    b = tf.get_variable('output_b', shape=(output_length,), initializer=tf.random_normal_initializer())
    return tf.matmul(h, w) + b

# 均方损失
def calc_loss(output, label):
    return tf.reduce_mean(tf.square(label - output))

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_length))
label = tf.placeholder(dtype=tf.float32, shape=(batch_size, output_length))
# 定义LSTM为基础结构
lstm = tf.nn.rnn_cell.BasicLSTMCell
rnn = tf.nn.rnn_cell.BasicRNNCell
# 定义每层都基于LSTM的深层网络结构，再给每层加上dropout
deep_layers = 3  # 每一刻的层数都为3
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[tf.nn.rnn_cell.DropoutWrapper(lstm(num_units=h_length),
                    input_keep_prob=1.0, output_keep_prob=0.8) for _ in range(deep_layers)])
# 初始化状态值为0
state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
loss = 0.0
# 展开成为前馈神经网络以便训练
for i in range(10):
    if i > 0: tf.get_variable_scope().reuse_variables()
    stacked_lstm_output, state = stacked_lstm(x, state=state)  # 更新状态h和c
    # 将当前时刻的最后一层的h传入全连接层
    final_output = fully_connected(stacked_lstm_output)
    # 计算当前时刻的损失值并做序列上的累加
    loss += calc_loss(final_output, label)
# 优化器
optimzer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
# 训练
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # 只训练一个batch，并输出均方损失
    cost, opt = sess.run([loss, optimzer], feed_dict={
        x: np.random.randint(1, 100, size=(batch_size, input_length)),
        label: np.random.randint(30, 50, size=(batch_size, output_length))
    })
    print(cost)



