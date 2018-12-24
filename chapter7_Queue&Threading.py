# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
import numpy as np
import threading
import time


"""多线程输入数据：
1.图像预处理会减慢整个模型的训练过程，为避免图像预处理成为训练效率瓶颈，则需要多线程处理；
2.多线程输入数据处理步骤：
    （1）指定原始数据的文件列表；
    （2）创建文件列表队列；
    （3）从队列中读取数据；
    （4）数据预处理；
    （5）整理成batch作为神经网络的输入。
3.队列是TF多线程输入数据处理框架的基础；
4.
"""


# # 先进先出队列
# que = tf.FIFOQueue(capacity=2, dtypes=tf.float32)
# que_init = que.enqueue_many(vals=([1, 10],))  # 多元素入列
# x = que.dequeue()  # 出列
# que_inc = que.enqueue(x + 1)  # 入列
# with tf.Session() as sess:
#     sess.run(que_init)
#     for _ in range(10):
#         v, _ = sess.run([x, que_inc])  # 必须同时运行，因为有关联
#         print(v)


# # 随机队列（队列内部顺序是乱的），min_after_dequeue限制了出队时队列中元素的最少个数
# que = tf.RandomShuffleQueue(capacity=3, min_after_dequeue=1, dtypes=tf.float32)
# que_init = que.enqueue_many(vals=([1, 10, 100],))  # 多元素入列
# x = que.dequeue()  # 出列
# que_inc = que.enqueue(x + 1)  # 入列
# with tf.Session() as sess:
#     sess.run(que_init)
#     for _ in range(10):
#         v, _ = sess.run([x, que_inc])  # 必须同时运行，因为有关联
#         print(v)


# # Coordinator类协同多线程一起停止
# def loops(coord, worker_id):
#     while not coord.should_stop():  # 判断是否需要停止
#         if np.random.rand() < 0.1:  # 随机停止
#             print('Stoping from id:', worker_id)
#             coord.request_stop()  # 调用该函数使得其它线程判断should_stop函数时返回True，以停止当前线程
#         else:
#             print('Wording on id:', worker_id)
#         time.sleep(1)  # 程序暂停1秒
#
# coord = tf.train.Coordinator()  # 声明此类来协同多线程
# threads = [threading.Thread(target=loops, args=(coord, i)) for i in range(1, 6)]  # 声明5个线程
# for t in threads: t.start()  # 启动所有线程
# coord.join(threads)  # 等待所有线程退出


# # QueueRunner类使多线程操作同一个队列
# que = tf.FIFOQueue(capacity=100, dtypes=tf.float32)
# enque = que.enqueue(vals=[tf.random_normal((1,))])
# qr = tf.train.QueueRunner(queue=que, enqueue_ops=[enque] * 10)  # 创建5个线程的入队操作
# tf.train.add_queue_runner(qr=qr)  # 把qr加入TF计算图上指定的结合（这里为默认集合）
# deque = que.dequeue()
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     # Coordinator类协同启动默认集合中所有线程
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for _ in range(3):
#         print(sess.run(deque))
#     # 停止所有线程，并退出
#     coord.request_stop()
#     coord.join(threads)


# # 创建多TFRecord文件
# num_shards = 5
# for i in range(1, num_shards + 1):  # 文件数
#     filename = 'TFRecord\\test.tfrecords-%.2d-of-%.2d' % (i, num_shards)
#     writer = tf.python_io.TFRecordWriter(path=filename)  # 每个文件依次创建写句柄
#     for j in range(10):  # 每文件里记录数
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'file': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
#             'record': tf.train.Feature(int64_list=tf.train.Int64List(value=[j]))
#         }))
#         writer.write(record=example.SerializeToString())  # 样例数据序列化
#     writer.close()


# # 多线程文件读取
# files = tf.train.match_filenames_once(pattern='TFRecord\\test.tfrecords-*')  # 通过正则获取文件列表
# file_queue = tf.train.string_input_producer(files, shuffle=False, num_epochs=None)  # 创建输入队列，循环输入
# # 读取队列并解析一个样本
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(file_queue)
# features = tf.parse_single_example(serialized_example, features={
#     'file': tf.FixedLenFeature([], tf.int64),
#     'record': tf.FixedLenFeature([], tf.int64)
# })
# file, record = tf.cast(features['file'], tf.int32), tf.cast(features['record'], tf.int32)
# with tf.Session() as sess:
#     tf.local_variables_initializer().run()  # 初始化match_filenames_once局部变量
#     # Coordinator类协同并启动多线程
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # 多次执行获取数据
#     for i in range(10):
#         print(sess.run([file, record]))
#     # 停止所有线程，并退出
#     coord.request_stop()
#     coord.join(threads)


# 组合训练数据(batching)
files = tf.train.match_filenames_once(pattern='TFRecord\\test.tfrecords-*')
file_queue = tf.train.string_input_producer(files, shuffle=False)  # 创建输入队列，按顺序循环输入
# 读取队列并解析一个样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
features = tf.parse_single_example(serialized_example, features={
    'file': tf.FixedLenFeature([], tf.int64),
    'record': tf.FixedLenFeature([], tf.int64)
})
# 数据预处理
example, label = tf.cast(features['file'], tf.int32), tf.cast(features['record'], tf.int32)
batch_size = 10  # 一个batch中样例个数
capacity = 1000 + batch_size * 3  # 组合样例队列最多可以存储的个数（太大占资源，太小容易变空而阻碍出列）
# 组合为一个batch的样例和标签（还有shuffle_batch）
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
with tf.Session() as sess:
    # 初始化所有全局变量和局部变量
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取当前batch的样例和标签，作为神经网络的输入
    for i in range(10):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)


