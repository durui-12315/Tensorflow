# -*- coding:UTF-8 -*-
# !/usr/bin/python

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""TFRecord数据结构：
1.tf.train.Example包含了一个属性名称到取值的字典map<string, Feature>；
2.属性名称为一个字符串；
3.属性的取值为字符串(BytesList)、实数列表(FloatList)、整数列表(Int64List)
"""

# mnist训练数据存入TFRecord中
mnist = input_data.read_data_sets(train_dir='data\\mnist', one_hot=False)
idx = np.random.randint(0, 10000, size=(10,))
images = mnist.test.images[idx]  # 图片矩阵作为一个字符串属性保存在TFRecord中
labels = mnist.test.labels[idx]  # 标签作为一个整数列表属性保存在TFRecord中
pixels = images.shape[1]  # 图片分辨率作为一个整数列表属性也保存在TFRecord中
writer = tf.python_io.TFRecordWriter(path='TFRecord\\mnist.predict.tfrecords')  # 产生TFRecord文件写的句柄
for i in range(idx.size):  # mnist.test.num_examples
    # 将图像矩阵转化成字符串
    image_raw = images[i].tostring()
    # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': tf.train.Feature(int64_list=tf.train.Int64List(value=[pixels])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    }))
    # 将一个Example写入TFRecord文件中
    writer.write(record=example.SerializeToString())
writer.close()


# # 读取TFRecord文件
# reader = tf.TFRecordReader()  # 创建读取句柄
# filename_queue = tf.train.string_input_producer(['TFRecord\\mnist.tfrecords'])  # 创建文件输入队列
# _, serialized_example = reader.read(queue=filename_queue)  # 从文件中读出一个样例，read_up_to()可一次读多个
# features = tf.parse_single_example(  # 解析读入的一个样例，一次读多个为parse_example()
#     serialized=serialized_example,  # 要解析的样例
#     features={  # 解析后的格式
#         # tf.FixedLenFeature的解析结果为Tensor，tf.VarLenFeature的解析结果为SparseTensor，用于处理稀疏数据
#         'image_raw': tf.FixedLenFeature([], dtype=tf.string),  # 解析格式需和写入时选择的格式一致
#         'pixels': tf.FixedLenFeature([], dtype=tf.int64),
#         'label': tf.FixedLenFeature([], dtype=tf.int64),
#     })
# image = tf.decode_raw(features['image_raw'], tf.uint8)  # 将字符串解析成图像对应的像素数组
# label = tf.cast(features['label'], tf.int32)
# pixels = tf.cast(features['pixels'], tf.int32)
# with tf.Session() as sess:
#     # 启动多线程处理数据
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # 循环读取TFRecord样例
#     for i in range(10):
#         print(sess.run([pixels, label, image]))


import chapter7_image3

"""数据集Dataset
1.读取数据的三个基本步骤：
    （1）定义数据集的构造方法；
    （2）定义遍历器来遍历数据集；
    （3）使用get_next()方法从遍历器中读取数据张量，作为计算图其它部分输入。
"""

# # 从一个张量创建数据集，并遍历
# input_data = [1, 2, 3, 4, 5]
# ds = tf.data.Dataset.from_tensor_slices(input_data)  # 构造函数
# iterator = ds.make_one_shot_iterator()  # 遍历器
# x = iterator.get_next()  # 读取数据
# y = tf.square(x)
# with tf.Session() as sess:
#     for _ in range(len(input_data)):
#         print(sess.run(y))


# # 从文本中创建数据集(自然语言处理)
# input_files = ['data\\text\\1.txt', 'data\\text\\2.txt', 'data\\text\\3.txt']
# ds = tf.data.TextLineDataset(filenames=input_files)  # 构造函数
# iterator = ds.make_one_shot_iterator()  # 迭代器
# txt = iterator.get_next()  # 读取数据
# with tf.Session() as sess:
#     for _ in range(10):
#         print(sess.run(txt))


# # 从TFRecord中创建数据集(图像处理)
# def parse(record):  # 解析样例
#     features = tf.parse_single_example(serialized=record, features={  # 解析内容
#         'example': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.int64),
#         'height': tf.FixedLenFeature([], tf.int64),
#         'width': tf.FixedLenFeature([], tf.int64),
#         'channels': tf.FixedLenFeature([], tf.int64),
#     })
#     shape = tf.stack([features['height'], features['width'], features['channels']])  # 通过拼接得到形状
#     example = tf.reshape(tf.decode_raw(features['example'], out_type=tf.uint8), shape=shape)  # 解码变形状
#     label = tf.cast(features['label'], dtype=tf.int32)  # 变类型
#     return example, label
#
# # 方法一
# input_files = ['TFRecord\\flowers-train-%d.tfrecords' % i for i in range(1, 5)]
# ds = tf.data.TFRecordDataset(filenames=input_files)  # 构造函数
# ds = ds.map(map_func=parse)  # map函数表示对数据集中的每一条数据进行调用parse方法
# iterator = ds.make_one_shot_iterator()  # 迭代器
# example, label = iterator.get_next()  # 读取一个个经parse后的数据
# example = chapter7_image3.image_preprocess(example, 300, 300)  # 图像预处理
# with tf.Session() as sess:
#     for i in range(5):
#         print(sess.run([example, label]))

# # 方法二，使用占位符
# input_files = tf.placeholder(dtype=tf.string)  # 文本占位符
# ds = tf.data.TFRecordDataset(filenames=input_files)
# ds = ds.map(map_func=parse)
# iterator = ds.make_initializable_iterator()  # 使用占位符，就得使用make_initializable_iterator
# example, label = iterator.get_next()  # 遍历一个个的数据
# example = chapter7_image3.image_preprocess(example, 300, 300)  # 图像预处理
# with tf.Session() as sess:
#     sess.run(iterator.initializer, feed_dict={
#         input_files: ['TFRecord\\flowers-train-%d.tfrecords' % i for i in range(1, 5)]
#     })
#     while True:
#         try:
#             print(sess.run([example, label]))
#         except tf.errors.OutOfRangeError:  # 通过抛出异常来结束遍历
#             break
