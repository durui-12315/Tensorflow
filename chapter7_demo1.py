# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
import os
import numpy as np
import chapter7_image3
import matplotlib.pyplot as plt

"""通过队列与多线程实现数据输入的完整实例"""

# # 按文件夹顺序制作TFRecord
# dir_list, num, i = None, None, -1
# sess = tf.InteractiveSession()
# for path, dirs, files in os.walk('data\\flower_photos'):
#     if i == -1:
#         dir_list = dirs
#         num = len(dirs)
#         i += 1
#         continue
#     name = dir_list[i]
#     i += 1
#     writer = tf.python_io.TFRecordWriter(path='TFRecord\\flower.tfrecords.%s-%.2d_%.2d' % (name, i, num))
#     for f in files:
#         file = tf.read_file(os.path.join(path, f))
#         img = tf.image.decode_jpeg(file).eval()
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'example': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
#             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
#             'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
#             'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
#             'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[2]]))
#         }))
#         writer.write(record=example.SerializeToString())
#     writer.close()
# sess.close()


# 按随机顺序制作TFRecord
flower_path, i = [], 0
for path, dirs, files in os.walk('data\\flower_photos'):
    if len(dirs): continue
    for f in files: flower_path.append((os.path.join(path, f), i))
    i += 1
flower_path = np.array(flower_path, dtype=object)
np.random.shuffle(flower_path)
print(flower_path[:20])
batch_start, batch_size, i = 0, 800, 1
sess = tf.InteractiveSession()
while batch_start < flower_path.size / 2:
    writer = tf.python_io.TFRecordWriter(path='TFRecord\\flowers-train-%d.tfrecords' % i)
    for path, label in flower_path[batch_start:batch_start + batch_size]:
        file = tf.read_file(path)
        img = tf.image.decode_jpeg(file).eval()
        w, h, c = img.shape
        example = tf.train.Example(features=tf.train.Features(feature={
            'example': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[c]))
        }))
        writer.write(record=example.SerializeToString())
    writer.close()
    batch_start += batch_size
    i += 1
sess.close()


# # 队列读取并预处理
# # 1.获取文件列表 TFRecord\\flower.tfrecords.*
# files = tf.train.match_filenames_once(pattern='TFRecord\\flowers-*.tfrecords')
# # 2.将文件随机输入队列
# file_queue = tf.train.string_input_producer(files)
# # 3.读取并解析队列中文件内容
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(queue=file_queue)  # 读取内容
# features = tf.parse_single_example(serialized=serialized_example, features={  # 解析内容
#     'example': tf.FixedLenFeature([], tf.string),
#     'label': tf.FixedLenFeature([], tf.int64),
#     'height': tf.FixedLenFeature([], tf.int64),
#     'width': tf.FixedLenFeature([], tf.int64),
#     'channels': tf.FixedLenFeature([], tf.int64),
# })
# shape = tf.stack([features['height'], features['width'], features['channels']])  # 通过拼接得到形状
# example = tf.reshape(tf.decode_raw(features['example'], out_type=tf.uint8), shape=shape)  # 解码并变形
# label = tf.cast(features['label'], dtype=tf.int32)  # 变类型
# # 4.数据预处理
# h, w, c = 300, 300, 3
# example = chapter7_image3.image_preprocess(example, h, w)
# batch_size, min_after_dequeue = 10, 1000
# capacity = min_after_dequeue + batch_size * 3
# # 5.样例组合队列
# example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size,  # 形成batch
#         capacity=capacity, min_after_dequeue=min_after_dequeue, shapes=[[w, h, c], []])  # ((w, h, c), ())
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()
#     # 6.开启多线程
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     print(sess.run([label, example]))
#     # 7.batch数据展示
#     for i in range(5):
#         cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
#         print(cur_example_batch[0], cur_label_batch)
#         plt.imshow(cur_example_batch[0])
#         plt.show()
#     # 8.关闭队列与线程
#     coord.request_stop()
#     coord.join(threads)


