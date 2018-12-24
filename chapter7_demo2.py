# -*- coding:UTF-8 -*-
# !/usr/bin/python

import tensorflow as tf
from common import fc, conv2d, image_preprocess

"""DataSet数据集实现数据batch输入完整实例"""

# 参数
C = {'image-size': 300, 'batch-size': 10, 'shuffle-buffer': 50, 'n-class': 5, 'learning-rate': 0.01,
     'l2': tf.contrib.layers.l2_regularizer(0.001), 'num-epochs': 10000, 'collection-name': 'losses',
     'batch-display': 100, 'test-batch-size': 50}
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = tf.train.exponential_decay(C['learning-rate'], global_step, C['batch-display'] * 3, .99)

# VGGNet神经网络
def VGGNet(inputs, is_train=True):
    net = conv2d('layer1', inputs, depth=32, ksize=(3, 3, 3), reuse=not is_train)
    net = conv2d('layer2', net, depth=64, ksize=(3, 3, 32), reuse=not is_train)
    net = conv2d('layer3', net, depth=64, ksize=(3, 3, 64), reuse=not is_train)
    net = fc('fc1', net, output_num=1000, flatten=True, keep_prob=0.8, reuse=not is_train,
             weights_regularizer=C['l2'], collection_name=C['collection-name'])
    net = fc('fc2', net, output_num=500, keep_prob=0.8, reuse=not is_train,
             weights_regularizer=C['l2'], collection_name=C['collection-name'])
    return fc('fc3', net, output_num=C['n-class'], reuse=not is_train,
              weights_regularizer=C['l2'], collection_name=C['collection-name'])

# TFRecord单样例解析
def parse(record):
    features = tf.parse_single_example(serialized=record, features={  # 解析内容
        'example': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    })
    shape = tf.stack([features['height'], features['width'], features['channels']])  # 通过拼接得到形状
    example = tf.reshape(tf.decode_raw(features['example'], out_type=tf.uint8), shape=shape)  # 解码变形状
    # label = tf.cast(features['label'], dtype=tf.int32)  # 变类型
    label = features['label']
    return example, label


# 训练和测试文件列表
train_files = tf.train.match_filenames_once('TFRecord\\flowers-train-*.tfrecords')
test_files = tf.train.match_filenames_once('TFRecord\\flowers-test.tfrecords')

# 定义训练数据集、预处理、随机和batch操作
train_ds = tf.data.TFRecordDataset(filenames=train_files).repeat()  # 定义并无限重复
train_ds = train_ds.map(map_func=parse).map(lambda example, label:  # 解析与预处理
                        (image_preprocess(example, C['image-size'], C['image-size']), label))
train_ds = train_ds.shuffle(buffer_size=C['shuffle-buffer']).batch(batch_size=C['batch-size'])  # 随机、batch

# 训练数据集迭代器(使用match_filenames_once和使用占位符类似，所以使用make_initializable_iterator)
train_iter = train_ds.make_initializable_iterator()
train_example_batch, train_label_batch = train_iter.get_next()

# 定义损失函数，优化器
train_outputs_batch = VGGNet(train_example_batch)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label_batch, logits=train_outputs_batch)
loss += tf.add_n(tf.get_collection(key=C['collection-name']))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 定义测试集、缩放、batch操作
test_ds = tf.data.TFRecordDataset(filenames=test_files).repeat()
test_ds = test_ds.map(map_func=parse).map(lambda example, label:  # 解析与缩放
                (image_preprocess(example, C['image-size'], C['image-size'], is_train=False), label))
test_ds = test_ds.batch(batch_size=C['test-batch-size'])

# 定义测试集迭代器
test_iter = test_ds.make_initializable_iterator()
test_example_batch, test_label_batch = test_iter.get_next()

# 正确率
test_outputs_batch = VGGNet(test_example_batch, is_train=False)
acc = tf.equal(tf.argmax(test_outputs_batch, axis=1), test_label_batch)
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# 运行
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # 全局和局部初始化
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # 初始化迭代器
    sess.run([train_iter.initializer, test_iter.initializer])
    while True:
        try:
            sess.run(optimizer)  # 训练
            # sess.run(average_op)
            num = sess.run(global_step)
            if num % C['batch-display'] == 0:
                accuracy, lr = sess.run([acc, learning_rate])
                print('Batch %d: learning-rate %.5f, accuracy %.2f%%' % (num, lr, accuracy * 100))
                if accuracy > 0.9: break
        except tf.errors.OutOfRangeError:  # 超出训练总轮数时
            print('Out of range!')
            break






