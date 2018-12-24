# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Estimator结合Dataset"""

# # CSV of mnist
# import tensorflow as tf
#
# tf.logging.set_verbosity(v=tf.logging.INFO)
#
# def LeNet(x, is_train, params):  # 建立模型，加入正则
#     x = tf.reshape(x, shape=(-1, 28, 28, 1))
#     tfl = tf.layers
#     relu = tf.nn.relu
#     net = tfl.Conv2D(32, kernel_size=5, activation=relu)(x)
#     net = tfl.MaxPooling2D(2, strides=2)(net)
#     net = tfl.Conv2D(64, kernel_size=3, activation=relu)(net)
#     net = tfl.MaxPooling2D(2, strides=2)(net)
#     net = tfl.Flatten()(net)
#     if is_train:
#         l2 = params['regularizer'](scale=params['regularizer-scale'])
#         net = tfl.Dense(1024, activation=relu, kernel_regularizer=l2)(net)
#         net = tfl.Dropout(0.5)(net)
#         return tfl.Dense(10, kernel_regularizer=l2)(net)
#     else:
#         net = tfl.Dense(1024, activation=relu)(net)
#         return tfl.Dense(10)(net)
#
# def parse(line):  # 解析样例
#     line = tf.decode_csv(line, record_defaults=model_params['record_defaults'])
#     example = tf.cast(line[1:], dtype=tf.float32) / 255
#     # label = tf.cast(line[0], dtype=tf.int32)
#     label = line[0]
#     return example, label
#
# def model_fn(features, labels, mode, params):  # 使用模型
#     predict = LeNet(features['input1'], mode == tf.estimator.ModeKeys.TRAIN, params)
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions={
#             'predict': tf.argmax(predict, 1)
#         })
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict))
#     loss += tf.losses.get_regularization_loss()  # 加入正则损失
#     if mode == tf.estimator.ModeKeys.EVAL:
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={
#             'metric1': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(predict, 1))
#         })
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning-rate'])
#     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
#     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
# def csv_input_fn(path, batch_size=128, shuffle=True, repeat=None):  # 定义数据集
#     ds = tf.data.TextLineDataset(filenames=path).map(map_func=parse)  # 先map，再batch
#     if shuffle:
#         ds = ds.shuffle(buffer_size=256)
#     ds = ds.batch(batch_size=batch_size).repeat(count=repeat)
#     example_batch, label_batch = ds.make_one_shot_iterator().get_next()
#     return {'input1': example_batch}, label_batch
#
# model_params = {  # 超参
#     'learning-rate': 0.01,
#     'regularizer': tf.contrib.layers.l2_regularizer,
#     'regularizer-scale': 0.001,
#     'record_defaults': [[0] for _ in range(784 + 1)]
# }
# estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)  # estimator声明
#
# # 训练
# train_input_fn = lambda: csv_input_fn(['data\\mnist\\mnist_train.csv'])
# estimator.train(input_fn=train_input_fn, steps=2000)
#
# # 测试
# test_input_fn = lambda: csv_input_fn(['data\\mnist\\mnist_test.csv'], 1000, False)
# test_re = estimator.evaluate(input_fn=test_input_fn, steps=1)
# print(test_re)
#
# # 预测
# predict_input_fn = lambda: csv_input_fn(['data\\mnist\\mnist_test.csv'], 1, False, 1)
# predict_re = estimator.predict(input_fn=predict_input_fn)
# for p in predict_re:
#     print(p); break


# # CSV to TFRecords
# import numpy as np
# import tensorflow as tf
#
# f = open('data\\mnist\\mnist_test.csv', 'r')
# writer = tf.python_io.TFRecordWriter(path='TFRecord\\mnist.predict1.tfrecords')  # 产生TFRecord文件写的句柄
# end = -10
# for i in f:
#     data = i.strip('\n').split(',')
#     label = int(data[0])
#     image = np.array(data[1:], dtype=np.float32) / 255
#     pixels = image.size
#     image_raw = image.tostring()  # 将图像矩阵转化成字符串
#     # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'pixels': tf.train.Feature(int64_list=tf.train.Int64List(value=[pixels])),
#         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#         'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
#     }))
#     # 将一个Example写入TFRecord文件中
#     writer.write(record=example.SerializeToString())
#     end += 1
#     if not end: break
# writer.close()
# f.close()


# TFRecords of mnist
import os
import tensorflow as tf

def LeNet(x, is_train, params):  # 建立模型，加入正则
    x = tf.reshape(x, shape=(-1, 28, 28, 1))
    tfl = tf.layers
    relu = tf.nn.relu
    net = tfl.Conv2D(32, kernel_size=5, activation=relu)(x)
    net = tfl.MaxPooling2D(2, strides=2)(net)
    net = tfl.Conv2D(64, kernel_size=3, activation=relu)(net)
    net = tfl.MaxPooling2D(2, strides=2)(net)
    net = tfl.Flatten()(net)
    if is_train:
        l2 = params['regularizer'](scale=params['regularizer-scale'])
        net = tfl.Dense(1024, activation=relu, kernel_regularizer=l2)(net)
        net = tfl.Dropout(0.5)(net)
        return tfl.Dense(10, kernel_regularizer=l2)(net)
    else:
        net = tfl.Dense(1024, activation=relu)(net)
        return tfl.Dense(10)(net)

def parse(record):  # 解析样例
    features = tf.parse_single_example(serialized=record, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'pixels': tf.FixedLenFeature([], tf.int64)
    })
    shape = tf.stack(values=(features['pixels'],))  # 通过拼接得到形状
    example = tf.reshape(tf.decode_raw(features['image_raw'], out_type=tf.float32), shape=shape)
    label = tf.cast(features['label'], dtype=tf.int32)
    return example, label

def model_fn(features, labels, mode, params):  # 使用模型
    predict = LeNet(features['input1'], mode == tf.estimator.ModeKeys.TRAIN, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={
            'predict': tf.argmax(predict, 1)
        })
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict))
    loss += tf.losses.get_regularization_loss()  # 加入正则损失
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={
            'metric1': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(predict, 1))
        })
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning-rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # train_op = optimizer.compute_gradients(loss)
    # train_op = optimizer.apply_gradients(train_op, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def dataset_input_fn(path, batch_size=128, shuffle=True, repeat=None):  # 定义数据集
    ds = tf.data.TFRecordDataset(filenames=path).map(map_func=parse)  # 先map，再batch
    if shuffle:
        ds = ds.shuffle(buffer_size=256)
    ds = ds.batch(batch_size=batch_size).repeat(count=repeat)
    example_batch, label_batch = ds.make_one_shot_iterator().get_next()
    return {'input1': example_batch}, label_batch

def main(argv=None):  # 主函数
    tf.logging.set_verbosity(v=tf.logging.INFO)  # 日志屏幕上显示
    model_params = {  # 超参
        'learning-rate': 0.01,
        'regularizer': tf.contrib.layers.l2_regularizer,
        'regularizer-scale': 0.001,
        'model-dir': 'model\\mnist_estimator'
    }
    """estimator模型保存配置：
        1.模型保存后，若测试时指定配置项，会自动调用已有模型然后测试；
        2.若模型已保存，再指定相同配置项训练时，会在已有模型的基础上进行再训练"""
    ckpt_config = tf.estimator.RunConfig(
        model_dir=model_params['model-dir'],  # 模型保存路径
        save_checkpoints_secs=10,  # 每10秒保存一次
        keep_checkpoint_max=3  # 保留最近的3次
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params,  # estimator声明
                                       config=ckpt_config)

    if not os.path.exists(model_params['model-dir']):  # 训练
        train_input_fn = lambda: dataset_input_fn(['TFRecord\\mnist.train.tfrecords'])
        estimator.train(input_fn=train_input_fn, steps=10000)

    # 测试
    test_input_fn = lambda: dataset_input_fn(['TFRecord\\mnist.test.tfrecords'], 5000, False)
    test_re = estimator.evaluate(input_fn=test_input_fn, steps=1)
    print(test_re)

    # 预测
    predict_input_fn = lambda: dataset_input_fn(['TFRecord\\mnist.predict1.tfrecords'], 1, False, 1)
    predict_re = estimator.predict(input_fn=predict_input_fn)
    for p in predict_re:
        print(p)

    # 把计算图输出到tensorboard日志文件
    writer = tf.summary.FileWriter('model\\mnist_log', graph=tf.get_default_graph())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
