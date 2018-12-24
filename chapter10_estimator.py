# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Tensorflow高层封装: Estimator"""

# # Estimator基本用法
# import numpy as np
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.logging.set_verbosity(tf.logging.INFO)  # 日志信息输出到屏幕
# mnist = input_data.read_data_sets(train_dir='data\\mnist', one_hot=False)
#
# # 指定输入层。所有的输入都会拼接在一起作为整个网络的输入
# feature_columns = [tf.feature_column.numeric_column(key='input1', shape=(784,))]
#
# # 模型定义，只用2层全连接hidden_units=[500, 250]
# estimator = tf.estimator.DNNClassifier(hidden_units=[500, 250],
#                                        feature_columns=feature_columns,
#                                        n_classes=10,
#                                        optimizer=tf.train.AdamOptimizer(),
#                                        model_dir='model\\log')
#
# # 定义数据输入，和输入层一一对应
# train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'input1': mnist.train.images},
#                                                     y=mnist.train.labels.astype(np.int32),
#                                                     num_epochs=None,
#                                                     batch_size=128,
#                                                     shuffle=True)
#
# # 训练
# estimator.train(input_fn=train_input_fn, steps=10000)
#
# # 定义测试时的输入
# test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'input1': mnist.test.images},
#                                                    y=mnist.test.labels.astype(np.int32),
#                                                    num_epochs=1,
#                                                    batch_size=128,
#                                                    shuffle=False)
#
# # 评测
# acc_score = estimator.evaluate(input_fn=test_input_fn)
# print(acc_score)


# # Estimator自定义模型1
# import numpy as np
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.logging.set_verbosity(v=tf.logging.INFO)  # 日志输出到屏幕
#
# # 定义模型
# def LeNet(x, is_train):
#     x = tf.reshape(x, shape=(-1, 28, 28, 1))  # 变形
#     net = tf.layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu)(x)
#     net = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(net)
#     net = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(net)
#     net = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(net)
#     net = tf.layers.Flatten()(net)
#     net = tf.layers.Dense(units=1024, activation=tf.nn.relu)(net)
#     net = tf.layers.dropout(net, rate=0.5, training=is_train)
#     net = tf.layers.Dense(units=10)(net)
#     return net
#
# # 模型使用
# def model_fn(features, labels, mode, params):
#     # 前向传播
#     predict = LeNet(features['input1'], mode == tf.estimator.ModeKeys.TRAIN)
#     # 返回预测，若使用预测模式，运行predict时
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions={'result': tf.argmax(predict, 1)})
#     # 定义损失函数
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict))
#     # 优化器
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning-rate'])
#     # 定义训练
#     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
#     # 评价标准，运行evaluate时
#     eval_metric_ops = {'my_metric': tf.metrics.accuracy(tf.argmax(predict, axis=1), labels)}
#     # 返回损失函数、训练过程、评测，若使用训练模式，运行train时
#     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
#                                       eval_metric_ops=eval_metric_ops)
#
# # 数据与超参提供
# mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
# model_params = {'learning-rate': 0.01}
#
# # 生成Estimator类
# estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
#
# # 训练
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'input1': mnist.train.images},
#     y=mnist.train.labels.astype(np.int32),
#     num_epochs=None,
#     batch_size=128,
#     shuffle=True)
# estimator.train(input_fn=train_input_fn, steps=10000)
#
# # 测试
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'input1': mnist.test.images},
#     y=mnist.test.labels.astype(np.int32),
#     num_epochs=1,
#     batch_size=5000,
#     shuffle=False)
# test_result = estimator.evaluate(input_fn=test_input_fn)
# print(test_result)
#
# # 预测
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'input1': mnist.test.images[:10]},
#     num_epochs=1,
#     shuffle=False)
# predict_result = estimator.predict(input_fn=predict_input_fn)
# for p in predict_result:
#     print(p)


# Estimator自定义模型--默写
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(v=tf.logging.INFO)

def le_net(x, is_training):  # 建立模型
    x = tf.reshape(x, shape=(-1, 28, 28, 1))
    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=is_training)
    return tf.layers.dense(net, 10)

def model_fn(features, labels, mode, params):  # 使用模型
    predict = le_net(features['input1'], mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'result': tf.argmax(predict, axis=1)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning-rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    accuracy = {'metric1': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(predict, axis=1))}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=accuracy)

mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
model_params = {'learning-rate': 0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

train_input_fn = tf.estimator.inputs.numpy_input_fn(  # 训练
    x={'input1': mnist.train.images}, y=mnist.train.labels.astype(np.int32),
    batch_size=128, num_epochs=None, shuffle=True)
estimator.train(input_fn=train_input_fn, steps=100)

test_input_fn = tf.estimator.inputs.numpy_input_fn(  # 测试
    x={'input1': mnist.test.images}, y=mnist.test.labels.astype(np.int32),
    batch_size=5000, num_epochs=1, shuffle=False)
test_re = estimator.evaluate(input_fn=test_input_fn)
print(test_re)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(  # 预测
    x={'input1': mnist.train.images[np.random.randint(0, 10000, size=(10,))]}, num_epochs=1, shuffle=False)
predict_re = estimator.predict(input_fn=predict_input_fn)
for p in predict_re:
    print(p)

