# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Tensorflow高层封装: Keras"""

# # 1.使用原生态keras在mnist数据集上实现LeNet-5模型
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# import keras.backend as K
#
# # 参数
# C = {'n-classes': 10, 'img-rows': 28, 'img-cols': 28,
#      'data-path': 'E:\\#AI\\Excercise\\TensorFlow\\data\\mnist\\mnist.npz'}
#
# # 加载数据
# (trainX, trainY), (testX, testY) = mnist.load_data(path=C['data-path'])
#
# # 设置mnist黑白图像编码格式
# if K.image_data_format() == 'channels_first':
#     trainX = trainX.reshape(trainX.shape[0], 1, C['img-rows'], C['img-cols'])
#     testX = testX.reshape(testX.shape[0], 1, C['img-rows'], C['img-cols'])
#     input_shape = (1, C['img-rows'], C['img-cols'])
# else:
#     trainX = trainX.reshape(trainX.shape[0], C['img-rows'], C['img-cols'], 1)
#     testX = testX.reshape(testX.shape[0], C['img-rows'], C['img-cols'], 1)
#     input_shape = (C['img-rows'], C['img-cols'], 1)
#
# # 图像像素归一化
# trainX = trainX.astype('float32') / 255.0
# testX = testX.astype('float32') / 255.0
#
# # 标签one-hot转化
# trainY = keras.utils.to_categorical(trainY, C['n-classes'])
# testY = keras.utils.to_categorical(testY, C['n-classes'])
#
# # 使用kerasAPI定义LeNet-5模型
# model = Sequential()
# # 深度32，大小5*5的卷积核
# model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
# # 2*2最大池化层
# model.add(MaxPooling2D())  # 默认值为(2, 2)
# # 深度64，大小5*5的卷积核
# model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# # 2*2最大池化层
# model.add(MaxPooling2D())
# # 矩阵拉伸
# model.add(Flatten())
# # 全连接
# model.add(Dense(units=500, activation='relu'))
# # 最后输出
# model.add(Dense(units=C['n-classes'], activation='softmax'))
#
# # 定义损失函数、优化、评测
# model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.categorical_crossentropy,
#               metrics=['accuracy'])
#
# # 训练过程
# model.fit(x=trainX, y=trainY, batch_size=128, epochs=20, validation_data=(testX, testY))
#
# # 在测试数据上计算准确率
# score = model.evaluate(x=testX, y=testY)
# print('Test losses:', score[0])
# print('Test accuracy:', score[1])


# # 2.RNN建立情感分析模型
# from keras.preprocessing import sequence  # 预处理
# from keras.layers import Dense, Embedding, LSTM  # 层
# from keras.models import Sequential  # 模型
# from keras.datasets import imdb  # 数据
#
# # 参数
# C = {'max-features': 20000, 'maxlen': 80, 'batch-size': 200, 'hidden-size': 128,
#      'data-path': 'E:\\#AI\\Excercise\\TensorFlow\\data\\imdb.npz'}
#
# # 加载数据，并将单词转化为ID。和NPL一样，将频率较低的单词替换为统一的ID
# (trainX, trainY), (testX, testY) = imdb.load_data(path=C['data-path'], num_words=C['max-features'])
#
# # NLP中每段话的长度是不一样的，但RNN的循环长度是固定的，所以需要将所有段落统一固定长度，短者0填充，长者截断
# trainX = sequence.pad_sequences(sequences=trainX, maxlen=C['maxlen'])
# testX = sequence.pad_sequences(sequences=testX, maxlen=C['maxlen'])
#
# # 模型构建
# model = Sequential()
# # embedding层，起降维作用
# model.add(Embedding(input_dim=C['max-features'], output_dim=C['hidden-size']))  # 20000行，128列
# # LSTM层，默认得到最后一个节点输出，若得出每个时序的结果，则return_sequences=True
# model.add(LSTM(units=C['hidden-size'], dropout=0.2, recurrent_dropout=0.2))
# # 全连接层
# model.add(Dense(units=1, activation='sigmoid'))
#
# # 损失函数、优化、评测
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # 训练
# model.fit(x=trainX, y=trainY, batch_size=C['batch-size'], epochs=2)
#
# # 在测试数据上计算准确率
# score = model.evaluate(x=testX, y=testY, batch_size=500)
# print('The loss:', score[0])
# print('The accuracy:', score[1])


# # 3.以返回值的形式定义网络结构
# import keras
# from keras.models import Model
# from keras.datasets import mnist
# from keras.layers import Input, Dense
#
# # 参数
# C = {'n-classes': 10, 'img-rows': 28, 'img-cols': 28,
#      'data-path': 'E:\\#AI\\Excercise\\TensorFlow\\data\\mnist\\mnist.npz'}
#
# # 加载数据
# (trainX, trainY), (testX, testY) = mnist.load_data(path=C['data-path'])
#
# # 预处理
# trainX = trainX.reshape(trainX.shape[0], C['img-rows'] * C['img-cols'])
# testX = testX.reshape(testX.shape[0], C['img-rows'] * C['img-cols'])
# trainX = trainX.astype('float32') / 255.0
# testX = testX.astype('float32') / 255.0
# trainY = keras.utils.to_categorical(trainY, C['n-classes'])
# testY = keras.utils.to_categorical(testY, C['n-classes'])
#
# # 模型构建（只用全连接层，不需要把输入整理成三维矩阵）
# inputs = Input(shape=(C['img-rows'] * C['img-cols'],))  # 输入层
# fc1 = Dense(units=500, activation='relu')(inputs)  # 全连接层1
# fc2 = Dense(units=C['n-classes'], activation='softmax')(fc1)
# model = Model(inputs=inputs, outputs=fc2)
#
# # 损失、优化、评测
# model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.categorical_crossentropy,
#               metrics=['accuracy'])
#
# # 训练
# model.fit(x=trainX, y=trainY, batch_size=200, epochs=20, validation_data=(testX, testY))
#
# # 在测试数据上计算准确率
# score = model.evaluate(x=testX, y=testY)
# print('Test losses:', score[0])
# print('Test accuracy:', score[1])


# # 4.实现类似Inception模型结构
# from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
#
# inputs = Input(shape=(300, 300, 3))  # 图像输入
#
# # 分支1
# tower1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
# tower1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tower1)
#
# # 分支2
# tower2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
# tower2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(tower2)
#
# # 分支3
# tower3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
# tower3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(tower3)
#
# # 通道维度上合并分支
# output = concatenate(inputs=[tower1, tower2, tower3], axis=3)
# print(tower1.shape, tower2.shape, tower3.shape, output.shape)


# # 5.实现多输入、多输出
# import keras
# from keras.models import Model
# from keras.datasets import mnist
# from keras.layers import Input, Dense, concatenate
#
# # 参数
# C = {'n-classes': 10, 'img-rows': 28, 'img-cols': 28,
#      'data-path': 'E:\\#AI\\Excercise\\TensorFlow\\data\\mnist\\mnist.npz'}
#
# # 加载数据
# (trainX, trainY), (testX, testY) = mnist.load_data(path=C['data-path'])
#
# # 预处理
# trainX = trainX.reshape(trainX.shape[0], C['img-rows'] * C['img-cols'])
# testX = testX.reshape(testX.shape[0], C['img-rows'] * C['img-cols'])
# trainX = trainX.astype('float32') / 255.0
# testX = testX.astype('float32') / 255.0
# trainY = keras.utils.to_categorical(trainY, C['n-classes'])
# testY = keras.utils.to_categorical(testY, C['n-classes'])
#
# # 模型构建
# inputs1 = Input(shape=(C['img-rows'] * C['img-cols'],), name='input1')  # 图片信息输入
# inputs2 = Input(shape=(C['n-classes'],), name='input2')  # 正确答案输入
# fc1 = Dense(units=1, activation='relu')(inputs1)  # 只有1个隐藏节点的全连接层
# output1 = Dense(units=C['n-classes'], activation='softmax', name='output1')(fc1)  # 输出1
# fc2 = concatenate([fc1, inputs2], axis=1)  # 答案输入和输出1链接
# output2 = Dense(units=C['n-classes'], activation='softmax', name='output2')(fc2)  # 输出2
# model = Model(inputs=[inputs1, inputs2], outputs=[output1, output2])
#
# # 损失、优化、评测
# model.compile(optimizer=keras.optimizers.SGD(), loss={
#     'output1': keras.losses.categorical_crossentropy,  # 相同时，可以合二为一
#     'output2': keras.losses.categorical_crossentropy
# }, loss_weights=[1, 0.1], metrics=['accuracy'])  # loss_weights为损失权重，定义优化比重
#
# # 训练
# model.fit(x={'input1': trainX, 'input2': trainY}, y={'output1': trainY, 'output2': trainY},
#           batch_size=128, epochs=20,
#           validation_data=({'input1': testX, 'input2': testY}, {'output1': testY, 'output2': testY}))
#
# # 在测试数据上计算准确率
# score = model.evaluate(x={'input1': testX, 'input2': testY}, y={'output1': testY, 'output2': testY})
# print('total loss: %.2f, output1_loss: %.2f, output2_loss: %.2f' % (score[0], score[1], score[2]))
# print('output1_accuracy: %.2f%%, output2_accuracy: %.2f%%' % (score[3] * 100, score[4] * 100))


# 6.结合原生态TF，实现更灵活方式
import tensorflow as tf
from keras.layers import Dense
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets(train_dir='data\\mnist', one_hot=True)
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 784))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 10))

net = Dense(units=500, activation='relu')(inputs)
outputs = Dense(units=10, activation='softmax')(net)

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=outputs))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_true=labels, y_pred=outputs))  # 正确率

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    i = 0
    while True:
        xs, ys = mnist_data.train.next_batch(100)  # 批次执行，减少内存消耗
        _, loss_value = sess.run([optimizer, loss], feed_dict={inputs: xs, labels: ys})
        if i % 100 == 0:
            xs_, ys_ = mnist_data.test.next_batch(5000)
            acc_value = sess.run(acc, feed_dict={inputs: xs_, labels: ys_})
            print('Batch %d: loss %.4f, accuracy %.2f%%' % (i, loss_value, acc_value * 100))
            if acc_value > 0.97: break
        i += 1






