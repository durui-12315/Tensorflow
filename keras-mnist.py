# -*- coding:UTF-8 -*-
# !/usr/bin/python

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# 获取数据
(x_train, y_train), (x_test, y_test) = mnist.load_data('E:\#AI\Excercise\TensorFlow\data\\mnist\\mnist.npz')

# 数据归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 变形
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

# # 标签转成独热编码
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# # 模型构建1
# model = Sequential()
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# 模型构建2
input = Input((*input_shape,), name='input')
net = Conv2D(32, kernel_size=3, activation='relu')(input)
net = Conv2D(64, kernel_size=3, activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)
net = Dropout(0.25)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dropout(0.5)(net)
output = Dense(num_classes, activation='softmax')(net)
model = Model(inputs=input, outputs=output)

# 定义损失、优化
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))

# 评测
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %.4f, accuracy: %.2f%%' % (score[0], score[1] * 100))

