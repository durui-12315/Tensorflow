# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""基于softmax的交叉熵损失函数的BP神经网络"""
import numpy as np

class BPNN:
    def __init__(self, input_num, hidden_num, output_num, learning_rate=0.001):
        self.__input_num = input_num
        self.__hidden_num = hidden_num
        self.__output_num = output_num
        self.__input_weights = np.random.normal(0, 0.1, size=(input_num, hidden_num))
        self.__hidden_weights = np.random.normal(0, 0.1, size=(hidden_num, output_num))
        self.lr = learning_rate

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self, x):
        s = np.sum(np.exp(x))
        return x / s

    def train(self, inputs, labels):
        inputs = np.array(inputs, ndmin=2)
        labels = np.array(labels, ndmin=2)
        # 前向传播
        hidden_i = np.dot(inputs, self.__input_weights)
        hidden_o = self.__sigmoid(hidden_i)
        output_i = np.dot(hidden_o, self.__hidden_weights)
        output_o = self.__sigmoid(output_i)  # 原始输出层
        output_s = self.__softmax(output_o)  # softmax输出层
        # softmax层反向传播至原始输出层
        output_error = output_s - labels
        # 原始输出层反向传播至隐层
        self.__hidden_weights -= self.lr * np.dot(hidden_o.T, output_error * output_o * (1 - output_o))
        # 隐层反向传播
        hidden_error = np.dot(labels - output_s, self.__hidden_weights.T)
        self.__input_weights -= self.lr * np.dot(inputs.T, hidden_error * hidden_o * (1 - hidden_o))

    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        hidden = self.__sigmoid(np.dot(inputs, self.__input_weights))
        output = self.__sigmoid(np.dot(hidden, self.__hidden_weights))
        output_s = self.__softmax(output)[0]
        return np.argmax(output_s)

def number(n):
    arr = np.zeros(shape=10)
    arr[int(n)] = 1
    return arr

if __name__ == '__main__':
    train_data, test_data = np.load('data/mnist/mnist_train.npy'), np.load('data/mnist/mnist_test.npy')
    train_len, test_len = len(train_data), len(test_data)
    train_label = np.ones(shape=(train_len, 10)) * 0.01
    train_label[[i for i in range(train_len)], train_data[:, 0].astype('int')] = 0.99
    test_label = np.ones(shape=(test_len, 10)) * 0.01
    test_label[[i for i in range(test_len)], test_data[:, 0].astype('int')] = 0.99
    bp = BPNN(28 ** 2, 128, 10)
    batch, size = 0, 1000
    while True:
        # 模型训练
        start = 0
        for i in range(start, start + size):
            bp.train(train_data[i][1:], train_label[i][0])
        start += size
        if start >= train_len: start = 0
        if batch % 10 == 0:
            # 模型预测
            acc = 0
            for i in range(test_len):
                if bp.predict(test_data[i][1:]) == test_data[i][0]: acc += 1
            rate = acc / test_len
            print('Batch %d: accuracy %.2f%%' % (batch, rate * 100))
            if rate > 0.97: break
        batch += 1
