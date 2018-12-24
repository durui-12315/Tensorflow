# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""基于平方和损失函数的BP神经网络"""
import numpy as np

class BPNN:
    def __init__(self, input_num, hidden_num, output_num, leargin_rate=0.005):
        self.__input_num = input_num
        self.__hidden_num = hidden_num
        self.__output_num = output_num
        self.__input_weight = np.random.normal(0, 0.1, size=(input_num, hidden_num))
        self.__hidden_weight = np.random.normal(0, 0.1, size=(hidden_num, output_num))
        self.learning_rate = leargin_rate

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, inputs, labels):
        inputs = np.array(inputs, ndmin=2)
        labels = np.array(labels, ndmin=2)
        hiddens_i = np.dot(inputs, self.__input_weight)
        hiddens_o = self.__sigmoid(hiddens_i)
        outputs_i = np.dot(hiddens_o, self.__hidden_weight)
        outputs_o = self.__sigmoid(outputs_i)
        # 输出层反向传播
        outputs_error = labels - outputs_o
        self.__hidden_weight += self.learning_rate * \
                                np.dot(hiddens_o.T, outputs_error * outputs_o * (1 - outputs_o))
        # 隐藏层反向传播
        hiddens_error = np.dot(outputs_error, self.__hidden_weight.T)
        self.__input_weight += self.learning_rate * \
                               np.dot(inputs.T, hiddens_error * hiddens_o * (1 - hiddens_o))

    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        hidden = self.__sigmoid(np.dot(inputs, self.__input_weight))
        output = self.__sigmoid(np.dot(hidden, self.__hidden_weight))[0]
        return np.argmax(output)

def number(n):
    arr = np.ones(shape=10) * 0.01
    arr[int(n)] = 0.99
    return arr

if __name__ == '__main__':
    train_data, test_data = np.load('data/mnist/mnist_train.npy'), np.load('data/mnist/mnist_test.npy')
    train_len, test_len = len(train_data), len(test_data)
    bp = BPNN(28 ** 2, 128, 10)
    batch, size = 0, 5000
    while True:
        # 模型训练
        start = 0
        for data in train_data[start:start + size]:
            bp.train(data[1:], number(data[0]))
        start += size
        if start >= train_len: start = 0
        if batch % 10 == 0:
            # 模型预测
            acc = 0
            for data in test_data:
                if bp.predict(data[1:]) == data[0]: acc += 1
            rate = acc / test_len
            print('Batch %d: accuracy %.2f%%' % (batch, rate * 100))
            if rate > 0.97: break
        batch += 1