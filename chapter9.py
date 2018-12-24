# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""
语言模型简介：
    1.假设一门语言中所有可能的句子服从某一个概率分布，每个句子出现的概率加起来为1，那么“语言模型”的任务就是预测
每个句子在语言这种出现的概率。
    2.对于语言中常见的句子，好的语言模型应得出相对较高的概率；对于不合法的句子，计算出的概率应接近于0
    3.语言模型仅仅对句子出现的概率进行建模，并不尝试去“理解”句子的真实含义
    4.把句子看成单词的序列，语言模型可以表示为一个计算p(w1,w2,w3,...wm)的模型

语言模型评价方法：
    1.语言模型效果好坏的常用评价指标是复杂度(perplexity)，在测试集上得到对的perplexity越低，建模的效果越好
    2.perplexity的数学公式：perplexity(S) = [∏P(Xi|X1,X2,...Xi-1)]^(-1/m), i∈m
    3.perplexity可以理解为平均分支系数，即模型预测下一个词时的平均可选择数量
    4.perplexity的语言模型公式：log(perplexity(S)) = -1/m[∑logP(Xi|X1,X2,...Xi-1)]。相比数学公式，使用加法形式
可以加速计算，同时避免概率乘积数值过小而导致浮点数向下溢出的问题
"""

"""使用PTB数据上使用循环神经网络建立语言模型：向量层(embedding)、循环神经网络层(RNN)、softmax层"""

raw_path = 'data\\PTB\\data\\ptb.train.txt'
vocab = 'data\\PTB\\ptb.vocab'  # 词汇表文件
vocab_id = 'data\\PTB\\ptb.train'  # 单词编号

# import codecs
# import collections
#
# # PTB数据集预处理
# counter = collections.Counter()
# with open(raw_path) as f:
#     for line in f:
#         for word in line.strip().split():
#             counter[word] += 1  # 统计单词出现频率
# counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 排序
# counter = [i[0] for i in counter]
# counter = ['<eos>'] + counter
# with codecs.open(vocab, 'w', 'UTF-8') as f:
#     for w in counter:
#         f.write(w + '\n')  # 存储单词至文件
#
#
# # 词汇表到单词编号的转换
# def get_id(ws, w):
#     return ws[w] if w in ws else ws['<unk>']
# def hehe(a, b):
#     return a + b
#
# with codecs.open(vocab, 'r', 'UTF-8') as f:
#     words = [w.strip() for w in f.readlines()]
# words = {k: v for k, v in zip(words, range(len(words)))}
# fin = codecs.open(raw_path, 'r', 'UTF-8')
# fout = codecs.open(vocab_id, 'w', 'UTF-8')
# for line in fin:
#     li = line.strip().split() + ['<eos>']
#     li_id = ' '.join([str(get_id(words, l)) for l in li]) + '\n'
#     fout.write(li_id)
# fin.close()
# fout.close()


import numpy as np

# 所有词id放入一个数组中
# with open(file=vocab_id, mode='r', encoding='UTF-8') as f:
#     id_str = ' '.join([line.strip() for line in f.readlines()])
#     id_list = list(map(int, id_str.split()))
#     np.save('data\\PTB\\ptb.train.npy', arr=np.array(id_list, dtype=np.int32))

# 数据整理成batch：尽量保持batch之间的上下文连续
NUM_STEP, BATCH_SIZE = 35, 20
id_arr = np.load('data\\PTB\\ptb.train.npy')
num_batches = (id_arr.size - 1) // (BATCH_SIZE * NUM_STEP)
data = np.reshape(id_arr[:num_batches * BATCH_SIZE * NUM_STEP], newshape=(BATCH_SIZE, num_batches * NUM_STEP))
data = np.split(data, indices_or_sections=num_batches, axis=1)
label = []
for i in range(1, num_batches * BATCH_SIZE + 1): label.append(id_arr[i * NUM_STEP])
label = np.reshape(label, newshape=(BATCH_SIZE, num_batches))
label_ = []
for i in range(num_batches): label_.append(np.reshape(label[:, i], newshape=(BATCH_SIZE, 1)))
# label = np.reshape(id_arr[1:num_batches * BATCH_SIZE * NUM_STEP + 1],
#                    newshape=(BATCH_SIZE, num_batches * NUM_STEP))
# label = np.split(label, indices_or_sections=num_batches, axis=1)
train = list(zip(data, label_))





