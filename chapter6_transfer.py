# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 参数
C = {
    'lr': 0.01, 'batch-size': 32, 'n-classes': 5, 'batch-display': 100,
    'checkpoint-exclude-scopes': 'InceptionV3/Logits,InceptionV3/AuxLogits',
    'trainable-scopes': 'InceptionV3/Logits,InceptionV3/AuxLogits'
}

# 获取训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in C['checkpoint-exclude-scopes'].split(',')]
    variables_to_restore = []
    # 列举模型所有参数，并判断是否需要从列表中删除
    for var in slim.get_model_variables():
        flag = False
        for exclusion in exclusions:
            if var.op.name.startwitch(exclusion):
                flag = True
                break
        if not flag: variables_to_restore.append(var)
    return variables_to_restore

# 获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in C['trainable-scopes'].split(',')]
    variables_to_train = []
    # 列举所有需要训练的参数前缀，并通过前缀找到所有参数
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train



print(get_trainable_variables())
