# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""Tensorflow计算加速
    1.同台电脑中所有CPU在TF的中的名称统一为/cpu:0；而不同GPU名称是不同的，第n个GPU为/gpu:(n-1)
    2.一个比较好的实践是将计算密集型的运行放在GPU上，把其它操作放在CPU上。所以，为了提高运行速度，尽量把相关运算
放在同一个设备上
    3.GPU是相对独立的资源，将计算放入或转出GPU都需要额外的时间，而且GPU需要将计算的数据从内存复制到GPU上
    4.深度学习常用并行模式：同步模式和异步模式。相同点：相同的FP和BP结构，独立计算，随机获取不同的小批量训练数据
    5.异步模式：不同设备获取参数取值的时间不同，得到的值也会不同，会各自对参数进行更新，所以容易跳过极小值，
使用异步模式可能无法达到较优的效果；
    6.同步模式：不同设备同时取参数的取值，单个设备不会对参数进行更新，而会等待所有设备都完成反向传播之后，
计算所有设备参数梯度的平均值，再根据平均值统一更新参数。
    7.同步模式的精确度虽高于异步模式，但效率却低于异步模式，再加之随机梯度下降本身就是梯度下降的近似解法，所以即使
梯度下降也无法保证达到全局最优值。所以在实际应用中，异步模式不一定比同步模式差。若多个GPU性能相似，优先使用同步模式
"""

import tensorflow as tf

# # 指定不同设备运行
# with tf.device('/cpu:0'):
#     c1 = tf.Variable(tf.truncated_normal([3, 4]), name='c1')
#     c2 = tf.Variable(tf.ones([4, 1]), name='c2')
# with tf.device('/gpu:0'):
#     m = tf.matmul(c1, c2)
#     o = m + tf.Variable(1.0)
# conf = tf.ConfigProto(
#     log_device_placement=True,  # 输出运行每一个运算的设备
#     allow_soft_placement=True  # 自动将无法放在GPU上的操作放回CPU上
# )
# # TF默认一次性占用一个GPU的所有显存，但也支持动态分配GPU显存，实现运行TF时也可运行其它任务
# # conf.gpu_options.allow_growth = True  # 方法一：按需分配
# conf.gpu_options.per_process_gpu_memory_fraction = 0.8  # 方法二：指定分配比例
# with tf.Session(config=conf) as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(o))


# 多GPU并行计算
import tensorflow as tf
import chapter6_net
import time

BATCH_SIZE = 100
NUM_GPU = 1
TRAINING_STEPS = 10000

# 数据输入队列
def get_input(path):
    ds = tf.data.TFRecordDataset(path).repeat()
    def parse(record):  # 解析样例
        features = tf.parse_single_example(serialized=record, features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'pixels': tf.FixedLenFeature([], tf.int64)
        })
        example = tf.reshape(tf.decode_raw(features['image_raw'], out_type=tf.float32),
                             shape=(28, 28, 1))
        label = tf.cast(features['label'], dtype=tf.int32)
        return example, label
    ds = ds.map(parse).shuffle(buffer_size=500).batch(batch_size=BATCH_SIZE)
    examples, labels = ds.make_one_shot_iterator().get_next()
    return examples, labels

# le_net5网络
def LeNet(x, is_train, regularizer):
    tfl = tf.layers
    relu = tf.nn.relu
    net = tfl.conv2d(x, 32, kernel_size=5, activation=relu)
    net = tfl.max_pooling2d(net, 2, strides=2)
    net = tfl.conv2d(net, 64, kernel_size=3, activation=relu)
    net = tfl.max_pooling2d(net, 2, strides=2)
    net = tfl.flatten(net)
    if is_train:
        net = tfl.dense(net, 1024, activation=relu, kernel_regularizer=regularizer)
        net = tfl.dropout(net, 0.8)
        return tfl.dense(net, 10, kernel_regularizer=regularizer)
    else:
        net = tfl.dense(net, 1024, activation=relu)
        return tfl.dense(net, 10)

# 单个GPU的损失函数
def get_loss(inputs, labels, scope, reuse, is_train):
    l2 = tf.contrib.layers.l2_regularizer(scale=0.99)
    # with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    #     # outputs = chapter6_net.LeNet_5(inputs, 10, regularizer=l2)
    #     outputs = LeNet(inputs, is_train, regularizer=l2)
    if reuse: tf.get_variable_scope().reuse_variables()
    outputs = chapter6_net.LeNet_5(inputs, 10, regularizer=l2)
    # outputs = LeNet(inputs, is_train, regularizer=l2)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
    loss += tf.add_n(tf.get_collection('losses', scope=scope))
    # loss += tf.losses.get_regularization_loss(scope=scope)
    return loss

# 平均梯度
def average_gradients(tower_grads):
    avg_grads = []
    for grad_and_vars in zip(*tower_grads):  # 遍历所有变量和变量在不同GPU上的梯度
        grads = []
        for grad, _ in grad_and_vars:  # 遍历同一组参数在不同GPU上的梯度
            # grad = tf.expand_dims(grad, 0)  # 多一个维度是为了连接之后求平均值
            grads.append(grad)
        # grad = tf.concat(grads, axis=0)
        avg_grad = tf.reduce_mean(grads, axis=0)  # 可直接对列表进行求平均值
        grad_and_var = grad_and_vars[0][1]  # 同一组参数中每个变量都一样，所以取第一个就行
        avg_grads.append((avg_grad, grad_and_var))  # 组成元祖追加到列表中
    return avg_grads


# 主训练过程
def main(_):
    # 神经网络训练参数用GPU，其它运算用CPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        inputs, labels = get_input(['TFRecord\\mnist.train.tfrecords'])
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        lr = tf.train.exponential_decay(0.1, global_step, 300, 0.96)
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        # 神经网络的GPU并行优化
        tower_grads, reuse_variables = [], False
        for i in range(NUM_GPU):  # 1个GPU并行
            with tf.device('/gpu:%d' % i), tf.name_scope('GPU_%d' % i) as scope:
                cur_loss = get_loss(inputs, labels, scope, reuse_variables, True)
                reuse_variables = True
                grads = opt.compute_gradients(cur_loss)  # 根据损失计算当前的各个参数的梯度
                tower_grads.append(grads)
        # 计算参数的平均梯度
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None: tf.summary.histogram('average_gradients/%s'%var.op.name, grad)
        # 使用平均梯度更新参数
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for v in tf.trainable_variables():
            tf.summary.histogram(v.op.name, v)
        # # 计算变量的平均滑动值
        # variable_average = tf.train.ExponentialMovingAverage(0.99, num_updates=global_step)
        # variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
        # variables_average_op = variable_average.apply(variables_to_average)
        # # 每轮迭代需要更新的变量取值和滑动平均值
        # train_op = tf.group(apply_gradient_op, variables_average_op)
        # 运行
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                                           log_device_placement=True))
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter('log_parallel', graph=tf.get_default_graph())
        for step in range(TRAINING_STEPS):
            start_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, cur_loss])
            duration = time.time() - start_time + 1e-3
            if step % 100 == 0:  # 查看当前进度和统计训练速度
                num_examples_per_step = BATCH_SIZE * NUM_GPU
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / NUM_GPU
                # 输出信息
                print('%s: step %d, loss %.4f (%.1f examples/sec; %.3f sec/batch)' %
                      (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                       step, loss_value, examples_per_sec, sec_per_batch))
            # 训练过程可视化
            summary = sess.run(summary_op)
            summary_writer.add_summary(summary, global_step=step)
            # 每隔一段时间保存当前模型
            if step % 1000 == 0 or (step + 1) == TRAINING_STEPS:
                saver.save(sess, save_path='log_parallel\\parallel', global_step=step + 1)
        summary_writer.close()
        sess.close()


if __name__ == '__main__':
    tf.app.run()



