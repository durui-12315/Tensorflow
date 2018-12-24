# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""分布式Tensorflow:
    1.分布式TF训练深度学习有2种方式：计算图内分布式、计算图之间分布式
    2.使用计算图内分布式时，所有任务都会使用一个TF计算图中的参数，而只是将计算部分发布到不同的计算服务器上
    3.使用计算图之间分布式时，每一个计算服务器都会创建一个独立的TF计算图，但不同计算图中的相同参数需要以
一种固定的方式放到同一个参数服务器上。此种方式并行度最好。
    4.参数服务器只负责变量的维护和管理，计算服务器负责每一轮迭代时运行反向传播过程
    5.和异步模式不同，同步模式下global_step差不多是每个计算服务器的local_step的均值
"""

import tensorflow as tf

# # 一个TF集群执行两个任务
# cluster = tf.train.ClusterSpec(cluster={'local': ['localhost:2222', 'localhost:2223']})
# # 任务1
# c = tf.constant('Hello from sever1')
# server = tf.train.Server(cluster, job_name='local', task_index=0)
# sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))
# server.join()
# # 任务2
# c = tf.constant('Hello from sever2')
# server = tf.train.Server(cluster, job_name='local', task_index=1)
# sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
# print(sess.run())
# server.join()


# # 计算图之间分布式：异步模式
# import time
# import tensorflow as tf
# import chapter6_net
# from tensorflow.examples.tutorials.mnist import input_data
#
# # 通过FLAGS运行指定的参数
# FLAGS = tf.app.flags.FLAGS
# # 指定当前运行的服务器类别：ps OR worker，默认值为第二个参数
# tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
# # 指定集群中参数服务器地址
# tf.app.flags.DEFINE_string('ps_hosts', 'localhost:2222',
#     'Comma-separated list of hostname: port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
# # 指定集群中计算服务器地址
# tf.app.flags.DEFINE_string('worker_hosts', 'localhost:2223,localhost:2224',
#     'Comma-separated list of hostname: port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
# # 指定当前程序的任务ID。注：1.TF会根据ps和worker列表中的端口号启动服务；2.ID从0开始
# tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
#
# def build_mode(x, y_, is_chief):  # 定义计算图
#     y = chapter6_net.LeNet_5(x, 10, regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
#     # global_step = tf.contrib.framework.get_or_create_global_step()
#     global_step = tf.train.get_or_create_global_step()
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
#     loss += tf.add_n(tf.get_collection('losses'))
#     lr = tf.train.exponential_decay(0.1, global_step, 300, 0.96)
#     opt_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
#     if is_chief:  # 计算滑动平均
#         variable_averages = tf.train.ExponentialMovingAverage(0.99, num_updates=global_step)
#         variable_averages_op = variable_averages.apply(tf.trainable_variables())
#         with tf.control_dependencies([variable_averages_op, opt_op]):
#             train_op = tf.no_op()  # nothing to do 等价：train_op = tf.group([variable_averages_op, opt_op])
#     return global_step, loss, train_op
#
# def main(_):
#     # 解析flags，配置TF集群
#     ps_hosts = FLAGS.ps_hosts.split(',')
#     worker_hosts = FLAGS.worker_hosts.split(',')
#     cluster = tf.train.ClusterSpec(cluster={'ps': ps_hosts, 'worker': worker_hosts})
#     # 通过cluster创建server
#     server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)
#     # 参数服务器只需管理变量，不训练变量，故调用server.join()无休止等待，若为计算服务器，直接跳过
#     if FLAGS.job_name == 'ps':
#         with tf.device('/cpu:0'): server.join()
#     # 通过replica_device_setter函数指定执行每个运算设备。它把参数自动分配到ps，把计算分配到当前的worker
#     device_setter = tf.train.replica_device_setter(
#         worker_device='/job:worker/task:%d' % FLAGS.task_id, cluster=cluster)
#     # 定义计算服务器运行的操作
#     is_chief = FLAGS.task_id == 0
#     mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
#     with tf.device(device_setter):
#         x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='x-input')
#         y_ = tf.placeholder(tf.int32, shape=(None,), name='y-input')
#         global_step, loss, train_op = build_mode(x, y_, is_chief=is_chief)
#         sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         # 管理训练深度学习模型的通用功能，比tf.Session强大
#         with tf.train.MonitoredTrainingSession(
#             master=server.target, is_chief=is_chief,  checkpoint_dir='log_async',
#             hooks=[tf.train.StopAtStepHook(last_step=10000)],
#             save_checkpoint_secs=60, config=sess_config) as mon_sess:
#             print('Session started.')
#             step, start_time = 0, time.time()
#             # 执行迭代过程，过程中MonitoredTrainingSession会帮助完成初始化、加载训练过的模型、输出日志、模型保存
#             # StopAtStepHook会帮忙判断是否需要退出
#             with not mon_sess.should_shop():
#                 xs, ys = mnist.train.next_batch(100)
#                 xs = tf.reshape(xs, shape=(-1, 28, 28, 1))
#                 _, loss_v, global_step_v = mon_sess.run([train_op, loss, global_step],
#                                                         feed_dict={x: xs, y_: ys})
#                 # 不同的计算服务器都会更新全局的训练轮数，故global_step_v得到的是全局训练轮数
#                 if step % 100 == 0:
#                     duration = time.time() - start_time
#                     sec_per_batch = duration / global_step_v
#                     print('After %d training steps (%d global steps), loss %g (%.3f sec/batch)' %
#                           (step, global_step_v, loss_v, sec_per_batch))
#                 step += 1
#
#
# if __name__ == '__main__':
#     tf.app.run()


# 计算图之间分布式：同步模式。
import time
import tensorflow as tf
import chapter6_net
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string('ps_hosts', 'localhost:2222',
    'Comma-separated list of hostname: port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:2223,localhost:2224',
    'Comma-separated list of hostname: port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')

def build_mode(x, y_, n_workers, is_chief):
    y = chapter6_net.LeNet_5(x, 10, regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
    global_step = tf.train.get_or_create_global_step()
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    loss += tf.add_n(tf.get_collection('losses'))
    lr = tf.train.exponential_decay(0.1, global_step, 300, 0.96)
    # 实现同步更新。
    opt = tf.train.SyncReplicasOptimizer(opt=tf.train.GradientDescentOptimizer(learning_rate=lr),
        # replicas_to_aggregate要求收集n_workers份梯度才会进行均化更新，但不要求每份梯度都来自不同计算服务器
        replicas_to_aggregate=n_workers, total_num_replicas=n_workers)
    sync_replicas_hook = opt.make_session_run_hook(is_chief=is_chief)
    train_op = opt.minimize(loss, global_step=global_step)
    # 滑动平均
    variables_average = tf.train.ExponentialMovingAverage(0.99, num_updates=global_step)
    variables_average_op = variables_average.apply(tf.trainable_variables())
    train_op = tf.group([variables_average_op, train_op])
    return global_step, loss, train_op, sync_replicas_hook

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec(cluster={'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)
    if FLAGS.job_name == 'ps':
        with tf.device('/cpu:0'): server.join()
    is_chief = FLAGS.task_id == 0
    mnist = input_data.read_data_sets('data\\mnist', one_hot=False)
    device_setter = tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,
                                                   cluster=cluster)
    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='x-input')
        y_ = tf.placeholder(tf.int32, shape=(None,), name='y-input')
        global_step, loss, train_op, sync_replicas_hook = build_mode(x, y_, n_workers, is_chief)
        # 把处理同步更新的hook也加进来
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=10000)]
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,
                                               checkpoint_dir='log_async', hooks=hooks,
                                               save_checkpoint_secs=60, config=sess_config) as mon_sess:
            print('Session started.')
            step, start_time = 0, time.time()
            while not mon_sess.should_stop():
                xs, ys = mnist.train.next_batch(100)
                xs = tf.reshape(xs, shape=(-1, 28, 28, 1))
                _, loss_v, global_step_v = mon_sess.run([train_op, loss, global_step],
                                                        feed_dict={x: xs, y_: ys})
                if step % 100 == 0:
                    sec_per_batch = (time.time() - start_time) / global_step_v
                    print('After %d training steps (%d global steps), loss %g (%.3f sec/batch)'
                          % (step, global_step_v, loss_v, sec_per_batch))
                step += 1


if __name__ == '__main__':
    tf.app.run()
