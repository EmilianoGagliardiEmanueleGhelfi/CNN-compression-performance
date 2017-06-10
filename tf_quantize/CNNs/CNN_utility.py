import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    # use get variable for variable reuse, it gets an another variable if it exists
    return tf.get_variable(name, shape=shape, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape=shape, initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
