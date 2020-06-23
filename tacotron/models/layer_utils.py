import tensorflow as tf


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())
