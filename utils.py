import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

def get_shape(tensor):
    return tensor.get_shape.as_list()

def deconvolution(x, filter_size, stride_size, output_size):
    '''
        input:
            x : input 4D tensor
            filter_size : lists of 4
            stride_size : lists of 4
            output_size : lists of 4
    '''
    w = tf.get_variable(name = 'deconv_weights',
                        shape = filter_size,
                        initializer= tf.random_normal_initializer())
    b = tf.get_variable(name = 'conv_biases',
                        shape = filter_size[2], 
                        initializer = tf.random_normal_initializer())

    return tf.nn.conv2d_transpose(x, w, strides = stride_size, output_shape= output_size, padding='SAME') + b

def convolution(x, height, width, in_layer, out_layer):
    w = tf.get_variable(name = 'conv_weights',
                        shape = [height, width, in_layer, out_layer],
                        initializer= tf.random_normal_initializer())
    b = tf.get_variable(name = 'conv_biases',
                        shape = [out_layer], 
                        initializer = tf.random_normal_initializer())
    return tf.nn.elu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID') + b)

def fully_connected(x, fan_in, fan_out):
    '''
        input:
            x - 1D tensor
            fan_in - num of input nodes
            fan_out - num of output nodes
        return :
            1D tensor
            linear regression according to activation function 
    '''
    w = tf.get_variable(name = 'fc_weights',
                        shape = [fan_in, fan_out],
                        initializer= tf.contrib.layers.xavier_initializer(uniform=False))
    
    b = tf.get_variable(name = 'fc_biases',
                        shape = [fan_out],
                        initializer= tf.constant_initializer(0.0))
    return tf.matmul(x, w) + b
