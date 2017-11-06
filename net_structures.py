# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:38:07 2017

@author: Charles
"""
import tensorflow as tf
def prelu(x):
    with tf.name_scope('prelu'):
        alphas = tf.get_variable('alpha',
                                 x.get_shape()[-1], 
                                 initializer = tf.constant_initializer(0.1),
                                 dtype = tf.float32
                                 )
        pos = tf.nn.relu(x)
        neg = alphas * (x-tf.abs(x)) * 0.5
        return pos + neg
def net_1(inputs, keep_prob = None):
    conv1_1 = tf.contrib.layers.conv2d(inputs = inputs, num_outputs = 32, kernel_size = [5,5], activation_fn = prelu, scope = 'conv1_1')
    conv1_2 = tf.contrib.layers.conv2d(inputs = conv1_1,num_outputs = 32,kernel_size = [5,5],activation_fn = prelu,scope = 'conv1_2')
    pool1 = tf.contrib.layers.max_pool2d(inputs = conv1_2,kernel_size = 2, scope = 'pool1')
    conv2_1 = tf.contrib.layers.conv2d(inputs = pool1,num_outputs = 64,kernel_size = [5,5],activation_fn = prelu,scope = 'conv2_1' )           
    conv2_2 = tf.contrib.layers.conv2d(inputs = conv2_1,num_outputs = 64,kernel_size = [5,5],activation_fn = prelu,scope = 'conv2_2')       
    pool2 = tf.contrib.layers.max_pool2d(inputs = conv2_2,kernel_size = 2,scope = 'pool2')
    conv3_1 = tf.contrib.layers.conv2d(inputs = pool2,num_outputs = 128,kernel_size = [5,5],activation_fn = prelu,scope = 'conv3_1') 
    conv3_2 = tf.contrib.layers.conv2d(inputs = conv3_1,num_outputs = 128,kernel_size = [5,5],activation_fn = prelu,scope = 'conv3_2')
    conv4 = tf.contrib.layers.max_pool2d(inputs = conv3_2,kernel_size = 2,scope = 'pool3')
#    conv4 = tf.contrib.layers.conv2d(inputs = conv3_2,
#                                             num_outputs = 64,
#                                             kernel_size = [3,3],
#                                             activation_fn = prelu,
#                                             scope = 'conv4',
#                                             stride = 2,
#                                             padding = 'VALID'
#                                             )            
    flat = tf.contrib.layers.flatten(inputs = conv4, scope = 'flatten')
    fully4 = tf.contrib.layers.fully_connected(inputs = flat,num_outputs = 2,activation_fn = prelu,scope = 'fully4')
    hidden = tf.contrib.layers.fully_connected(inputs = fully4,num_outputs = 100,activation_fn = prelu,scope = 'hidden')
    if keep_prob is not None:
        hidden = tf.nn.dropout(hidden, keep_prob = keep_prob)
    fully5 = tf.contrib.layers.fully_connected(inputs = hidden,num_outputs = 10,activation_fn = None,scope = 'fully5')
                                                   #最后logits不用激活函数
    return fully4, fully5
def net_2(inputs, keep_prob = None):
    conv1_1 = tf.contrib.layers.conv2d(inputs = inputs, num_outputs = 32, kernel_size = [3,3], activation_fn = prelu, scope = 'conv1_1')
    conv1_2 = tf.contrib.layers.conv2d(inputs = conv1_1,num_outputs = 32,kernel_size = [3,3],activation_fn = prelu,scope = 'conv1_2')
    pool1 = tf.contrib.layers.max_pool2d(inputs = conv1_2,kernel_size = 2, scope = 'pool1')
    conv2_1 = tf.contrib.layers.conv2d(inputs = pool1,num_outputs = 64,kernel_size = [3,3],activation_fn = prelu,scope = 'conv2_1' )           
    conv2_2 = tf.contrib.layers.conv2d(inputs = conv2_1,num_outputs = 64,kernel_size = [3,3],activation_fn = prelu,scope = 'conv2_2')       
    pool2 = tf.contrib.layers.max_pool2d(inputs = conv2_2,kernel_size = 2,scope = 'pool2')
    conv3_1 = tf.contrib.layers.conv2d(inputs = pool2,num_outputs = 128,kernel_size = [3,3],activation_fn = prelu,scope = 'conv3_1') 
    conv3_2 = tf.contrib.layers.conv2d(inputs = conv3_1,num_outputs = 128,kernel_size = [3,3],activation_fn = prelu,scope = 'conv3_2')
    conv4 = tf.contrib.layers.max_pool2d(inputs = conv3_2,kernel_size = 2,scope = 'pool3')
#    conv4 = tf.contrib.layers.conv2d(inputs = conv3_2,
#                                             num_outputs = 64,
#                                             kernel_size = [3,3],
#                                             activation_fn = prelu,
#                                             scope = 'conv4',
#                                             stride = 2,
#                                             padding = 'VALID'
#                                             )
        
    flat = tf.contrib.layers.flatten(inputs = conv4, scope = 'flatten')
    #flat = tf.contrib.layers.fully_connected(inputs = flat,num_outputs = 100,activation_fn = prelu,scope = 'fully4')
    fully4 = tf.contrib.layers.fully_connected(inputs = flat,num_outputs = 2,activation_fn = prelu)
    #hidden = tf.contrib.layers.fully_connected(inputs = fully4,num_outputs = 100,activation_fn = prelu,scope = 'hidden')
    #if keep_prob is not None:
    #    hidden = tf.nn.dropout(hidden, keep_prob = keep_prob)
    #hidden = tf.contrib.layers.fully_connected(inputs = hidden,num_outputs = 100,activation_fn = prelu)
    fully5 = tf.contrib.layers.fully_connected(inputs = fully4,num_outputs = 10,activation_fn = None,scope = 'fully5')
                                                   #最后logits不用激活函数
    return fully4, fully5