import read_tf_record
# todo: this import has to be above tf_utils???

import tf_utils
import numpy as np
import os
import subprocess
import collections
import copy
import tensorflow as tf
# from path import project_dir, tf_data_dir
import cv2

from path import project_dir, tf_data_dir

slim = tf.contrib.slim
# from nets import alexnet_conv, inception_v3_conv, vgg_conv, vgg_16, alexnet_geurzhoy

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

# def data_generator(conf, sess):
#     image_batch, action_batch, state_batch, object_pos_batch = read_tf_record.build_tfrecord_input(conf, training=True)
#     while True:
#         print "getting data"
#         image_data, action_data, state_data, object_pos = sess.run([image_batch, action_batch, state_batch, object_pos_batch])
#         print "got data"
#         yield image_data, action_data, state_data, object_pos

def conv_network(img, reuse=False):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse):
        net = slim.conv2d(img, 32, [3, 3], scope='conv1', )
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 32, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, 32, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net, 32, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.conv2d(net, 32, [4, 4], padding='VALID', scope='conv5')
        net = tf.reshape(net, [-1, 32])
    return net

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 0.01))

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def make_network(x, network_size):
    """Makes fully connected network with input x and given layer sizes.
    Assume len(network_size) >= 2
    """
    input_size = network_size[0]
    output_size = network_size.pop()
    a = input_size
    cur = x
    i = 0
    for a, b in zip(network_size, network_size[1:]):
        W = init_weights("W" + str(i), [a, b])
        B = init_weights("B" + str(i), [1, b])
        cur = tf.nn.elu(tf.matmul(cur, W) + B)
        i += 1
    W = init_weights("W" + str(i), [b, output_size])
    B = init_weights("B" + str(i), [1, output_size])
    prediction = tf.matmul(cur, W) + B
    return prediction

def dict_to_string(params):
    name = ""
    for key in params:
        if params[key] is not None:
            name = name + str(key) + "_" + str(params[key]) + "_"
    return name[:-1]

def pred_network(f1, f2, reuse=False):
    # with slim.arg_scope([slim.fully_connected],
    #     weights_initializer=tf.contrib.layers.xavier_initializer,
    #     weights_regularizer=slim.l2_regularizer(0.0005),
    #     activation_fn=tf.nn.elu, reuse=reuse):
    #     x = tf.concat(1, [f1, f2])
    #     print x
    #     print x.get_shape()
    #     net = slim.fully_connected(x, 32, scope='fc_1')
    #     net = slim.fully_connected(net, 32, scope='fc_2')
    #     # net = slim.fully_connected(net, 2, activation_fn=None, scope='fc_3')
    # return net

    with tf.variable_scope("fc", reuse=reuse) as sc:
        x = tf.concat(1, [f1, f2])
        a = make_network(x, [64, 32, 2])
        return a

def autoencoder(current_input, n_filters=[8, 16, 16, 8], filter_sizes=[3, 3, 3, 3]):
    X = current_input
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(tf.truncated_normal([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32), trainable=True, name="biases")
        encoder.append(W)
        output = lrelu(tf.nn.bias_add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # Store latent representation
    Y = current_input
    print Y.get_shape()
    encoder.reverse()
    shapes.reverse()

    # Building decoder using same weights
    for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.constant(0.0, shape=[W.get_shape().as_list()[2]]), trainable=True, name="biases")
            output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W, tf.pack([tf.shape(self.X)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

    # Store reconstruction
    Z = current_input
    loss = tf.nn.l2_loss(Z - X)
    return Y, Z, loss

class AutoencoderModel(object):
    def __init__(self, train_conf, test_conf = {}):
        print "setting up network"
        self.name = dict_to_string(train_conf)
        self.network = tf_utils.TFNet(self.name,
            logDir= tf_data_dir + 'tf_logs/',
            modelDir= tf_data_dir + 'tf_models/',
            outputDir= tf_data_dir + 'tf_outputs/',)
            # eraseModels=erase_model)

        self.conf = train_conf

        # image_batch  = tf.placeholder("float", [None, 15, 64, 64, 3])
        # action_batch = tf.placeholder("float", [None, 15, 2])
        # self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))

        self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))
        image_batch, action_batch, state_batch, object_pos_batch = self.inputs

        img_features = []
        auto_losses = []
        for i in range(self.conf['sequence_length']):
            Y, Z, l  = autoencoder(image_batch[:, i, :, :, :], i != 0)
            img_features.append(Y)
            auto_losses.append(l)
            if i == 0:
                print "image features: (batch, featsize)", f.get_shape()

        self.loss = tf.add_n(auto_losses)
        self.network.add_to_losses(self.loss)

        # make a training network
        self.train_network = tf_utils.TFTrain(self.inputs, self.network, batchSz=self.conf['batch_size'], initLr=self.conf['initLr'])
        self.train_network.add_loss_summaries([self.loss], ['loss'])

        print "done with network setup"

    def init_sess(self):
        self.sess = tf.InteractiveSession() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
        tf.train.start_queue_runners(self.sess)
        self.sess.run(tf.initialize_all_variables())

    def train_batch(self, inputs, batch_size, isTrain):
        # image_batch, action_batch, state_batch, object_pos_batch = inputs
        # image_data, action_data, state_data, object_pos = self.sess.run([image_batch, action_batch, state_batch, object_pos_batch])
        # feed_dict = {image_batch: image_data, action_batch: action_data}
        # return feed_dict
        return {}

    def train(self, max_iters = 100000, use_existing = True):
        self.init_sess()
        self.train_network.maxIter_ = max_iters
        self.train_network.dispIter_ = 100
        self.train_network.train(self.train_batch, self.train_batch, use_existing=use_existing, sess=self.sess)

