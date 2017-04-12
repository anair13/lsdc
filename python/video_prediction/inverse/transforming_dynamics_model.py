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

import discretize

# def data_generator(conf, sess):
#     image_batch, action_batch, state_batch, object_pos_batch = read_tf_record.build_tfrecord_input(conf, training=True)
#     while True:
#         print "getting data"
#         image_data, action_data, state_data, object_pos = sess.run([image_batch, action_batch, state_batch, object_pos_batch])
#         print "got data"
#         yield image_data, action_data, state_data, object_pos

def conv_network(img, reuse=False):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                      # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse):
        net = slim.conv2d(img, 32, [6, 6], 2, padding='SAME', scope='conv1')
        net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv2')
        net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv3')
        net = slim.conv2d(net, 32, [3, 3], 2, padding='SAME', scope='conv4', activation_fn=None)
        net = tf.reshape(net, [-1, 512])
    return net

def transforming_conv_network(img, reuse=False):
    with tf.variable_scope('transformer', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d], padding='SAME',
                          # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse):
            net = slim.conv2d(img, 32, [6, 6], 2, padding='SAME', scope='conv1')
            net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv2')
            net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv3')
            net = slim.conv2d(net, 32, [3, 3], 2, padding='SAME', scope='conv4')
            net = slim.conv2d_transpose(net, 32, [3, 3], 2, padding='SAME', scope='conv5')
            net = slim.conv2d_transpose(net, 32, [6, 6], 2, padding='SAME', scope='conv6')
            net = slim.conv2d_transpose(net, 32, [6, 6], 2, padding='SAME', scope='conv7')
            net = slim.conv2d_transpose(net, 3, [6, 6], 2, padding='SAME', scope='conv8', activation_fn=None)
            net = tf.sigmoid(net)
            print "t conv shape", net.get_shape()
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
    excludes = ['data_dir']
    print params
    name = ""
    for key in params:
        if key in excludes:
            continue
        if params[key] is not None:
            name = name + str(key) + "_" + str(params[key]) + "/"
    return name[:-1]

def pred_network(f1, f2, reuse=False, N=20):
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
        a = make_network(x, [1024, 100, 2*N])
        return tf.reshape(a, [-1, 2, N])

def discretize_actions(x, N=20):
    batches = x.shape[0]
    frames = x.shape[1]
    actiondim = x.shape[2]
    X = np.zeros((batches, frames, actiondim, N))
    for b in range(batches):
        for f in range(frames):
            for a in range(actiondim):
                X[b, f, a, :] = discretize.one_hot_encode(x[b, f, a], -10, 10, N)
    return np.float32(X)

class DynamicsModel(object):
    """An inverse model I with a adversarial transformer T that tries to hide information
    from the inverse model in order to force the model to pay attention to more
    """
    def __init__(self, train_conf, test_conf = {}):
        print "setting up network"
        self.name = dict_to_string(train_conf)
        self.network = tf_utils.TFNet(self.name,
            logDir= tf_data_dir + 'tf_logs/',
            modelDir= tf_data_dir + 'tf_models/',
            outputDir= tf_data_dir + 'tf_outputs/',)
            # eraseModels=erase_model)

        self.conf = train_conf
        self.sess = None

        # image_batch  = tf.placeholder("float", [None, 15, 64, 64, 3])
        # action_batch = tf.placeholder("float", [None, 15, 2])
        # self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))

        self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))
        image_batch, raw_action_batch, state_batch = self.inputs

        D = lambda x: discretize_actions(x, self.conf['discretize'])
        action_batch = tf.py_func(D, [raw_action_batch], tf.float32)

        self.t_masks = []
        self.img_features = []
        for i in range(self.conf['sequence_length']):
            m = transforming_conv_network(image_batch[:, i, :, :, :], i != 0)
            t_i = image_batch[:, i, :, :, :]
            if self.conf['masks']:
                t_i = m * t_i
            f = conv_network(t_i, i != 0)
            self.t_masks.append(m)
            self.img_features.append(f)
            if i == 0:
                print "mask size: ", m.get_shape()
                print "image features: (batch, featsize)", f.get_shape()

        self.action_preds = []
        self.correct_predictions = []
        action_loss = []
        for i in range(self.conf['sequence_length'] - 1):
            a = pred_network(self.img_features[i], self.img_features[i+1], i != 0, self.conf['discretize'])
            A = action_batch[:, i, :, :]
            l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a, A))
            c = tf.equal(tf.argmax(a, 2), tf.argmax(A, 2))
            self.correct_predictions.append(tf.cast(c, tf.float32))
            self.action_preds.append(a)
            action_loss.append(l)

        self.accuracy = tf.reduce_mean(tf.concat(1, self.correct_predictions))

        self.loss = tf.add_n(action_loss)
        self.network.add_to_losses(self.loss)

        # make a training network
        self.train_network = tf_utils.TFTrain(self.inputs, self.network, batchSz=self.conf['batch_size'], initLr=self.conf['learning_rate'])
        self.train_network.add_loss_summaries([self.loss, self.accuracy], ['loss', 'accuracy'])

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
        self.train_network.saveIter_ = 1000
        self.train_network.train(self.train_batch, self.train_batch, use_existing=use_existing, sess=self.sess)

    def run(self, dataset, batches=1, i = None, sess=None):
        """Return batches*batch_size examples from the dataset ("train" or "val")
        i: specific model to restore, or restores the latest
        """
        f = self.get_f(i, sess)
        if not f:
            return None

        ret = []

        for j in range(batches):
            tb = self.train_batch(self.inputs, self.batch_size, dataset=="train")
            inps, out = f(tb)
            ret.append([inps, out])

        return ret

    def get_f(self, i = None, sess=None):
        """Return the network forward function"""
        ret = []
        if not self.sess:
            self.init_sess()
            self.sess.run(tf.initialize_all_variables())
            restore = self.network.restore_model(self.sess, i)
            if i and not restore: # model requested but not found
                return None

        names = ["pred_f0", "image", "action", "state"]
        def f():
            feed_dict = {}
            result = self.sess.run(self.action_preds[0:1] + self.inputs, feed_dict)
            # inps = [feed_dict[x] for x in self.inputs]
            out = {}
            for i, name in enumerate(names):
                out[name] = result[i]
            return out

        return f


