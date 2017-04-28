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

# import hack
import sys
import os
HERE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(HERE_DIR+"/..")
from utils_vpred import create_gif

def init_weights(name, shape):
    return tf.get_variable(name, shape=shape, initializer=tf.random_normal_initializer(0, 0.01))

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

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

        image_batch  = tf.placeholder("float", [None, 15, 64, 64, 3])
        raw_action_batch = tf.placeholder("float", [None, 15, 2])
        self.inputs = [image_batch, raw_action_batch]

        train_conf = self.conf.copy()
        train_conf["data_dir"] += '/train'
        self.train_input_readers = list(read_tf_record.build_tfrecord_input(train_conf, training=True))
        test_conf = self.conf.copy()
        test_conf["data_dir"] += '/test'
        self.test_input_readers = list(read_tf_record.build_tfrecord_input(test_conf, training=True))

        # self.inputs = list(read_tf_record.build_tfrecord_input(self.conf, training=True))
        # image_batch, raw_action_batch, state_batch = self.inputs

        D = lambda x: discretize_actions(x, self.conf['discretize'])
        action_batch = tf.py_func(D, [raw_action_batch], tf.float32)
        self.action_batch = action_batch

        self.fsize = self.conf.get('fsize', 100)
        self.dsize = self.conf.get('discretize', 20)
        self.batch_size = self.conf['batch_size']
        self.sequence_length = self.conf['sequence_length']
        self.context_frames = self.conf['context_frames']

        self.feat_activation = tf.sigmoid # bad default because of previous models
        if self.conf.get('featactivation') == "none":
            self.feat_activation = None 

        transformed_image_batch = image_batch
        if self.conf.get('transform') == "meansub":
            transformed_image_batch = transformed_image_batch - self.get_image_mean_tensor()

        self.t_masks = []
        self.img_features = []
        self.img_reconstructions = []
        self.img_reconstruction_losses = [tf.constant(0.0)]
        for i in range(self.conf['sequence_length']):
            t_i = transformed_image_batch[:, i, :, :, :]
            if self.conf['masks']:
                m = self.transforming_conv_network(transformed_image_batch[:, i, :, :, :], i != 0)
                t_i = m * t_i
                self.t_masks.append(m)
                if i == 0:
                    print "mask size: ", m.get_shape()

            f = self.conv_network(t_i, i != 0)
            self.img_features.append(f)
            if i == 0:
                print "image features: (batch, featsize)", f.get_shape()

            autoencoder = self.conf.get('autoencoder', None)
            if autoencoder == "decode": # means no gradients passed back
                f = tf.stop_gradient(f)
            if autoencoder:
                I = self.decoder_network(f, i != 0)
                l = tf.nn.l2_loss(I - t_i)
                self.img_reconstructions.append(I)
                self.img_reconstruction_losses.append(l)

        self.action_preds = []
        self.action_loss = []
        self.correct_predictions = []
        self.forward_predictions = []
        self.forward_losses = []
        feats = [self.img_features[0] for _ in range(self.context_frames)]
        for i in range(self.conf['sequence_length'] - 1):
            f1 = self.img_features[i]
            f2 = self.img_features[i+1]

            feats.pop(0)
            feats.append(f1)
            u = tf.reshape(action_batch[:, i, :, :], [-1, 2, self.dsize])

            a = self.action_pred_network(feats, i != 0)
            l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a, u))
            c = tf.equal(tf.argmax(a, 2), tf.argmax(u, 2))
            self.correct_predictions.append(tf.cast(c, tf.float32))
            self.action_preds.append(a)
            self.action_loss.append(l)

            if self.conf.get('forwardloss') == "gaussian":
                mu, sigma = self.forward_gaussian_pred_network(feats, u, i != 0)
                x = (f2 - mu) / sigma
                l = tf.reduce_mean(0.5 * x * x + tf.log(sigma))
                self.forward_predictions.append(mu)
                self.forward_predictions.append(sigma)
            else:        
                f = self.forward_pred_network(feats, u, i != 0)
                l = tf.reduce_mean(tf.nn.l2_loss(f2 - f))
                self.forward_predictions.append(f)
            self.forward_losses.append(l)

        # compute accuracy and losses below

        self.accuracy = tf.reduce_mean(tf.concat(1, self.correct_predictions))
        self.feat_norm = tf.reduce_mean([tf.abs(f) for f in self.img_features])

        t_vars = tf.trainable_variables()
        self.dynamics_vars = [var for var in t_vars if not 'transformer' in var.name]
        self.transformer_vars = [var for var in t_vars if 'transformer' in var.name]

        seq = self.conf.get('seq', None) if self.conf['seq'] is not None else 0
        mu1 = tf.constant(float(self.conf['mu1'])) if self.conf.get('mu1', None) is not None else tf.constant(0.0)
        mu2 = tf.constant(float(self.conf['mu2'])) if self.conf.get('mu2', None) is not None else tf.constant(0.0)
        mu3 = tf.constant(float(self.conf['mu3'])) if self.conf.get('mu3', None) is not None else tf.constant(0.0)
        self.forward_loss = tf.add_n(self.forward_losses)
        self.inverse_loss = tf.add_n(self.action_loss)
        self.reconstruction_loss = tf.add_n(self.img_reconstruction_losses)
        self.dynamics_loss = self.inverse_loss + mu2 * self.forward_loss + mu3 * self.reconstruction_loss
        self.transformer_loss = -self.inverse_loss + mu1 * tf.reduce_mean(self.t_masks)

        feats = [self.img_features[0] for _ in range(self.context_frames)]
        self.rollout_outputs, self.rollout_reconstructions = self.rollout_network(feats, self.action_batch, imgs=True)

        if seq % 2 == 1:
            loss = self.transformer_loss
            var_list = self.transformer_vars
        else:
            loss = self.dynamics_loss
            var_list = self.dynamics_vars

        self.network.add_to_losses(loss)
        self.train_network = tf_utils.TFTrain(self.inputs, self.network, batchSz=self.conf['batch_size'], initLr=self.conf['learning_rate'], var_list=var_list)
        # self.train_network.add_loss_summaries([self.dynamics_loss, self.inverse_loss, self.forward_loss, self.transformer_loss, self.accuracy], ['dynamics_loss', 'inverse_loss', 'forward_loss', 'transformer_loss', 'accuracy'])
        self.train_network.add_loss_summaries([self.dynamics_loss, self.inverse_loss, self.forward_loss, self.reconstruction_loss, self.accuracy, self.feat_norm], ['dynamics_loss', 'inverse_loss', 'forward_loss', 'reconstruction_loss', 'accuracy', 'feat_norm'])

        print "done with network setup"

    def init_sess(self):
        self.sess = tf.InteractiveSession() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
        tf.train.start_queue_runners(self.sess)
        self.sess.run(tf.initialize_all_variables())

    def train_batch(self, inputs, batch_size, isTrain):
        readers = self.train_input_readers if isTrain else self.test_input_readers
        image_batch, raw_action_batch, state_batch = readers

        image_data, action_data, state_data = self.sess.run([image_batch, raw_action_batch, state_batch])
        image_batch, action_batch = inputs
        feed_dict = {image_batch: image_data, action_batch: action_data}
        return feed_dict

    def train(self, max_iters = 100000, use_existing = True, init_conf=None):
        if init_conf:
            model_n, init_conf = init_conf
            init_name = tf_data_dir + 'tf_models/' + dict_to_string(init_conf) + "/model-" + str(model_n)
        else:
            init_name = None

        self.init_sess()
        self.train_network.maxIter_ = max_iters
        self.train_network.dispIter_ = 100
        self.train_network.saveIter_ = 1000
        self.train_network.train(self.train_batch, self.train_batch, use_existing=use_existing, sess=self.sess, init_path=init_name)

        self.save_rollout_gifs()

    def train_transformer(self):
        """trash"""
        t_iter_  = tf.Variable(0, name='t_iteration')
        apply_grad_op = tf.train.AdamOptimizer(self.conf['learning_rate']).minimize(self.transformer_loss, var_list=self.transformer_vars, global_step=t_iter)

        for i in range(1000):
            sess.run([apply_grad_op])

    def run(self, dataset="test", batches=1, i = None, sess=None, f=None):
        """Return batches*batch_size examples from the dataset ("train" or "val")
        i: specific model to restore, or restores the latest
        """
        if not f:
            f = self.get_f(i, sess)
            if not f:
                return None

        ret = []

        for j in range(batches):
            tb = self.train_batch(self.inputs, self.batch_size, dataset=="train")
            out = f(tb)
            ret.append(out)

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

        query = self.action_preds + self.inputs + self.t_masks + self.img_reconstructions + self.forward_predictions + self.img_features
        def f(feed_dict):
            result = self.sess.run(query, feed_dict)
            d = collections.OrderedDict()
            for q, r in zip(query, result):
                d[q] = r
            return d

        return f

    def get_rollout_f(self, i = None, sess=None, imgs=False):
        """Return the network forward function"""
        ret = []
        if not self.sess:
            self.init_sess()
            self.sess.run(tf.initialize_all_variables())
            restore = self.network.restore_model(self.sess, i)
            if i and not restore: # model requested but not found
                return None

        query = self.inputs + self.rollout_outputs
        print len(query)
        if imgs:
            query += self.rollout_reconstructions
        print len(query)
        def f(feed_dict):
            result = self.sess.run(query, feed_dict)
            d = collections.OrderedDict()
            for q, r in zip(query, result):
                d[q] = r
            return d

        return f

    def save_rollout_gifs(self):
        f = self.get_rollout_f(imgs=True)
        result = self.run('test', f=f)[0]
        reconstructions = result.values()[16:]
        image_data, action_data = result.values()[:2]

        folder = self.network.outputDir_

        mean = np.zeros((64, 64, 3))
        if self.conf['transform'] == "meansub":
            mean = self.get_image_mean_array()

        for b in range(32):
            ims = []
            for i in range(14):
                pred_im = (reconstructions[i][b, :, :, :] + mean) * 256
                real_im = image_data[b, i+1, :, :, :] * 256
                im = np.concatenate([pred_im, real_im], 1)
                ims.append(im)
            create_gif.npy_to_gif(ims, folder + '/' + str(b))


    ##### NETWORK CONSTRUCTION FUNCTIONS


    def get_image_mean_array(self):
        img_mean = np.load(self.conf['data_dir'] + '/train/mean.npy')
        return img_mean

    def get_image_mean_tensor(self):
        img_mean = self.get_image_mean_array()
        tiled_img = np.tile(img_mean, [self.batch_size, self.sequence_length, 1, 1, 1])
        return tf.constant(tiled_img)

    def conv_network(self, img, reuse=False):
        with tf.variable_scope('conv', reuse=reuse) as scope:
            with slim.arg_scope([slim.conv2d], padding='SAME',
                              # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse):
                net = slim.conv2d(img, 32, [6, 6], 2, padding='SAME', scope='conv1')
                net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv2')
                net = slim.conv2d(net, 32, [6, 6], 2, padding='SAME', scope='conv3')
                net = slim.conv2d(net, 32, [3, 3], 2, padding='SAME', scope='conv4')
                net = tf.reshape(net, [-1, 512])
                net = slim.fully_connected(net, self.fsize, scope='fc5', activation_fn=self.feat_activation)
        return net

    def decoder_network(self, f, reuse=False):
        with tf.variable_scope('autodecoder', reuse=reuse) as scope:
            with slim.arg_scope([slim.conv2d], padding='SAME',
                              # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              weights_regularizer=slim.l2_regularizer(0.0005), reuse=reuse):
                net = slim.fully_connected(f, 512, scope='fc5')
                net = tf.reshape(net, [-1, 4, 4, 32])
                net = slim.conv2d_transpose(net, 32, [6, 6], 2, padding='SAME', scope='conv1')
                net = slim.conv2d_transpose(net, 32, [6, 6], 2, padding='SAME', scope='conv2')
                net = slim.conv2d_transpose(net, 32, [6, 6], 2, padding='SAME', scope='conv3')
                net = slim.conv2d_transpose(net, 3, [3, 3], 2, padding='SAME', scope='conv4', activation_fn=None)
        return net

    def transforming_conv_network(self, img, reuse=False):
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
                net = slim.conv2d_transpose(net, 3, [6, 6], 2, padding='SAME', scope='conv8', activation_fn=tf.sigmoid)
                net = tf.sigmoid(net)
        return net

    def action_pred_network(self, fs, reuse=False):
        """N is the discretization bins"""
        with tf.variable_scope('actionpred', reuse=reuse) as sc:
            with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=tf.nn.elu, reuse=reuse):
                net = tf.concat(1, fs)
                net = slim.fully_connected(net, 100, scope='fc_1')
                net = slim.fully_connected(net, self.dsize * 2, scope='fc_2', activation_fn=None)
                return tf.reshape(net, [-1, 2, self.dsize])

    def forward_pred_network(self, fs, u, reuse=False):
        """
        fcsize is the size of f1 and f2"""
        u = tf.reshape(u, [-1, 2*self.dsize])
        with tf.variable_scope('forwardpred', reuse=reuse) as sc:
            with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=tf.nn.elu, reuse=reuse):
                # print f1.get_shape()
                # print u.get_shape()
                # print reuse
                # U = slim.fully_connected(u, 100)
                net = tf.concat(1, fs + [u])
                net = slim.fully_connected(net, 100, scope='fc_1')
                net = slim.fully_connected(net, 100, scope='fc_2')
                net = slim.fully_connected(net, self.fsize, scope='fc_3', activation_fn=self.feat_activation)
                return net

    def forward_gaussian_pred_network(self, fs, u, reuse=False):
        """
        fcsize is the size of f1 and f2"""
        u = tf.reshape(u, [-1, 2*self.dsize])
        with tf.variable_scope('forwardpred', reuse=reuse) as sc:
            with slim.arg_scope([slim.fully_connected],
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=tf.nn.elu, reuse=reuse):
                # print f1.get_shape()
                # print u.get_shape()
                # print reuse
                # U = slim.fully_connected(u, 100)
                net = tf.concat(1, fs + [u])
                net = slim.fully_connected(net, 100, scope='fc_1')
                net = slim.fully_connected(net, 100, scope='fc_2')
                mu = slim.fully_connected(net, self.fsize, scope='fc_3', activation_fn=self.feat_activation)
                sigma = slim.fully_connected(net, self.fsize, scope='fc_3_sigma', activation_fn=tf.nn.relu) + 0.1
                return mu, sigma

    def rollout_network(self, fs, actions, reuse=True, imgs=False):
        u = tf.reshape(actions, [-1, self.sequence_length, 2*self.dsize])
        outputs = []
        reconstructions = []
        for i in range(self.sequence_length - 1):
            with tf.variable_scope('forwardpred', reuse=reuse) as sc:
                with slim.arg_scope([slim.fully_connected],
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    activation_fn=tf.nn.elu, reuse=reuse):
                    # print f1.get_shape()
                    # print u.get_shape()
                    # print reuse
                    # U = slim.fully_connected(u, 100)
                    net = tf.concat(1, fs + [u[:, i, :]])
                    net = slim.fully_connected(net, 100, scope='fc_1')
                    net = slim.fully_connected(net, 100, scope='fc_2')
                    f = slim.fully_connected(net, self.fsize, scope='fc_3', activation_fn=self.feat_activation)
                    outputs.append(f)

                    fs.pop(0)
                    fs.append(f)
        if imgs:
            for i in range(self.sequence_length - 1):
                f = outputs[i]
                reconstructions.append(self.decoder_network(f, True))
        return outputs, reconstructions
