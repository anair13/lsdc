import tensorflow as tf
import tf_utils as tfu
import tensorflow as tf
import tf_utils as tfu
import numpy as np
import read_datasets as rd
import matplotlib.pyplot as plt
import vis_utils as vu
from data import rope_data, figrim_data

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class Autoencoder:
    '''
    A Convolutional autoencoder for rope data
    ------------------------------------
    X : Input placeholder to network
    Y : Latent representation
    Z : Output reconstruction
    '''
    def __init__(self, input_shape=[None, 200, 200, 3], n_filters=[8, 16, 16, 8], filter_sizes=[3, 3, 3, 3]):

    self.name = "rope_autoencoder"

    # Init input placeholder
    self.X = tf.placeholder(tf.float32, input_shape)

    encoder = []
    shapes = []
    # Building encoder
    current_input = self.X
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(tf.truncated_normal([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output], stddev=0.1), name="weights")
        b = tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32), trainable=True, name="biases")
        encoder.append(W)
        output = lrelu(tf.nn.bias_add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output



    # Store latent representation
    self.Y = current_input
    print self.Y.get_shape()
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
    self.Z = current_input
    loss = tf.nn.l2_loss(self.Z - self.X)

    # Util settings
    self.network = tfu.TFNet(self.name)
    self.network.add_to_losses(loss)
    self.train_network = tfu.TFTrain([self.X], self.network, batchSz=100)
    self.train_network.add_loss_summaries(loss)
    self.batch_loader = rope_data


    print "Network setup"

    def train_batch(self, inputs, batch_size, isTrain):
    if isTrain:
        X1, X2, Y1, Y2, Y3, S1, S2 = self.batch_loader.get_batch("train", batch_size)
    else:
        X1, X2, Y1, Y2, Y3, S1, S2 = self.batch_loader.get_batch("val", batch_size)

    feed_dict = {inputs[0]: X1}
    return feed_dict

    def run(self, dataset, batch_size):
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        self.network.restore_model(sess)

        # feed_dict = self.train_batch([self.X], batch_size, dataset)
            feed_dict = {self.X: figrim_data.get_batch(batch_size)}
            prediction, target = sess.run([self.Z, self.X], feed_dict)
        return prediction, target



    def train(self, max_iters=10000):
    print "Training network", self.name
    self.train_network.maxIter_ = 10000
    self.train_network.dispIter_ = 100
    self.train_network.train(self.train_batch, self.train_batch, gpu_fraction=0.95, use_existing=True)


if __name__ == '__main__':
    print "Start training autoencoder"
    model = Autoencoder()
    model.train(10000)
