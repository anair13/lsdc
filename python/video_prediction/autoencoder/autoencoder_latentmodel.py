# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12



def encoder(image_pair, state_pair, conf, reuse= False):
    """
    :param image_pair:
    :param state_pair: low dimensional input coordinates
    :return:
    """
    with slim.arg_scope([ slim.layers.conv2d, slim.layers.fully_connected,
             tf_layers.layer_norm, slim.layers.conv2d_transpose],
            reuse=reuse):

        image_pair = tf.reshape(image_pair, shape=[conf['batch_size'], 64,64,3*2])
        enc0 = slim.layers.conv2d(   #32x32x32
            image_pair,
            32, [5, 5],
            stride=2,
            scope='conv0',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm1'}
        )
        enc1 = slim.layers.conv2d(   #16x16x64
            enc0,
            64, [5, 5],
            stride=2,
            scope='conv1',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        enc2 = slim.layers.conv2d(   #8x8x128
            enc1,
            128, [5, 5],
            stride=2,
            scope='conv2',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        enc3 = slim.layers.conv2d(  # 8x8x64
            enc2,
            64, [5, 5],
            stride=1,
            scope='conv3',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        enc4 = slim.layers.conv2d(  # 8x8x16
            enc3,
            16, [5, 5],
            stride=1,
            scope='conv4',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        enc5 = slim.layers.conv2d(  # 8x8x1
            enc4,
            1, [5, 5],
            stride=1,
            scope='conv5',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        return enc5


def decoder(lt_state, conf, reuse= False):
    """
    :param image_pair:
    :param state_pair: lt dimensional input coordinates
    :return:
    """
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected,
                         tf_layers.layer_norm, slim.layers.conv2d_transpose],
                        reuse=reuse):

        dec0 = slim.layers.conv2d_transpose(   #8x8x16
            lt_state,
            16, [5, 5],
            stride=1,
            scope='conv0t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm1'}
        )
        dec1 = slim.layers.conv2d_transpose(   #8x8x64
            dec0,
            16, [5, 5],
            stride=1,
            scope='conv1t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        dec2 = slim.layers.conv2d_transpose(   #8x8x128
            dec1,
            64, [5, 5],
            stride=1,
            scope='conv2t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        dec3 = slim.layers.conv2d_transpose(  # 16x16x64
            dec2,
            64, [5, 5],
            stride=2,
            scope='conv3t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        dec4 = slim.layers.conv2d_transpose(  # 32x32x32
            dec3,
            32, [5, 5],
            stride=2,
            scope='conv4t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        dec5 = slim.layers.conv2d_transpose(  # 64x64x6
            dec4,
            6, [5, 5],
            stride=2,
            scope='conv5t',
            # normalizer_fn=tf_layers.batch_norm,
            # normalizer_params={'scope': 'batch_norm2'}
        )
        return dec5


def predictor(lt_dim_state01, actions_01, conf, reuse =False):
    with slim.arg_scope([slim.layers.fully_connected],reuse=reuse):
        with tf.variable_scope('latent_model'):
            actions_01 = tf.reshape(actions_01, shape = [conf['batch_size'], - 1])
            lt_dim_state_flat = tf.reshape(lt_dim_state01,shape = [conf['batch_size'], - 1])

            # predicting the next hidden state:
            if 'stopgrad' in conf:
                lt_dim_state_flat = tf.stop_gradient(lt_dim_state_flat)

            lt_state_enc0 = tf.concat(1, [actions_01, lt_dim_state_flat])

            lt_state_enc1 = slim.layers.fully_connected(
                lt_state_enc0,
                200,
                scope='hid_state_enc1')
            lt_state_enc2 = slim.layers.fully_connected(
                lt_state_enc1,
                200,
                scope='hid_state_enc2')
            hid_state_enc3 = slim.layers.fully_connected(
                lt_state_enc2,
                int(lt_dim_state_flat.get_shape()[1]),
                scope='hid_state_enc3',
                activation_fn=None)

            hid_state_enc3 = tf.reshape(hid_state_enc3, [conf['batch_size'], 8, 8, 1])

            return hid_state_enc3


def construct_model(conf,
                    images_01,
                    actions_01,
                    states_01 = None,
                    images_12 = None,
                    states_12 = None,
                    test = False,

                    ):
    """
    :param conf:
    :param images_01:
    :param actions_01:
    :param states_01:
    :param images_12:
    :param states_12:
    :param test:
    :param lt_state1: used when propagating latent state through latent model
    :return:
    """

    if test:
        inf_lt_state_01 = encoder(images_01, None, conf)   #8x8x1
        pred_lt_state12 = predictor(inf_lt_state_01, actions_01, conf)   #8x8x1
        images_23_rec = decoder(pred_lt_state12, conf)

        return images_23_rec, pred_lt_state12

    else:

        inf_lt_state_01 = encoder(images_01, states_01, conf)
        inf_lt_state12 = encoder(images_12, states_12, conf, reuse=True)   #8x8x1
        images_01_rec = decoder(inf_lt_state_01, conf)
        pred_lt_state12 = predictor(inf_lt_state_01, actions_01, conf)   # 8x8x1

        return pred_lt_state12, inf_lt_state12, images_01_rec