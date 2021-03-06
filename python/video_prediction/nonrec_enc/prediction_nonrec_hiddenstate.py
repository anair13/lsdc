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
import pdb

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    context_frames=2,
                    conf = None):
    """Build convolutional lstm video predictor using STP, CDNA, or DNA.

    Args:
      images: tensor of ground truth image sequences
      actions: tensor of action sequences
      states: tensor of ground truth state sequences
      iter_num: tensor of the current training iteration (for sched. sampling)
      k: constant used for scheduled sampling. -1 to feed in own prediction.
      use_state: True to include state and action in prediction
      num_masks: the number of different pixel motion predictions (and
                 the number of masks for each of those predictions)
      stp: True to use Spatial Transformer Predictor (STP)
      cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
      dna: True to use Dynamic Neural Advection (DNA)
      context_frames: number of ground truth frames to pass in before
                      feeding in own predictions
      pix_distrib: the initial one-hot distriubtion for designated pixels
    Returns:
      gen_images: predicted future image frames
      gen_states: predicted future states

    Raises:
      ValueError: if more than one network option specified or more than 1 mask
      specified for DNA model.
    """

    if 'dna_size' in conf.keys():
        DNA_KERN_SIZE = conf['dna_size']
    else:
        DNA_KERN_SIZE = 5

    print 'constructing network with hidden state...'

    batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
    batch_size = int(batch_size)

    # Generated robot states and images.
    gen_states, gen_images, gen_masks, inf_low_state_list, pred_low_state_list = [], [], [], [], []
    current_state = states[0]

    if k == -1:
        feedself = True
    else:
        # Scheduled sampling:
        # Calculate number of ground-truth frames to pass in.
        num_ground_truth = tf.to_int32(
            tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
        feedself = False


    for t in range(1, len(images) -1):


        # Reuse variables after the first timestep.
        reuse = bool(gen_images)

        done_warm_start = len(gen_images) > context_frames - 1
        with slim.arg_scope(
                [slim.layers.conv2d, slim.layers.fully_connected,
                 tf_layers.layer_norm, slim.layers.conv2d_transpose],
                reuse=reuse):

            if feedself and done_warm_start:
                # Feed in generated image.
                prev_image = gen_images[-1]
                prev2_image = gen_images[-2]
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(images[t], gen_images[-1], batch_size,
                                              num_ground_truth)
                prev_image2 = scheduled_sample(images[t-1], gen_images[-2], batch_size,
                                              num_ground_truth)
            else:
                # Always feed in ground_truth
                prev_image = images[t]
                prev_image2 = images[t-1]

            action = actions[t]


            if (not 'prop_latent' in conf) or t < 2:   # encode!
                print 'encode {}'.format(t)

                # Predicted state is always fed back in
                state_action = tf.concat(1, [action, current_state])   # 6x

                comb_img = tf.concat(3, [prev_image, prev_image2])  # 64x64x6

                enc0 = slim.layers.conv2d(  #32x32x32
                    comb_img, 32, [5, 5], stride=2, scope='conv1')

                enc1 = slim.layers.conv2d(  #16x16x32
                    enc0, 32, [3, 3], stride=2, scope='conv2')

                enc2 = slim.layers.conv2d(  #16x16x64
                    enc1, 64, [3, 3], stride=1, scope='conv3')

                enc3 = slim.layers.conv2d(    #8x8x64
                    enc2, 64, [3, 3], stride=2, scope='conv4')

                enc4 = slim.layers.conv2d(  # 8x8x64
                    enc3, 64, [3, 3], stride=1, scope='conv5')

                enc5 = slim.layers.conv2d(  # 8x8x32
                    enc4, 32, [3, 3], stride=1, scope='conv6')

                # Pass in state and action.
                smear = tf.reshape(
                    state_action,
                    [batch_size, 1, 1, int(state_action.get_shape()[1])])
                smear = tf.tile(                               #8x8x6
                    smear, [1, int(enc5.get_shape()[1]), int(enc5.get_shape()[2]), 1])

                concat = tf.concat(3, [enc5, smear])
                enc6 = slim.layers.conv2d(                      #8x8x32
                    concat, 32, [3, 3], stride=1, scope='conv7')

                enc7 = slim.layers.conv2d(  # 8x8x64
                    enc6, 64, [3, 3], stride=1, scope='conv8')

                enc8 = slim.layers.conv2d(  # 8x8x32
                    enc7, 32, [3, 3], stride=1, scope='conv9')

                enc9 = slim.layers.conv2d(  # 8x8x8
                    enc8, 8, [3, 3], stride=1, scope='conv10')

                low_dim_state = slim.layers.conv2d(  # 8x8x1
                    enc9, 1, [3, 3], stride=1, scope='conv11')

                inf_low_state_list.append(low_dim_state)

                pred_low_state_list.append(project_fwd_lowdim(conf, low_dim_state))

                ## start decoding from here:
                print 'decode with inferred lt-state at t{}'.format(t)

            else:  #when propagating latent t = 2,3,...
                print 'decode with predicted lt-state at t{}'.format(t)

                pred_low_state_list.append(project_fwd_lowdim(conf, pred_low_state_list[-1]))

                low_dim_state = pred_low_state_list[-1]


            dec4 = low_dim_state


            dec5 = slim.layers.conv2d_transpose(  #  8x8x16
                dec4, 16, 3, stride=1, scope='convt1')

            dec6 = slim.layers.conv2d_transpose(  # 16x16x16
                dec5, 16, 3, stride=2, scope='convt2')

            dec7 = slim.layers.conv2d_transpose(  # 16x16x32
                dec6, 32, 3, stride=1, scope='convt3')

            dec8 = slim.layers.conv2d_transpose(    #32x32x32
                dec7, 32, 3, stride=2, scope='convt4')

            dec9 = slim.layers.conv2d_transpose(     #64x64x16
                dec8, 16, 3, stride=2, scope='convt5')

            # Using largest hidden state for predicting untied conv kernels.
            dec10 = slim.layers.conv2d_transpose(
                dec9, DNA_KERN_SIZE ** 2, 1, stride=1, scope='convt6')


            transformed = [dna_transformation(prev_image, dec10, DNA_KERN_SIZE)]
            [output] = transformed

            gen_images.append(output)

            current_state = decode_low_dim_obs(conf, low_dim_state)
            gen_states.append(current_state)

    return gen_images, gen_states, inf_low_state_list, pred_low_state_list


def decode_low_dim_obs(conf, low_dim_state):
    low_dim_state_flat = tf.reshape(low_dim_state, [conf['batch_size'], - 1])
    if 'stopgrad' in conf:
        low_dim_state_flat = tf.stop_gradient(low_dim_state_flat)
    state_enc1 = slim.layers.fully_connected(
        low_dim_state_flat,
        100,
        scope='state_enc1')
    state_enc2 = slim.layers.fully_connected(
        state_enc1,
        # int(current_state.get_shape()[1]),
        4,
        scope='state_enc2',
        activation_fn=None)
    current_state = tf.squeeze(state_enc2)
    return current_state


def project_fwd_lowdim(conf, low_state):
    sq_len_lt = int(low_state.get_shape()[-2])

    with tf.variable_scope('latent_model'):
        if 'inc_fl_ltmdl'in conf:  # increased size fully connected latent model
            low_state_flat = tf.reshape(low_state, [conf['batch_size'], - 1])

            low_state_enc1 = slim.layers.fully_connected(
                low_state_flat,
                200,
                scope='hid_state_enc1')
            low_state_enc2 = slim.layers.fully_connected(
                low_state_enc1,
                400,
                scope='hid_state_enc2')
            low_state_enc3 = slim.layers.fully_connected(
                low_state_enc2,
                400,
                scope='hid_state_enc3')
            low_state_enc4 = slim.layers.fully_connected(
                low_state_enc3,
                400,
                scope='hid_state_enc4')
            low_state_enc5 = slim.layers.fully_connected(
                low_state_enc4,
                400,
                scope='hid_state_enc5')
            low_state_enc6 = slim.layers.fully_connected(
                low_state_enc5,
                200,
                scope='hid_state_enc6')
            hid_state_enc7 = slim.layers.fully_connected(
                low_state_enc6,
                int(low_state_flat.get_shape()[1]),
                scope='hid_state_enc7',
                activation_fn=None)

            pred_low_state = tf.reshape(hid_state_enc7, [conf['batch_size'], sq_len_lt, sq_len_lt, 1])

        elif 'inc_conv_ltmdl' in conf:  # increased size fully connected latent model

            ltenc1 = slim.layers.conv2d(  # 8x8
                low_state, 16,[3, 3], stride=1, scope='conv1')

            ltenc2 = slim.layers.conv2d(  # 8x8
                ltenc1, 32, [3, 3], stride=1, scope='conv2')

            ltenc3 = slim.layers.conv2d(  # 8x8
                ltenc2, 64, [3, 3], stride=1, scope='conv3')

            ltenc4 = slim.layers.conv2d(  # 8x8
                ltenc3, 64, [3, 3], stride=1, scope='conv4')

            ltenc5 = slim.layers.conv2d(  # 8x8
                ltenc4, 32, [3, 3], stride=1, scope='conv5')

            ltenc6 = slim.layers.conv2d(  # 8x8
                ltenc5, 16, [3, 3], stride=1, scope='conv6')

            ltenc7 = slim.layers.conv2d(  # 8x8
                ltenc6, 1, [3, 3], stride=1, scope='conv7')

            pred_low_state = ltenc7

        else:
            low_state_flat = tf.reshape(low_state, [conf['batch_size'], - 1])

            # predicting the next hidden state:
            if 'stopgrad' in conf:
                low_state_flat = tf.stop_gradient(low_state_flat)
            low_state_enc1 = slim.layers.fully_connected(
                low_state_flat,
                100,
                scope='hid_state_enc1')
            low_state_enc2 = slim.layers.fully_connected(
                low_state_enc1,
                100,
                scope='hid_state_enc2')
            hid_state_enc3 = slim.layers.fully_connected(
                low_state_enc2,
                int(low_state_flat.get_shape()[1]),
                scope='hid_state_enc3',
                activation_fn=None)
            # predicted low-dimensional state

            pred_low_state = tf.reshape(hid_state_enc3, [conf['batch_size'],sq_len_lt, sq_len_lt, 1])

    return  pred_low_state



def dna_transformation(prev_image, dna_input, DNA_KERN_SIZE):
    """Apply dynamic neural advection to previous image.

    Args:
      prev_image: previous image to be transformed.
      dna_input: hidden lyaer to be used for computing DNA transformation.
    Returns:
      List of images transformed by the predicted CDNA kernels.
    """
    # Construct translated images.
    pad_len = int(np.floor(DNA_KERN_SIZE / 2))
    prev_image_pad = tf.pad(prev_image, [[0, 0], [pad_len, pad_len], [pad_len, pad_len], [0, 0]])
    image_height = int(prev_image.get_shape()[1])
    image_width = int(prev_image.get_shape()[2])

    inputs = []
    for xkern in range(DNA_KERN_SIZE):
        for ykern in range(DNA_KERN_SIZE):
            inputs.append(
                tf.expand_dims(
                    tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                             [-1, image_height, image_width, -1]), [3]))

    inputs = tf.concat(3, inputs)

    # Normalize channels to 1.
    kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
    kernel = tf.expand_dims(
        kernel / tf.reduce_sum(
            kernel, [3], keep_dims=True), [4])

    return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    generated_x = tf.squeeze(generated_x)

    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])


def make_cdna_kerns_summary(cdna_kerns, t, suffix):

    sum = []
    cdna_kerns = tf.split(4, 10, cdna_kerns)
    for i, kern in enumerate(cdna_kerns):
        kern = tf.squeeze(kern)
        kern = tf.expand_dims(kern,-1)
        sum.append(
            tf.image_summary('step' + str(t) +'_filter'+ str(i)+ suffix, kern)
        )

    return  sum
