import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle

from utils_vpred.adapt_params_visualize import adapt_params_visualize
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import utils_vpred.create_gif
import pdb

from read_tf_record import build_tfrecord_input

from utils_vpred.skip_example import skip_example

from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('pretrained', '', 'model weights from which to resume training')

## Helper functions
def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      peak signal to noise ratio (PSNR)
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):
    def __init__(self,
                 conf,
                 images=None,
                 actions=None,
                 states=None,
                 sequence_length=None,
                 reuse_scope=None,
                 ):


        from prediction_hiddenstate import construct_model

        if sequence_length is None:
            sequence_length = conf['sequence_length']

        self.prefix = prefix = tf.placeholder(tf.string, [])
        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        actions = tf.split(1, actions.get_shape()[1], actions)
        actions = [tf.squeeze(act) for act in actions]
        states = tf.split(1, states.get_shape()[1], states)
        states = [tf.squeeze(st) for st in states]
        images = tf.split(1, images.get_shape()[1], images)
        images = [tf.squeeze(img) for img in images]


        if reuse_scope is None:
            gen_images, gen_states, gen_masks, inf_low_state, pred_low_state = construct_model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                k=conf['schedsamp_k'],
                use_state=conf['use_state'],
                context_frames=conf['context_frames'],
                conf=conf)
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=True):
                gen_images, gen_states, gen_masks, inf_low_state, pred_low_state = construct_model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    k=conf['schedsamp_k'],
                    use_state=conf['use_state'],
                    context_frames=conf['context_frames'],
                    conf= conf)

        self.inf_low_state = inf_low_state
        self.gen_images = gen_images
        self.gen_masks = gen_masks
        self.gen_states = gen_states

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        if 'prop_latent' in conf:
            return # do not do backprop when visualizing latent model forward propagation

        # L2 loss, PSNR for eval.
        loss, psnr_all = 0.0, 0.0
        for i, x, gx in zip(
                range(len(gen_images)), images[conf['context_frames']:],
                gen_images[conf['context_frames'] - 1:]):
            recon_cost = mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(
                tf.scalar_summary(prefix + '_recon_cost' + str(i), recon_cost))
            summaries.append(tf.scalar_summary(prefix + '_psnr' + str(i), psnr_i))

            loss += recon_cost

        for i, state, gen_state in zip(
                range(len(gen_states)), states[conf['context_frames']:],
                gen_states[conf['context_frames'] - 1:]):
            state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
            summaries.append(
                tf.scalar_summary(prefix + '_state_cost' + str(i), state_cost))
            loss += state_cost
        summaries.append(tf.scalar_summary(prefix + '_psnr_all', psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])
        summaries.append(tf.scalar_summary(prefix + '_loss', loss))


        if 'train_latent_model' in conf:
            lt_state_cost_accum = 0.0
            for i, inf_state, pred_state in zip(
                    range(len(inf_low_state)), inf_low_state[1:],pred_low_state[:-1]):

                lt_state_cost = mean_squared_error(inf_state, pred_state)*conf['lt_state_factor']
                summaries.append(tf.scalar_summary(prefix + '_low_state_cost' + str(i+1), lt_state_cost))
                lt_state_cost_accum += lt_state_cost

            if not 'joint' in conf:
                lt_model_var = tf.get_default_graph().get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/latent_model')

                train_lt_op = tf.train.AdamOptimizer(self.lr).minimize(lt_state_cost_accum, var_list=lt_model_var)
                with tf.control_dependencies([train_lt_op]):
                    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

            else:
                loss += lt_state_cost_accum
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.merge_summary(summaries)


def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf = adapt_params_visualize(conf, FLAGS.visualize)
    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'Constructing models and inputs.'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images, actions, states = build_tfrecord_input(conf, training=True)
        model = Model(conf, images, actions, states, conf['sequence_length'])

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_model = Model(conf, val_images, val_actions, val_states,
                          conf['sequence_length'], training_scope)


    print 'Constructing saver.'
    # Make saver.
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES), max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.train.SummaryWriter(
        conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.initialize_all_variables())

    if conf['visualize']:
        saver.restore(sess, conf['visualize'])

        feed_dict = {val_model.lr: 0.0,
                     val_model.prefix: 'vis',
                     val_model.iter_num: 0 }

        file_path = conf['output_dir']
        if 'use_masks' in conf:
            gen_images, ground_truth, mask_list = sess.run([val_model.gen_images,
                                                            val_images, val_model.gen_masks],
                                                           feed_dict)
            cPickle.dump(mask_list, open(file_path + '/mask_list.pkl', 'wb'))
        else:
            gen_images, ground_truth = sess.run([val_model.gen_images,
                                                            val_images],
                                                           feed_dict)

        cPickle.dump(gen_images, open(file_path + '/gen_image_seq.pkl','wb'))
        cPickle.dump(ground_truth, open(file_path + '/ground_truth.pkl', 'wb'))

        print 'written files to:' + file_path

        if 'prop_latent' in conf:
            suffix = 'prop_lt'
        else: suffix = None
        trajectories = utils_vpred.create_gif.comp_video(conf['output_dir'], conf, suffix)
        if 'use_masks' in conf:
            utils_vpred.create_gif.comp_masks(conf['output_dir'], conf, trajectories)
        return

    itr_0 =0
    if FLAGS.pretrained or (conf['pretrained_model'] != ''):    # is the order of initialize_all_variables() and restore() important?!?
        if FLAGS.pretrained is not '':
            conf['pretrained_model'] = conf['output_dir'] + '/' + FLAGS.pretrained
        saver.restore(sess, conf['pretrained_model'])
        # resume training at iteration step of the loaded model:
        import re
        itr_0 = re.match('.*?([0-9]+)$', conf['pretrained_model']).group(1)
        itr_0 = int(itr_0)
        print 'resuming training at iteration:  ', itr_0

    tf.logging.info('iteration number, cost')

    starttime = datetime.now()
    t_iter = []
    # Run training.
    for itr in range(itr_0, conf['num_iterations'], 1):
        t_startiter = datetime.now()
        # Generate new batch of data_files.
        feed_dict = {model.prefix: 'train',
                     model.iter_num: np.float32(itr),
                     model.lr: conf['learning_rate']}
        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                        feed_dict)

        # Print info: iteration #, cost.
        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 2:
            # Run through validation set.
            feed_dict = {val_model.lr: 0.0,
                         val_model.prefix: 'val',
                         val_model.iter_num: np.float32(itr)}
            _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                          feed_dict)
            summary_writer.add_summary(val_summary_str, itr)


        if (itr) % SAVE_INTERVAL == 2:
            tf.logging.info('Saving model to' + conf['output_dir'])
            saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        t_iter.append((datetime.now() - t_startiter).seconds * 1e6 +  (datetime.now() - t_startiter).microseconds )

        if itr % 100 == 1:
            hours = (datetime.now() -starttime).seconds/3600
            tf.logging.info('running for {0}d, {1}h, {2}min'.format(
                (datetime.now() - starttime).days,
                hours,
                (datetime.now() - starttime).seconds/60 - hours*60))
            avg_t_iter = np.sum(np.asarray(t_iter))/len(t_iter)
            tf.logging.info('time per iteration: {0}'.format(avg_t_iter/1e6))
            tf.logging.info('expected for complete training: {0}h '.format(avg_t_iter /1e6/3600 * conf['num_iterations']))

        if (itr) % SUMMARY_INTERVAL:
            summary_writer.add_summary(summary_str, itr)

    tf.logging.info('Saving model.')
    saver.save(sess, conf['output_dir'] + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
