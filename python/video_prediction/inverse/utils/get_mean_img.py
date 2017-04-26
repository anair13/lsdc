import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf
import read_tf_record
from tensorflow.python.platform import gfile

import numpy as np

DATA_DIR = '/home/ashvin/lsdc/pushing_data/finer_temporal_resolution_substep10/train'

def get_conf():
        # DATA_DIR = '/home/frederik/Documents/pushing_data/settled_scene_rnd3/train'

    conf = {
    'experiment_name': 'originial_with_newdata',
    'data_dir': DATA_DIR,       # 'directory containing data.' ,
    'num_iterations': 50000,   #'number of training iterations.' ,
    'pretrained_model': '',     # 'filepath of a pretrained model to resume training from.' ,
    'sequence_length': 15,      # 'sequence length, including context frames.' ,
    'skip_frame': 2,            # 'use ever i-th frame to increase prediction horizon' ,
    'context_frames': 2,        # of frames before predictions.' ,
    'use_state': 1,             #'Whether or not to give the state+action to the model' ,
    'model': 'CDNA',            #'model architecture to use - CDNA, DNA, or STP' ,
    'num_masks': 10,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
    'schedsamp_k': 900.0,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
    'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
    'batch_size': 32,           #'batch size for training' ,
    'learning_rate': 0.001,     #'the base learning rate of the generator' ,
    'visualize': '',            #'load model from which to generate visualizations
    'file_visual': '',          # datafile used for making visualizations
    }
    # conf = collections.OrderedDict()
    # conf['schedsamp_k'] = -1  # don't feed ground truth
    # conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    # conf['skip_frame'] = 1
    # conf['train_val_split']= 0.95
    # conf['sequence_length']= 15      # 'sequence length, including context frames.'
    # conf['use_state'] = True
    # conf['batch_size']= 2
    # conf['visualize']=False
    # conf['use_object_pos'] = True
    # conf['initLr'] = 0.001

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    return conf

# training_data = data_generator(get_conf(), sess)
conf = get_conf()
filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
with tf.device('/cpu:0'):
    image_batch, action_batch, state_batch = read_tf_record.build_tfrecord_input(conf, training=True)

sess = tf.InteractiveSession() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.train.start_queue_runners(sess)
sess.run(tf.initialize_all_variables())

image_means = []
N = 1000
print "averaging", N*conf['batch_size']*conf['sequence_length'], "examples"
for i in range(N):
    # print i, filenames[i]
    image_data, action_data, state_data = sess.run([image_batch, action_batch, state_batch])
    image_means.append(np.sum(image_data, (0, 1))/float(conf['batch_size']*conf['sequence_length']))
image_means = np.sum(np.array(image_means), 0)/N
savefile = DATA_DIR + '/mean.npy'

np.save(savefile, image_means)
