import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics_model
import collections

import tensorflow as tf
import read_tf_record

def get_conf():
        # DATA_DIR = '/home/frederik/Documents/pushing_data/settled_scene_rnd3/train'
    DATA_DIR = '/home/ashvin/lsdc/python/ashvin/train'

    conf = collections.OrderedDict()
    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size']= 2
    conf['visualize']=False
    conf['use_object_pos'] = True
    conf['initLr'] = 0.001

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    return conf

# training_data = data_generator(get_conf(), sess)
with tf.device('/cpu:0'):
    image_batch, action_batch, state_batch, object_pos_batch = read_tf_record.build_tfrecord_input(get_conf(), training=True)

sess = tf.InteractiveSession() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.train.start_queue_runners(sess)
sess.run(tf.initialize_all_variables())

print "yo"
image_data, action_data, state_data, object_pos = sess.run([image_batch, action_batch, state_batch, object_pos_batch])
print "done"
