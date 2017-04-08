import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import autoencoder_model
import collections

def get_conf():
        # DATA_DIR = '/home/frederik/Documents/pushing_data/settled_scene_rnd3/train'
    DATA_DIR = '/home/ashvin/lsdc/python/ashvin/train'

    conf = collections.OrderedDict()
    conf['model'] = 'autoencoder'
    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size']= 32
    conf['visualize']=False
    conf['use_object_pos'] = True
    conf['initLr'] = 0.0001

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    return conf

model = autoencoder_model.AutoencoderModel(get_conf())

model.train(100000, True)
# model.init_sess()
# print model.train_batch(model.inputs, None, None)
