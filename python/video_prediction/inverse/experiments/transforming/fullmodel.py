import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import transforming_dynamics_model
import collections

import tensorflow as tf

def get_train_conf():
    DATA_DIR = '/home/ashvin/lsdc/pushing_data/finer_temporal_resolution_substep10'
    conf = collections.OrderedDict()
    conf['experiment_name'] = 'fullcontext'
    conf['transform'] = 'meansub'
    conf['data'] = 'ftrs'
    conf['data_dir'] = DATA_DIR       # 'directory containing data.'
    conf['sequence_length'] = 15      # 'sequence length including context frames.'
    conf['skip_frame'] = 2            # 'use ever i-th frame to increase prediction horizon'
    conf['context_frames'] = 1        # of frames before predictions.'
    conf['use_state'] = 1             #'Whether or not to give the state+action to the model'
    conf['train_val_split'] = 1.0    #'The percentage of files to use for the training set vs. the validation set.'
    conf['batch_size'] = 32           #'batch size for training'
    conf['learning_rate'] = 0.001      #'the base learning rate of the generator'
    conf['visualize'] = ''            #'load model from which to generate visualizations
    conf['file_visual'] = ''          # datafile used for making visualizations
    conf['discretize'] = 20
    conf['fsize'] = 100
    conf['masks'] = 0
    conf['run'] = 0
    conf['mu1'] = 0 # transforming mask regularizing weight
    conf['mu2'] = 0.000001 # forward weight
    conf['mu3'] = 0.001 # autoencoder weight
    conf['seq'] = 0 # to alternate training phase
    conf['autoencoder'] = True # autoencoder mode, decode means do not pass gradients, None means no autoencoder at all
    return conf

if __name__ == "__main__":
    new_conf = get_train_conf()

    model = transforming_dynamics_model.DynamicsModel(new_conf)
    model.train(50000)
