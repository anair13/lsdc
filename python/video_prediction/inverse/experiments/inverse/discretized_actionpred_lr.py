import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import dynamics_model
import collections

def get_conf():
    DATA_DIR = '/home/ashvin/lsdc/pushing_data/finer_temporal_resolution_substep10/train'
    conf = collections.OrderedDict()
    conf['experiment_name'] = 'discretized_actionpred'
    conf['transform'] = 'none'
    conf['data'] = 'ftrs'
    conf['data_dir'] = DATA_DIR       # 'directory containing data.'
    conf['num_iterations'] = 50000    #'number of training iterations.'
    conf['pretrained_model'] = ''     # 'filepath of a pretrained model to resume training from.'
    conf['sequence_length'] = 15      # 'sequence length including context frames.'
    conf['skip_frame'] = 2            # 'use ever i-th frame to increase prediction horizon'
    conf['context_frames'] = 2        # of frames before predictions.'
    conf['use_state'] = 1             #'Whether or not to give the state+action to the model'
    conf['model'] = 'ac'            #'model architecture to use - CDNA DNA or STP'
    conf['num_masks'] = 10            # 'number of masks usually 1 for DNA 10 for CDNA STN.'
    conf['schedsamp_k'] = 900.0       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.'
    conf['train_val_split'] = 0.95    #'The percentage of files to use for the training set vs. the validation set.'
    conf['batch_size'] = 32           #'batch size for training'
    conf['learning_rate'] = 0.001      #'the base learning rate of the generator'
    conf['visualize'] = ''            #'load model from which to generate visualizations
    conf['file_visual'] = ''          # datafile used for making visualizations
    conf['discretize'] = 20
    return conf

if __name__ == "__main__":
    model = dynamics_model.DynamicsModel(get_conf())
    model.train(100000, True)
