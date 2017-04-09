import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import dynamics_model
import collections

def get_conf():
        # DATA_DIR = '/home/frederik/Documents/pushing_data/settled_scene_rnd3/train'
    DATA_DIR = '/home/ashvin/lsdc/pushing_data/finer_temporal_resolution_substep10/train'

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
    # conf['batch_size']= 32
    # conf['visualize']=False
    # conf['use_object_pos'] = True

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    return conf

if __name__ == "__main__":
    model = dynamics_model.DynamicsModel(get_conf())

    model.train(100000, True)
    # model.init_sess()
    # print model.train_batch(model.inputs, None, None)
