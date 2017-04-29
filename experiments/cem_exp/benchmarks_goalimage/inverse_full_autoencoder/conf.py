import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
LSDC_BASE = '/'.join(str.split(current_dir, '/')[:-4])

# local output directory
OUT_DIR = current_dir + '/modeldata'

# from video_prediction.setup_predictor_ltstate import setup_predictor
from video_prediction.inverse.setup_predictor import setup_predictor

import collections

def get_train_conf():
    DATA_DIR = '/home/ashvin/lsdc/pushing_data/finer_temporal_resolution_substep10'
    conf = collections.OrderedDict()
    conf['experiment_name'] = 'fullcontext'
    conf['transform'] = 'meansub'
    conf['data'] = 'ftrs'
    conf['data_dir'] = DATA_DIR       # 'directory containing data.'
    conf['sequence_length'] = 15      # 'sequence length including context frames.'
    conf['skip_frame'] = 2            # 'use ever i-th frame to increase prediction horizon'
    conf['context_frames'] = 2        # of frames before predictions.'
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

configuration = {
	'experiment_name': 'lt_state',
	'setup_predictor': setup_predictor,
	'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
	'current_dir': current_dir,   #'directory for writing summary.' ,
	'pretrained_model': get_train_conf(),     # 'filepath of a pretrained model to resume training from.' ,
	'sequence_length': 15, ##################15,      # 'sequence length, including context frames.' ,
	'context_frames': 2,        # of frames before predictions.' ,
	'use_state': 1,             #'Whether or not to give the state+action to the model' ,
	'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
	'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
	'train_val_split': 0.95,    #'The percentage of files to use for the training set vs. the validation set.' ,
	'batch_size': 32,           #'batch size for training' ,
	'learning_rate': 0.0,     #'the base learning rate of the generator' ,
	'visualize': '',            #'load model from which to generate visualizations
	'file_visual': '',          # datafile used for making visualizations
	'use_conv_low_dim_state':'',  # use low dimensional state computed by convolutions
	'train_latent_model':'',       # whether to add a loss for the latent space model to the objective
	'dna_size': 9,
	'lt_state_factor': 1.0,
	# 'no_pix_distrib': True,
}

