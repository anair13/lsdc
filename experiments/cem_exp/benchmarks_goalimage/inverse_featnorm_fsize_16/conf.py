import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
LSDC_BASE = '/'.join(str.split(current_dir, '/')[:-4])

# local output directory
OUT_DIR = current_dir + '/modeldata'

# from video_prediction.setup_predictor_ltstate import setup_predictor
from video_prediction.inverse.setup_predictor import setup_predictor
from video_prediction.inverse import transforming_dynamics_model

import collections

def get_train_conf():
    conf = transforming_dynamics_model.DEFAULT_CONF.copy()
    conf['fsize'] = 16
    conf['mu2'] = 1e-6 # forward weight
    conf['mu3'] = 1e-3 # autoencoder weight
    conf['autoencoder'] = 1
    conf['mu4'] = 1
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

