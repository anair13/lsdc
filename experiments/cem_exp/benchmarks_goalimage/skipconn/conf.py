import os
current_dir = os.path.dirname(os.path.realpath(__file__))

# tf record data location:
LSDC_BASE = '/'.join(str.split(current_dir, '/')[:-4])

# local output directory
OUT_DIR = current_dir + '/modeldata'

from video_prediction.setup_predictor_ltstate import setup_predictor

configuration = {
'experiment_name': __file__,
'setup_predictor': setup_predictor,
'output_dir': OUT_DIR,      #'directory for model checkpoints.' ,
'current_dir': current_dir,   #'directory for writing summary.' ,
'num_iterations': 50000,   #'number of training iterations.' ,
'pretrained_model': LSDC_BASE +'/tensorflow_data/hidden_state/skipconn/modeldata/model48002',     # 'filepath of a pretrained model to resume training from.' ,
'sequence_length': 15,     # 'sequence length, including context frames.' ,
'context_frames': 2,        # of frames before predictions.' ,
'use_state': 1,             #'Whether or not to give the state+action to the model' ,
'model': 'DNA',            #'model architecture to use - CDNA, DNA, or STP' ,
'num_masks': 1,            # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
'schedsamp_k': -1,       # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
'batch_size': 200,           #'batch size for training' ,
'learning_rate': 0.0,     #'the base learning rate of the generator' ,
'file_visual': '',          # datafile used for making visualizations
'use_conv_low_dim_state':'',  # use low dimensional state computed by convolutions
'train_latent_model':'',       # whether to add a loss for the latent space model to the objective
'dna_size': 9,
'lt_state_factor': 1.0,
'num_lt_featuremaps':16,
'skip':''
}