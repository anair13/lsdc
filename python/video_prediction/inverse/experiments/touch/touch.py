import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import transforming_dynamics_model
import collections

import tensorflow as tf

if __name__ == "__main__":
    DATA_DIR = '/home/ashvin/lsdc/pushing_data/random_action_var10_touch'
    new_conf = transforming_dynamics_model.get_conf(
        data_dir = DATA_DIR,
        experiment_name="touch",
        transform="none",
        fsize=32,
        mu2=1e-6,
        mu3 = 1,
        autoencoder="decode",
        touch=1,
    )
    conf = new_conf
    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['skip_frame'] = 1
    # conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['visualize']=False

    model = transforming_dynamics_model.DynamicsModel(new_conf)
    model.train(20000, False)
