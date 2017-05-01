import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import transforming_dynamics_model
import collections

import tensorflow as tf

def get_train_conf():
    conf = transforming_dynamics_model.DEFAULT_CONF.copy()
    conf['fsize'] = 128
    conf['mu2'] = 1e-6 # forward weight
    conf['mu3'] = 1 # autoencoder weight
    conf['autoencoder'] = "decode"
    return conf

if __name__ == "__main__":
    new_conf = get_train_conf()

    model = transforming_dynamics_model.DynamicsModel(new_conf)
    model.train(50000)
