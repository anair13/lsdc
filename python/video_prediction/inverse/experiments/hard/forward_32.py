import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import transforming_dynamics_model
import collections

import tensorflow as tf

if __name__ == "__main__":
    new_conf = transforming_dynamics_model.get_conf(
        fsize=32,
        mu2=1e-6,
        mu3 = 1,
        autoencoder="decode",
        loadalldata=1,
    )

    model = transforming_dynamics_model.DynamicsModel(new_conf)
    model.train(50000, False)
