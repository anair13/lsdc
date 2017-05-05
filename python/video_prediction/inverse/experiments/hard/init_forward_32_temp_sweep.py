import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import transforming_dynamics_model
import collections

import tensorflow as tf

if __name__ == "__main__":
    init_conf = transforming_dynamics_model.get_conf(
        fsize=32,
        mu2=1e-6,
    )

    for i in [1, 2, 3, 4]:
        new_conf = transforming_dynamics_model.get_conf(
            experiment_name="mining",
            fsize=32,
            mu2=1e-6,
            mu3 = 1,
            autoencoder="decode",
            loadalldata=500,
            miningtemp=5 * (10.0 ** (-i)),
            initialize=1,
        )

        model = transforming_dynamics_model.DynamicsModel(new_conf)
        model.train(10000, init_conf=(50000, init_conf))

        tf.reset_default_graph()

        del model
