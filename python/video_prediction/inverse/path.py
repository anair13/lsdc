import os

project_dir = os.getenv("LSDC_PROJECT_DIR", "/home/ashvin/tf-poke/pokebot/baxter-poke-prediction/")
tf_data_dir = os.getenv("LSDC_TF_DIR", "/home/ashvin/tf-lsdc/tf-data/")
poke_data_dir = os.getenv("LSDC_DATA_DIR", "/data/shared/arena/")