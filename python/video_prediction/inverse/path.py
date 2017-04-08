import os

project_dir = os.getenv("POKE_PROJECT_DIR", "/home/ashvin/tf-poke/pokebot/baxter-poke-prediction/")
tf_data_dir = os.getenv("POKE_TF_DIR", "/home/ashvin/tf-poke/tf-data/")
poke_data_dir = os.getenv("POKE_DATA_DIR", "/data/shared/arena/")