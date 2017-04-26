

current_dir = '/'.join(str.split(__file__, '/')[:-1])
bench_dir = '/'.join(str.split(__file__, '/')[:-2])

from lsdc.algorithm.policy.cem_controller_goalimage import CEM_controller
policy = {
    'type' : CEM_controller,
    'use_goalimage':"",
    'low_level_ctrl': None,
    'usenet': True,
    'num_samples': 32,
    'nactions': 5,
    'repeat': 3,
    'initial_std': 7,
    'netconf': current_dir + '/conf.py',
    'use_first_plan': False, # execute MPC instead using firs plan
    'iterations': 5,
    'load_goal_image':'make_easy_goal'
}

agent = {
    'T': 30,
    'use_goalimage':"",
    'start_confs': bench_dir + '/make_easy_goal/configs_easy_goal',
    # 'num_objects': 4,
    'skip_first': 5,   #skip first N time steps to let the scene settle
    'substeps': 10,  #6
}


# agent = {
#     'type': AgentMuJoCo,
#     'filename': './mjc_models/pushing2d.xml',
#     'filename_nomarkers': './mjc_models/pushing2d.xml',
#     'data_collection': True,
#     'x0': np.array([0., 0., 0., 0.]),
#     'dt': 0.05,
#     'substeps': 10,  #6

#     'conditions': common['conditions'],
#     'T': 30,
#     'skip_first': 5,   #skip first N time steps to let the scene settle
#     'additional_viewer': True,
#     'image_dir': common['data_files_dir'] + "imagedata_file",
#     'image_height' : IMAGE_HEIGHT,
#     'image_width' : IMAGE_WIDTH,
#     'image_channels' : IMAGE_CHANNELS,
#     'num_objects': num_objects,
#     'record':False
# }
