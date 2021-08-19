# -*- coding: utf-8 -*-
"""
@author: Wiebke Toussaint

Support functions for the src module
"""

import os
from pathlib import Path
import datetime as dt
import delprocess.support as delprocess_support

home_dir = str(Path.home())
usr_dir = os.path.join(home_dir, 'del_data','usr')
obs_dir, profiles_dir, table_dir, rawprofiles_dir = delprocess_support.specifyDataDir()

# root dir
dlrdb_dir = str(Path(__file__).parents[0])

# level 1
experiment_dir = os.path.join(dlrdb_dir, 'experiment')
#obs_dir = os.path.join(dlrdb_dir, 'observations')
feature_dir = os.path.join(dlrdb_dir, 'features')
data_dir = os.path.join(dlrdb_dir, 'data')
eval_dir = os.path.join(dlrdb_dir, 'evaluation')
image_dir = os.path.join(dlrdb_dir, 'img')
log_dir = os.path.join(dlrdb_dir, 'log')
results_dir = os.path.join(dlrdb_dir, 'results')

# level 2 & 3 DATA
dpet_dir = os.path.join(data_dir, 'benchmark_model', 'dpet')
emdata_dir = os.path.join(data_dir, 'experimental_model')
#table_dir = os.path.join(data_dir, 'obs_datasets', 'tables')
#profiles_dir = os.path.join(data_dir, 'obs_datasets', 'profiles')
fdata_dir = os.path.join(data_dir, 'feature_data')
cdata_dir = os.path.join(data_dir, 'class_data')
cluster_dir = os.path.join(data_dir, 'cluster_results')

# level4 data
#rawprofiles_dir = os.path.join(profiles_dir, 'raw')
aggprofiles_dir = os.path.join(profiles_dir, 'aggProfiles')
