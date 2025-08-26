import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil
from shutil import copyfile

import re
import gzip
import json
import arviz as az
from tqdm import tqdm
import pickle

##################
output_folder = '7_reduction_percentages'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    try:
        for ea in os.listdir(output_folder):
            os.remove(output_folder + '/' + ea)
    except:
        for ea in os.listdir(output_folder):
            shutil.rmtree(output_folder + '/' + ea)


before_folder = '6_NPC_DFS_after2-2'
after_folder = '5_NPC_DFS_after2-1'

for file in os.listdir(before_folder):
    if '.csv' in file:
        df_before = pd.read_csv(before_folder+'/'+file)
        df_after = pd.read_csv(after_folder+'/'+file.replace('after2-2','after_2-1'))

        to_add_cols = ['predicted_after_mcmc']
        df_before[to_add_cols] = df_after[to_add_cols].values

        df_before['reduction_perc_mcmc'] = (df_before['predicted_after2-2_mcmc'].values - df_after['predicted_after_mcmc'].values)/df_before['predicted_after2-2_mcmc'].values
        df_before['overall_reduction_perc_mcmc'] = (df_before['predicted_after2-2_mcmc'].sum() - df_after['predicted_after_mcmc'].sum())/df_before['predicted_after2-2_mcmc'].sum()
        
        df_before.to_csv(output_folder+'/'+file.replace('6_DFS_sites_pred_after2-2','7_reduction_perc'))
































