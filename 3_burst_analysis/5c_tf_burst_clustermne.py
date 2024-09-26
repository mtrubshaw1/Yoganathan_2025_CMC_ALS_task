#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:16:06 2024

@author: mtrubshaw
"""

"""Fit a GLM and perform statistical significance testing.

"""

import numpy as np
import os
import pandas as pd
from scipy import stats

import glmtools as glm
from osl_dynamics.analysis import power
import mne
from scipy.sparse import coo_matrix
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('plots/compiled_plots', exist_ok=True)
# Load target data

freq_n = 'betas'
bl_window = [0,6]
is_bursts = np.load('data/is_bursts.npy',allow_pickle=True)
burst_tc = []    
for sub in range(len(is_bursts)):
    burst_tc.append(np.mean(is_bursts[sub],axis=0).reshape(-1))

burst_tc = np.array(burst_tc)
time = np.arange(burst_tc.shape[1])/250
burst_tc_bl = mne.baseline.rescale(burst_tc,time,bl_window)

data =burst_tc_bl
# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")


category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "PLS"] = 3
category_list[category_list == "rPLS"] = 3
category_list[category_list == "FDR"] = 3
category_list[category_list == "rFDR"] = 3
uniques = np.unique(category_list)

age = demographics["Age"].values

gender = []
for g in demographics["Sex"].values:
    if g == "M":
        gender.append(0)
    else:
        gender.append(1)
gender = np.array(gender)


missing_struc = demographics["Missing_struc"].values

n_tasks = 3
n_hems = 2




X = [data[category_list==1],data[category_list==2]]
F_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_test(X,adjacency=None,n_permutations=5000)
np.save('data/clusters.npy',cluster)
np.save('data/cluster_pv.npy',cluster_pv)


