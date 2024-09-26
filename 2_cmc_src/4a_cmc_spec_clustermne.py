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


cohs = np.load('data/cohs_tasks.npy',allow_pickle=True)
coh_rmc =np.expand_dims(np.mean(cohs[:,:,[0], 3],axis=(2)),axis=0)
coh_lmc= np.expand_dims(np.mean(cohs[:,:,[1], 2],axis=(2)),axis=0)

data_=np.concatenate((coh_rmc,coh_lmc),axis=0).swapaxes(0, 1).swapaxes(1, 2)

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

clusters = []
cluster_ps = []
for nt in range(n_tasks):
    for nh in range(n_hems):
            data = data_[:,nt,nh]
            
            X = [data[category_list==1],data[category_list==2]]
            F_obs, cluster, cluster_pv, H0 = mne.stats.permutation_cluster_test(X,adjacency=None)
            
            
            clusters.append(cluster)
            cluster_ps.append(cluster_pv)

# clusters = np.squeeze(clusters)
np.save('data/clusters.npy',clusters)
np.save('data/cluster_ps.npy',cluster_ps)
