#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:35:56 2024

@author: mtrubshaw
"""
import numpy as np
from scipy.io import loadmat
import pickle

import sys
sys.path.append("/ohba/pi/knobre/mtrubshaw/scripts/helpers/")
from burst_analysis import burst_detection, burst_time_metrics, custom_burst_metric

import mne
import pandas as pd
import os
import matplotlib.pyplot as plt

is_bursts = np.load('data/is_bursts.npy',allow_pickle=True)
betas = np.load('data/psd_tf_betas_range_meg.npy',allow_pickle=True)
participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values
group = participants["Group"].values
# clusters = np.load('data/clusters.npy',allow_pickle=True)
# pvals = np.load('data/cluster_pv.npy',allow_pickle=True)
# beta_pval_mask =  np.where(np.load('data/beta_bl_pval_mask.npy')==True)[-1]

bl_window = [0,6]
# #%% plot burst tf
burst_tc = []    
for sub in range(len(is_bursts)):
    burst_tc.append(np.mean(is_bursts[sub],axis=0).reshape(-1))



burst_tc = np.array(burst_tc)
time = np.arange(burst_tc.shape[1])/250

burst_tc_bl = mne.baseline.rescale(burst_tc,time,bl_window)


tc_als = np.mean(burst_tc_bl[group=='ALS'],axis=0)
tc_hc = np.mean(burst_tc_bl[group=='HC'],axis=0)

        
# plt.plot(tc_als,label="ALS", linewidth=1)
plt.plot(tc_hc,label="HC", linewidth=1)
y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits
plt.axvline(x=250, color='red', linestyle='--', linewidth=1, label='Trigger')
plt.axvline(x=1000, color='red', linestyle='--', linewidth=1)
# if (pvals<0.05).any():
#     mask = np.where(pvals<0.05)
#     mask = mask[0]
#     h = '33'

#     for s in range(len(mask)):
#         if s>1:
#             h = 'D5'
#         idd = clusters[mask[s]][0]
#         plt.fill_between(idd, y_min, y_max, color=f'#FF{h}33', alpha=0.3, label= f'p = {pvals[mask[s]]:.3f}')
#         h = '87'

# plt.title('Beta burst probability over trials (baseline corrected)')

plt.ylabel('Bursting probability change from baseline')
plt.xlabel('Time (samples)')
plt.legend(loc = 'lower left', fontsize = 8)
plt.savefig('plots/burst_prob_tc_HC.png', dpi=300)
plt.show()
