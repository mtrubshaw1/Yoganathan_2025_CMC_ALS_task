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

#%%plot beta tf
beta_tc = np.mean(betas[:,[4,30]],axis=(1))



beta_tc = np.array(beta_tc)
time = np.arange(beta_tc.shape[1])/250

beta_tc_bl = mne.baseline.rescale(beta_tc,time,bl_window)




# tc_als = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "ALS"  ],axis=0)
tc_hc = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "HC"],axis=0)
# tc_fdr = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "FDR"],axis=0)

# plt.title('Motor cortex beta power modulation (baseline corrected)')
plt.axvline(x=250, color='red', linestyle='--', linewidth=1, label='Trigger')
plt.axvline(x=1000, color='red', linestyle='--', linewidth=1)
# plt.plot(tc_als,label="ALS", linewidth=1)
plt.plot(tc_hc,label="HC", linewidth=1)
plt.xlabel('time (samples)')
plt.ylabel('Beta power change from baseline')
# Specify x-values for fill_between
x_values = np.arange(5)  # Assuming 5 data points (0 to 4)
y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits

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

# plt.fill_between(beta_pval_mask[20:], y_min, y_max, color='red', alpha=0.3)    

# plt.fill_between([0,1,2,3], 0, 4, where=(0 > 10), color='C0', alpha=0.3)
# plt.text(350,45,'Reduced beta burst occupancy (p=0.04) \nand rate (p=0.05) in ALS',bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8)
# plt.plot(tc_fdr,label="FDR", linestyle="dashed", linewidth=1)

plt.legend(loc = 'lower left', fontsize = 8)
plt.savefig('plots/beta_power_tf_HC', dpi=300)
plt.show()



# for idx in range(60):
#     plt.title('Motor cortex beta power modulation (baselined 0.5-1s)')
#     plt.axvline(x=250, color='red', linestyle='--', linewidth=1, label='Trigger')
#     plt.axvline(x=(idx*25+12.5), color='green', linewidth=1, label='Trigger')
#     plt.axvline(x=1000, color='red', linestyle='--', linewidth=1)
#     plt.plot(tc_als,label="ALS", linewidth=1)
#     plt.plot(tc_hc,label="HC", linewidth=1)
#     plt.xlabel('time (samples)')
#     plt.ylabel('beta power change from rest')
#     # Specify x-values for fill_between
#     x_values = np.arange(5)  # Assuming 5 data points (0 to 4)
#     y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits

#     # Fill between x0 and x4 across the entire y-axis
#     # plt.fill_between(beta_pval_mask[0:], y_min, y_max, color='red', alpha=0.3)
#     if (pvals<0.05).any():
#         mask = np.where(pvals<0.05)
#         idd = clusters[mask][0,0]
#         # Fill between x0 and x4 across the entire y-axis
#         plt.fill_between(idd, y_min, y_max, color='red', alpha=0.3)

#     # plt.fill_between([0,1,2,3], 0, 4, where=(0 > 10), color='C0', alpha=0.3)
#     # plt.text(350,45,'Reduced beta burst occupancy (p=0.04) \nand rate (p=0.05) in ALS',bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8)
#     # plt.plot(tc_fdr,label="FDR", linestyle="dashed", linewidth=1)
#     plt.legend()
#     plt.savefig(f'../topography/plots/topo/line/beta{idx:02d}', dpi=300)
#     plt.close()