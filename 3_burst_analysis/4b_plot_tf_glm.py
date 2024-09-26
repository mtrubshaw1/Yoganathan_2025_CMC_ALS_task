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
freq_n = 'betas'

bl_window = [0,6]
is_bursts = np.load('data/is_bursts.npy',allow_pickle=True)
betas = np.load(f'data/psd_tf_{freq_n}_range_meg.npy',allow_pickle=True)
participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values
group = participants["Group"].values

beta_pval_mask =  np.where(np.load('data/beta_bl_pval_mask.npy')==True)[-1]
# beta_pval_mask =  np.where(np.load('data/beta_bl_pval_mask.npy')==True)[-1]

# #%% plot burst tf
# burst_tc = []    
# for sub in range(len(is_bursts)):
#     burst_tc.append(np.mean(is_bursts[sub],axis=0).reshape(-1))



# burst_tc = np.array(burst_tc)
# time = np.arange(burst_tc.shape[1])/250

# burst_tc_bl = mne.baseline.rescale(burst_tc,time,[0.5,1])
# # burst_tc_bl= burst_tc

# tc_als = np.mean([burst_tc_bl[i] for i in range(len(group)) if group[i] == "ALS"  ],axis=0)
# tc_hc = np.mean([burst_tc_bl[i] for i in range(len(group)) if group[i] == "HC"],axis=0)
# tc_fdr = np.mean([burst_tc_bl[i] for i in range(len(group)) if group[i] == "FDR"],axis=0)

# plt.plot(tc_als,label="ALS", linewidth=1)
# plt.plot(tc_hc,label="HC", linewidth=1)
# # plt.plot(tc_fdr,label="FDR", linestyle="dashed", linewidth=1)
# plt.legend()
# plt.show()

#%%plot beta tf
beta_tc = []    

beta_tc= np.mean(betas[:,[4,30]],axis=(1))



beta_tc = np.array(beta_tc)
time = np.arange(beta_tc.shape[1])/250

beta_tc_bl = mne.baseline.rescale(beta_tc,time,bl_window)
# beta_tc_bl= beta_tc
np.save('data/beta_tc.npy',beta_tc)
np.save('data/beta_tc_bl.npy',beta_tc_bl)



tc_als = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "ALS"  ],axis=0)
tc_hc = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "HC"],axis=0)
# tc_fdr = np.mean([beta_tc_bl[i] for i in range(len(group)) if group[i] == "FDR"],axis=0)

# plt.title('Motor cortex beta power modulation (baseline corrected)')
plt.axvline(x=250, color='red', linestyle='--', linewidth=1, label='Trigger')
plt.axvline(x=1000, color='red', linestyle='--', linewidth=1)
plt.plot(tc_als,label="ALS", linewidth=1)
plt.plot(tc_hc,label="HC", linewidth=1)
plt.xlabel('time (samples)')
plt.ylabel('beta power change from rest')
# Specify x-values for fill_between
x_values = np.arange(5)  # Assuming 5 data points (0 to 4)
y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits

# Fill between x0 and x4 across the entire y-axis
plt.fill_between(beta_pval_mask[:34], y_min, y_max, color='red', alpha=0.3)
plt.fill_between(beta_pval_mask[34:], y_min, y_max, color='red', alpha=0.3)
# plt.fill_between(beta_pval_mask, y_min, y_max, color='red', alpha=0.3)    

# plt.fill_between([0,1,2,3], 0, 4, where=(0 > 10), color='C0', alpha=0.3)
# plt.text(350,45,'Reduced beta burst occupancy (p=0.04) \nand rate (p=0.05) in ALS',bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8)
# plt.plot(tc_fdr,label="FDR", linestyle="dashed", linewidth=1)
plt.legend(loc = 'lower left', fontsize = 8)
plt.savefig('plots/beta_power_tf_glm', dpi=300)
plt.show()



# for idx in range(60):
#     plt.title('Motor cortex beta power modulation (baselined 0-0.6s)')
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
#     plt.fill_between(beta_pval_mask[0:], y_min, y_max, color='red', alpha=0.3)


#     # plt.fill_between([0,1,2,3], 0, 4, where=(0 > 10), color='C0', alpha=0.3)
#     # plt.text(350,45,'Reduced beta burst occupancy (p=0.04) \nand rate (p=0.05) in ALS',bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8)
#     # plt.plot(tc_fdr,label="FDR", linestyle="dashed", linewidth=1)
#     plt.legend()
#     plt.savefig(f'../topography/plots/topo/line/beta{idx:02d}', dpi=300)
#     plt.close()