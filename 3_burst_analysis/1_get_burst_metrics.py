#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:05:14 2024

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
from dask.distributed import Client
from pathlib import Path

bl_window = [0,6]


os.makedirs('data',exist_ok=True)
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched")

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values
group = participants["Group"].values

preproc_files = []
smri_files = []
subject_list = []
task = 'task1'








data_meg = []
for n, subject in enumerate(subjects):
    epoch_file_meg = epoch_dir / f'{task}_s0{subject}_meg_epo.fif'
    epochs_meg = mne.read_epochs(epoch_file_meg)
    d_meg = epochs_meg.get_data()
    if task == 'task1':
        if missing_task2[n] == 'No' or missing_task2[n] == 'Mri':
            epoch_file_meg = epoch_dir / f'task2_s0{subject}_meg_epo.fif'
            epochs_meg = mne.read_epochs(epoch_file_meg)
            d_meg2 = epochs_meg.get_data()
            d_meg = np.concatenate((d_meg,d_meg2))
            
    data_meg.append(d_meg)
    
#Standardise timecourses prior to power calculation
data_meg_s = []    
for sub in range(len(data_meg)):
    d = data_meg[sub]
    mean_last_dim = np.mean(d, axis=-1, keepdims=True)
    std_last_dim = np.std(d, axis=-1, keepdims=True)
    data_meg_s.append((d-mean_last_dim)/std_last_dim)
data_meg = data_meg_s    


burst_lifetimes = []
burst_FOs = []
burst_rates = []
burst_meanLTs = []
burst_amps = []
psd_tf= []

is_bursts = []
for sub in range(len(data_meg)):

    data = data_meg[sub][:,:,np.arange(500,1000)]
    freq_range = [13,30]
    fsample = 250
    
    motor_r = np.concatenate(data[:,4,:],axis=0)
    motor_l =np.concatenate(data[:,30,:],axis=0)
    

    is_burst_r, _ = burst_detection(motor_r, freq_range, fsample = fsample, normalise = 'median', 
                                                  threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
    
    
    is_burst_l, _ = burst_detection(motor_l, freq_range, fsample = fsample, normalise = 'median', 
                                                  threshold_dict = {'Method': 'Percentile', 'threshold': 75}, min_n_cycles = 1)
    
    #Calc burst amplitude
    burst_amps.append((np.mean(motor_r[is_burst_r])+np.mean(motor_r[is_burst_l]))/2)
    

                 
    # looks at right and left separately
    is_burst = np.hstack((is_burst_r,is_burst_l))
    
    # if want to use logical or
    is_burst_ = np.logical_or(is_burst_l,is_burst_r)
    tc = is_burst_.reshape(data[:,30,:].shape)
    is_bursts.append(tc)
    
    
    
    # ----- Calculate Burst Metrics -----
    
    # Get Burst Time Metrics
    burst_time_dict = burst_time_metrics(is_burst, fsample)
    
    # Get Sinle Burst Lifetimes      
    burst_lifetimes.append(burst_time_dict['Life Times'])
    
    # Get Average Measures
    burst_FOs.append(burst_time_dict['Burst Occupancy'])
    burst_rates.append(burst_time_dict['Burst Rate'])
    burst_meanLTs.append(np.nanmean(burst_time_dict['Life Times']))


# calculate beta power in rest period (0-1s) (unbaselined)
beta_tc_unstd = np.load(f'data/psd_tf_betas_range_meg_unstd.npy')
toi = np.arange(0,250)#,np.arange(1200,1500)
# toi = np.hstack((toi[0],toi[1]))
beta_rest = np.mean(np.squeeze(beta_tc_unstd[:,:,[toi]])[:,[4,30]],axis=(1,2))

# calculate beta power in rest period (1-3s) (baselined)
beta_tc = np.load(f'data/psd_tf_betas_range_meg.npy')
time = np.arange(beta_tc.shape[-1])/250
beta_tc_bl =  mne.baseline.rescale(beta_tc,time,bl_window)
beta_15 = np.mean(np.squeeze(beta_tc_bl[:,:,[np.arange(500,1000)]])[:,[4,30]],axis=(1,2))


np.save('data/burst_FOs.npy',burst_FOs)
np.save('data/burst_rates.npy',burst_rates)
np.save('data/burst_meanLTs.npy',burst_meanLTs)
np.save('data/burst_amps.npy',burst_amps)
# np.save('data/is_bursts.npy',is_bursts)
np.save('data/beta_rest.npy',beta_rest)
np.save('data/beta_15.npy',beta_15)


