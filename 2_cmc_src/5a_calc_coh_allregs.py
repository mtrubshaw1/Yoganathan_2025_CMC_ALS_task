#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:23:04 2024

@author: mtrubshaw
"""

import mne
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from osl_dynamics.analysis import connectivity
from osl_dynamics.analysis import spectral


# Directories
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched")
participants = pd.read_csv("../demographics/task_demographics.csv")
subjects = participants["Subject"].values
group = participants["Group"].values
missing_task2= participants["Missing_task2"].values #some ppts had missing task2

bl_window = [0,6]


tasks = ['task1',  'right','left']

cohs_tasks  = []
for task in tasks:
    #choose task -- options ['task1', 'left','right] - if task1 is selected, task2 will automatically be added if available
    # task = 'left'
    
    
    # sampling frequency
    fsample = 250
    
    # load epoched emg and meg source files, read data  
    data_emg = []
    data_meg = []
    for n, subject in enumerate(subjects):
        epoch_file_emg = epoch_dir / f'{task}_s0{subject}_emg_grip_epo.fif'
        epochs_emg = mne.read_epochs(epoch_file_emg)
        d_emg = epochs_emg.get_data()
        if task == 'task1':
            if missing_task2[n] == 'No' or missing_task2[n] == 'Mri':
                epoch_file_emg = epoch_dir / f'task2_s0{subject}_emg_grip_epo.fif'
                epochs_emg = mne.read_epochs(epoch_file_emg)
                d_emg2 = epochs_emg.get_data()
                d_emg = np.concatenate((d_emg,d_emg2))
                
        data_emg.append(d_emg)
        
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
    
    # extract channel names
    ch_names_emg = np.array(epochs_emg.ch_names)
    ch_names_meg = np.array(epochs_meg.ch_names)
    
    
    # extract data from motor cortices and emg channels and concatenate
    emg_motor = []
    channels = [1]
    emg_ch = [1]
    for n, subject in enumerate(subjects):
        meg_sub = data_meg[n]
        motor = data_meg[n][:,:52]
        
        emg_sub = data_emg[n]
        emg = emg_sub[:,[0,1]]
        
        time = np.arange(motor.shape[2])/250

        emg_motor_ = np.concatenate((motor,emg),axis=1).swapaxes(0, 1)
        
        #baseline the signal to the period just before the stimulus (1s)
        emg_motor_bl = mne.baseline.rescale(emg_motor_,time,bl_window)
        
        #rectify the emg channels
        motor_bl = emg_motor_bl[:52]
        emg_bl = abs(emg_motor_bl[52:])
        emg_motor_bl_rec = np.concatenate((motor_bl,emg_bl),axis=0)
        
        # select data from 500-1000 samples (that is 1s to 3s post stimulus)
        emg_motor.append(emg_motor_bl_rec[:,:,500:1000])
        
    
    #calculate multitaper spectra and save coherence matrices
    cohs = []
    fs =[]
    for n in range(len(emg_motor)):
        em = list(emg_motor[n].swapaxes(0, 1).swapaxes(2, 1))
        f, psd, coh = spectral.multitaper_spectra(em, sampling_frequency=fsample,calc_coh=True,standardize=True,n_jobs=6
                                                  ,frequency_range=[8,40],window_length=200)
        
        psd = []
        cohs.append(np.mean(coh,axis=0))
    cohs = (np.array(cohs))
    # np.save('data/cohs.npy',cohs)
    cohs_tasks.append(cohs)
    
    
    
    # # plot CMC
    # right_mc = [0]
    # left_mc = [1]
    # r_emg = 2
    # l_emg =3
    
    # cohs_als = cohs[group=='ALS']
    # cohs_hc = cohs[group=='HC']
    
    # coh_rmc_lemg_als = np.mean(cohs_als[:,right_mc, l_emg],axis=(0,1)) 
    # coh_rmc_lemg_hc = np.mean(cohs_hc[:,right_mc, l_emg],axis=(0,1))
    # coh_rmc_remg_als = np.mean(cohs_als[:,right_mc, r_emg],axis=(0,1)) 
    # coh_rmc_remg_hc = np.mean(cohs_hc[:,right_mc, r_emg],axis=(0,1))
    
    # plt.plot(f,coh_rmc_lemg_als,label='ALS - contra')
    # plt.plot(f,coh_rmc_lemg_hc,label='HC - contra')
    # plt.plot(f,coh_rmc_remg_als,label='ALS - ipsi', linestyle='dashdot')
    # plt.plot(f,coh_rmc_remg_hc,label='HC - ipsi',linestyle='dashdot')
    # plt.xlabel('frequency')
    # plt.ylabel('CMC')
    # plt.title(f'CMC right motor cortex - task = {task}')
    # plt.legend()
    # plt.savefig(f'plots/{task}_rmc_cmc.png',dpi=300)
    # plt.show()
    
    
    # coh_rmc_lemg_als = np.mean(cohs_als[:,left_mc,r_emg],axis=(0,1))
    # coh_rmc_lemg_hc = np.mean(cohs_hc[:,left_mc,r_emg],axis=(0,1))
    
    # plt.plot(f,coh_rmc_lemg_als,label='ALS - contra')
    # plt.plot(f,coh_rmc_lemg_hc,label='HC - contra')
    # plt.plot(f,coh_rmc_remg_als,label='ALS - ipsi', linestyle='dashdot')
    # plt.plot(f,coh_rmc_remg_hc,label='HC - ipsi',linestyle='dashdot')
    # plt.title(f'CMC left motor cortex - task = {task}')
    # plt.xlabel('frequency')
    # plt.ylabel('CMC')
    # plt.legend()
    # plt.savefig(f'plots/{task}_lmc_cmc.png',dpi=300)
    # plt.show()

cohs_tasks = np.array(cohs_tasks).swapaxes(0,1)
np.save('data/cohs_tasks_allregs.npy',cohs_tasks)
np.save('data/f.npy',f)
