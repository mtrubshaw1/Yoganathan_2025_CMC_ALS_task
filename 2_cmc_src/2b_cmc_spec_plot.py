#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:01:50 2024

@author: mtrubshaw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Directories
participants = pd.read_csv("../demographics/task_demographics.csv")
group = participants["Group"].values
#de-correct over 3 tasks
pvalues = np.load("data/contrast_0_pvalues.npy")/3

tasks = ['task1', 'left', 'right']


cohs_tasks = np.load('data/cohs_tasks.npy')
f = np.load('data/f.npy')

for n,task in enumerate(tasks):
    cohs = cohs_tasks[:,n]

    
    # plot CMC
    right_mc = [0]
    left_mc = [1]
    r_emg = 2
    l_emg =3
    
    cohs_als = cohs[group=='ALS']
    cohs_hc = cohs[group=='HC']
    
    coh_rmc_lemg_als = np.mean(cohs_als[:,right_mc, l_emg],axis=(0,1)) 
    coh_rmc_lemg_hc = np.mean(cohs_hc[:,right_mc, l_emg],axis=(0,1))
    coh_rmc_remg_als = np.mean(cohs_als[:,right_mc, r_emg],axis=(0,1)) 
    coh_rmc_remg_hc = np.mean(cohs_hc[:,right_mc, r_emg],axis=(0,1))
    
    plt.plot(f,coh_rmc_lemg_als,label='ALS - contra')
    plt.plot(f,coh_rmc_lemg_hc,label='HC - contra')
    # plt.plot(f,coh_rmc_remg_als,label='ALS - ipsi', linestyle='dashdot')
    # plt.plot(f,coh_rmc_remg_hc,label='HC - ipsi',linestyle='dashdot')
    y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits    
    if (pvalues[n, 0] < 0.05).any():
        mask = list(np.squeeze(np.array(np.where(pvalues[n, 0]<0.05))))
        for m in mask:
            plt.fill_between([f[m]-0.57,f[m]+0.57], y_min, y_max, alpha=0.2,color='red')
    plt.xlabel('frequency')
    plt.ylabel('CMC')
    plt.title(f'CMC right motor cortex - task = {task}')
    plt.legend()
    plt.savefig(f'plots/{task}_rmc_cmc.png',dpi=300)
    plt.show()
    
    
    coh_rmc_lemg_als = np.mean(cohs_als[:,left_mc,r_emg],axis=(0,1))
    coh_rmc_lemg_hc = np.mean(cohs_hc[:,left_mc,r_emg],axis=(0,1))
    coh_rmc_remg_als = np.mean(cohs_als[:,left_mc, l_emg],axis=(0,1)) 
    coh_rmc_remg_hc = np.mean(cohs_hc[:,left_mc, l_emg],axis=(0,1))
    
    plt.plot(f,coh_rmc_lemg_als,label='ALS - contra')
    plt.plot(f,coh_rmc_lemg_hc,label='HC - contra')
    # plt.plot(f,coh_rmc_remg_als,label='ALS - ipsi', linestyle='dashdot')
    # plt.plot(f,coh_rmc_remg_hc,label='HC - ipsi',linestyle='dashdot')
    y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits    
    if (pvalues[n, 1] < 0.06).any():
        mask = list(np.squeeze(np.array(np.where(pvalues[n, 1]<0.06))))
        for m in mask:
            plt.fill_between([f[m]-0.57,f[m]+0.57], y_min, y_max, alpha=0.2,color='red')
    plt.title(f'CMC left motor cortex - task = {task}')
    plt.xlabel('frequency')
    plt.ylabel('CMC')
    plt.legend()
    plt.savefig(f'plots/{task}_lmc_cmc.png',dpi=300)
    plt.show()