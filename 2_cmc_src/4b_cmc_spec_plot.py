#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:01:50 2024

@author: mtrubshaw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('plots/cluster',exist_ok=True)
# Directories
participants = pd.read_csv("../demographics/task_demographics.csv")
group = participants["Group"].values
#de-correct over 3 tasks
pvalues = np.load("data/cluster_ps.npy",allow_pickle=True)
pvalues = pvalues*len(pvalues)
clusters = np.load("data/clusters.npy",allow_pickle=True)
tasks = ['bilateral', 'left', 'right']
hemis = ['right','left']



cohs_tasks = np.load('data/cohs_tasks.npy')
f = np.load('data/f.npy')

j = 0
for n,task in enumerate(tasks):
    for h, hemi in enumerate(hemis):
        mask = []
        cohs = cohs_tasks[:,n]
    
        
        # plot CMC
        right_mc = [0]
        left_mc = [1]
        r_emg = 2
        l_emg =3
        
        cohs_als = cohs[group=='ALS']
        cohs_hc = cohs[group=='HC']
        
        if hemi == 'right':
            coh_contra_als = np.mean(cohs_als[:,right_mc, l_emg],axis=(0,1)) 
            coh_contra_hc = np.mean(cohs_hc[:,right_mc, l_emg],axis=(0,1))
            coh_ipsi_als = np.mean(cohs_als[:,right_mc, r_emg],axis=(0,1)) 
            coh_ipsi_hc = np.mean(cohs_hc[:,right_mc, r_emg],axis=(0,1))
            
        if hemi == 'left':
            coh_contra_als = np.mean(cohs_als[:,left_mc, r_emg],axis=(0,1)) 
            coh_contra_hc = np.mean(cohs_hc[:,left_mc, r_emg],axis=(0,1))
            coh_ipsi_als = np.mean(cohs_als[:,left_mc, l_emg],axis=(0,1)) 
            coh_ipsi_hc = np.mean(cohs_hc[:,left_mc, l_emg],axis=(0,1))
        
        plt.plot(f,coh_contra_als,label='ALS - contra')
        plt.plot(f,coh_contra_hc,label='HC - contra')
        plt.plot(f,coh_ipsi_als,label='ALS - ipsi',linestyle='dashdot',alpha=0.6)
        plt.plot(f,coh_ipsi_hc,label='HC - ipsi',linestyle='dashdot',alpha=0.6)
        plt.ylim([0.26,0.3])
        # plt.plot(f,coh_rmc_remg_als,label='ALS - ipsi', linestyle='dashdot')
        # plt.plot(f,coh_rmc_remg_hc,label='HC - ipsi',linestyle='dashdot')
        y_min, y_max = plt.gca().get_ylim()  # Get y-axis limits    
        if (pvalues[j] < 0.05).any()==True:
            if len(pvalues[j])>1:
                sig = np.where(pvalues[j]<0.05)
                mask = np.squeeze(np.array(clusters[j])[sig])
                pval = pvalues[j][np.where(pvalues[j]<0.05)]
            else:
                mask = np.squeeze(clusters[j])
                pval = pvalues[j]
            plt.fill_between(mask+min(f), y_min, y_max, alpha=0.2,color='red',label=f'pvalue = {float(pval):.3f}')
        else:
            print('no sig')
        plt.xlabel('frequency')
        plt.ylabel('CMC')
        # plt.title(f'CMC {hemi} motor cortex - {task} gripper task')
        plt.legend()
        plt.savefig(f'plots/cluster/cmc_{task}_{hemi}_grip.png',dpi=300)
        plt.show()
        print(clusters[j])
        j=j+1
        # print(mask)

