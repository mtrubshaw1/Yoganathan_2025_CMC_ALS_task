#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:20:49 2024

@author: mtrubshaw
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:01:49 2023

@author: okohl

Calculate HC vs. PD TFR-Power contrast.
Contrast performed with glmtools controlling for age and sex as confounds.

On medial motor cortex

"""

from glob import glob
import pickle
import numpy as np
import scipy.stats as stats
import mne
import glmtools as glm

from matplotlib import pyplot as plt
import pandas as pd


# ---------------------------------------------

# Gret Cluster forming threshold from model degrees of freedome
def get_cluster_forming_threshold(dof_model, alpha=0.05):
    return stats.t.ppf(1 - alpha/2, dof_model)
  
    

# Calculate Cluster Permutation tests of State TCs
def tc_ClusterPermutation_test(data, assignments, covariates, 
                               nPerm=1000,
                               metric='tstats',
                               cluster_forming_threshold=[],
                               pooled_dims=[1],
                               n_jobs=4):
    
    # --- Define Dataset for GLM -----    
    data = glm.data.TrialGLMData(data=data,
                                 **covariates,
                                 category_list=assignments,)
                                 
    # ----- Specify regressors and Contrasts in GLM Model -----
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='HC',rtype='Categorical',codes=1)
    DC.add_regressor(name='PD',rtype='Categorical',codes=2)
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    
    DC.add_contrast(name="HC > PD", values=[1, -1] + [0] * len(covariates))

    
    #  ---- Create design martix and fit model and grab tstats ----
    des = DC.design_from_datainfo(data.info)
    model = glm.fit.OLSModel(des,data)
    
    # -------------------------------------
    # Permutation Test Pooling Across States
    # ---------------------------------------
    
    # Calculate Cluster forming threshold from Mode Dof if not specified
    if not cluster_forming_threshold:
        cluster_forming_threshold = get_cluster_forming_threshold(model.dof_model)
      
    # Run Permutation Tests
    contrast = 0
    CP = glm.permutations.ClusterPermutation(des, data, contrast, nPerm,
                                            metric=metric,
                                            tail=0,
                                            cluster_forming_threshold=cluster_forming_threshold,
                                            pooled_dims=pooled_dims,
                                            nprocesses=n_jobs)

    
    # Get Cluster inndices and pvalues
    cluster_masks, cluster_stats = CP.get_sig_clusters(data, 95)    
    
    # Set Empty p and cluster inds in case no significant clusters
    if cluster_stats is None:
        pvalues = []
    
    elif len(cluster_stats) > 0: #len

    
        # get pvalues
        nulls = CP.nulls
        percentiles = stats.percentileofscore(nulls,abs(cluster_stats))
        pvalues = 1 - percentiles/100
        
    print(pvalues)
    return pvalues, cluster_masks, model.tstats, cluster_forming_threshold


# Function identifying cluster start and end points
def find_cluster_intervals(binary_vector):
    # Find indices where transitions from 0 to 1 occur
    start_indices = np.where(np.diff(binary_vector) > 0)[0]
    # Find indices where transitions from 1 to 0 occur and adjust for the end of intervals
    end_indices = np.where(np.diff(binary_vector) < 0)[0] - 1

    # Handle the case where the binary vector starts or ends with a true value
    if binary_vector[0] != 0:
        start_indices = np.insert(start_indices, 0, 0)
    if binary_vector[-1] != 0:
        end_indices = np.append(end_indices, len(binary_vector) - 1)

    # Combine start and end indices to form intervals
    intervals = list(zip(start_indices, end_indices))

    return intervals


# Calculate Cluster Permutation tests of State TCs
def get_ts(data, assignments, covariates):

    # --- Define Dataset for GLM -----    
    data = glm.data.TrialGLMData(data=data,
                                 **covariates,
                                 category_list=assignments,)
                                 
    # ----- Specify regressors and Contrasts in GLM Model -----
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='HC',rtype='Categorical',codes=1)
    DC.add_regressor(name='PD',rtype='Categorical',codes=2)
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    
    DC.add_contrast(name="HC > PD", values=[1, -1] + [0] * len(covariates))

    
    #  ---- Create design martix and fit model and grab tstats ----
    des = DC.design_from_datainfo(data.info)
    model = glm.fit.OLSModel(des,data)
    
    return model.tstats[-1]

# -------------------------------------------------------------------


cohs = np.load('data/cohs_tasks.npy',allow_pickle=True)
coh_rmc =np.expand_dims(np.mean(cohs[:,:,[0], 3],axis=(2)),axis=0)
coh_lmc= np.expand_dims(np.mean(cohs[:,:,[1], 2],axis=(2)),axis=0)

data_=np.concatenate((coh_rmc,coh_lmc),axis=0).swapaxes(0, 1).swapaxes(1, 2)[:,2,1]

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


# import glmtools
# --- GLMTOOLs-TFR Cluster Permutation Test ---    

covariates = {'Age': age, 'Gender':gender, 'Missing_struc':missing_struc}
category_list = category_list.astype(float)
# Do Cluster Permutation Tests
p, clu_inds, ts, th = tc_ClusterPermutation_test(data=data_,assignments=category_list, 
                                                 covariates=covariates, pooled_dims=[1], nPerm=1000)

# Sig Timewindow of clusters
# Zero Pad Clu_inds to correct time points in trial
# clu_inds = np.pad(clu_inds,((0,0),(250,250)))

# clu_wind = []
# for i,ip in enumerate(p):
#     binary_vector = (np.sum(clu_inds == i+1, axis=0) > 0).astype(int)
#     clu_wind.append(find_cluster_intervals(binary_vector))
    
#     print(f'Cluster Sample Range: {clu_wind[i]}; p: {ip}')
    
# # Sig Frequency Range  
# clu_freq = []
# for i,ip in enumerate(p):
#     binary_vector = (np.sum(clu_inds == i+1, axis=1) > 0).astype(bool)
#     cluf = freqs[binary_vector]
#     clu_freq.append([cluf[0],cluf[-1]])
    
#     print(f'Cluster Freq Range: {clu_freq[0][0]}-{clu_freq[0][1]}Hz; p: {ip}')  

# # Get Group Contrasts T-Stats for whole trial length for plotting
# t_all = get_ts(motor_power,assignments,covariates)

# # --- Make TFR Plot ---   
 
# # Set Up
# fig, ax = plt.subplots(figsize = (10,6))
# extent=(-1, 5, 8, 35) 

# # Plot
# im = ax.imshow(motor_power_cont, cmap = 'RdBu_r', extent=extent, origin="lower", aspect="auto", vmin = vmin, vmax = vmax);

# # Grip On-set on Off-Set
# ax.axvline(x = 0, linewidth = 1, linestyle ="--", color ='grey')
# ax.axvline(x = 3, linewidth = 1, linestyle ="--", color ='grey')

# # Agg significant contour
# if len(p) > 0: 
#     for iClu in range(len(p)):
#         big_mask = np.kron(np.squeeze(clu_inds == iClu+1), np.ones((10,10))) #interpolate to 10x real data to fix contours
#         ax.contour(big_mask, colors='black', extent=extent,linewidths=.1, corner_mask=False, antialiased=False)
    
# # Grip On-set on Off-Set
# ax.axvline(x = 0, linewidth = 1, linestyle ="--", color ='grey')
# ax.axvline(x = 3, linewidth = 1, linestyle ="--", color ='grey')

# # Ticks
# ax.set_yticks([10,20,30])
# ax.set_yticklabels([10,20,30],)

# # Ticks
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)

# # Add Labels
# ax.set_ylabel('Frequency',fontsize=16)
# ax.set_xlabel('Times (sec)',fontsize=16)

# # Despine
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Color Bar
# h = plt.colorbar(im)
# h.ax.set_ylabel('Power (HC - PD)', rotation=-90, va='bottom', fontsize = 14)

   
# plt.tight_layout()
# plt.show()
    
# # Save Fig
# fig.savefig(plot_dir + 'TFR_contrast.svg', format = 'svg', bbox_inches='tight', transparent=True)


# #%% ----------------------------
# # Make TFR plot with T-statistics

# # Values for Plotting
# peak_t = np.max(abs(t_all))
# vmin, vmax = -peak_t,peak_t

# # Set Up
# fig, ax = plt.subplots(figsize = (10,6))
# extent=(-1, 5, 8, 35) 

# # Plot
# im = ax.imshow(t_all.squeeze(), cmap = 'RdBu_r', extent=extent, origin="lower", aspect="auto", vmin = vmin, vmax = vmax);

# # Grip On-set on Off-Set
# ax.axvline(x = 0, linewidth = 1, linestyle ="--", color ='grey')
# ax.axvline(x = 3, linewidth = 1, linestyle ="--", color ='grey')

# # Agg significant contour
# if len(p) > 0: 
#     for iClu in range(len(p)):
#         big_mask = np.kron(np.squeeze(clu_inds == iClu+1), np.ones((10,10))) #interpolate to 10x real data to fix contours
#         ax.contour(big_mask, colors='black', extent=extent,linewidths=.8, corner_mask=False, antialiased=False)
    
# # Grip On-set on Off-Set
# ax.axvline(x = 0, linewidth = 4, linestyle ="--", color ='grey')
# ax.axvline(x = 3, linewidth = 4, linestyle ="--", color ='grey')

# # Ticks
# ax.set_yticks([10,20,30])
# ax.set_yticklabels([10,20,30],)

# # Ticks
# ax.tick_params(axis='x', labelsize= 14)
# ax.tick_params(axis='y', labelsize= 14)

# # Add Labels
# ax.set_ylabel('Frequency',fontsize=18, labelpad=8)
# ax.set_xlabel('Time (sec)',fontsize=18)

# # Despine
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Color Bar
# h = plt.colorbar(im)
# h.ax.set_ylabel('T-Statistic (HC vs PD)', rotation=-90, va='bottom', fontsize = 14)

   
# plt.tight_layout()
# plt.show()
    
# # Save Fig
# fig.savefig(plot_dir + 'tstats_TFR_contrast.svg', format = 'svg', bbox_inches='tight', transparent=True)