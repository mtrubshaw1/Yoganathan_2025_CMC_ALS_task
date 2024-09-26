"""Plot results.

"""

import numpy as np
import pandas as pd
from osl_dynamics.analysis import power
import os

os.makedirs('plots/cmc_heat',exist_ok=True)
cohs = np.load('data/cohs_tasks_allregs.npy')
participants = pd.read_csv("../demographics/task_demographics.csv")
group = participants["Group"].values
f = np.load('data/f.npy')
cohs_remg = []
cohs_lemg = []
for r in range(52):
    cohs_remg.append(cohs[:,:,r,52,np.arange(3,17)])
    cohs_lemg.append(cohs[:,:,r,53,np.arange(3,17)])
cohs_remg = np.expand_dims(np.array(cohs_remg),axis=0)
cohs_lemg = np.expand_dims(np.array(cohs_lemg),axis=0)

cohs_emg = np.mean(np.concatenate((cohs_remg,cohs_lemg),axis=0),axis=-1).swapaxes(2,0).swapaxes(1,-1)

np.save('data/beta_cohs.npy',cohs_emg)
cohs_emg_als = np.mean(cohs_emg[group=='ALS'],axis=0)
cohs_emg_hc = np.mean(cohs_emg[group=='HC'],axis=0)



data = [cohs_emg_als,cohs_emg_hc]
tasks = ['bilat','right','left']
emgs = ['emgr', 'emgl']
for t, task in enumerate(tasks):
    for e, emg in enumerate(emgs):
        
        data_ = data[1]
        cope = data_[t,e]
        vmax = 0.28
        vmin = 0.265
        
        power.save(
            cope,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={
                "cmap": "seismic",
                "bg_on_data": 1,
                "darkness": 0.3,
                "alpha": 1,
                "views": ["lateral"],
                "vmax":vmax,
                "vmin":vmin,
                
        
            },
            filename=f"plots/cmc_heat/cmc_{task}_{emg}_hc.png",
        )
        
        data_ = data[0]
        cope = data_[t,e]
        
        power.save(
            cope,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={
                "cmap": "seismic",
                "bg_on_data": 1,
                "darkness": 0.3,
                "alpha": 1,
                "views": ["lateral"],
                "vmax":vmax,
                "vmin":vmin,
                
        
            },
            filename=f"plots/cmc_heat/cmc_{task}_{emg}_als.png",
        )