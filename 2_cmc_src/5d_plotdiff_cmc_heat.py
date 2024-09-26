"""Plot results.

"""

import numpy as np
import pandas as pd
from osl_dynamics.analysis import power
import os

os.makedirs('plots/cmc_heat_diff',exist_ok=True)
cohs = np.load('data/cohs_tasks_allregs.npy')
participants = pd.read_csv("../demographics/task_demographics.csv")
group = participants["Group"].values

data = np.load('data/contrast_0.npy')
pvals = np.load('data/contrast_0_pvalues.npy')

tasks = ['bilat','right','left']
emgs = ['emgr', 'emgl']
for t, task in enumerate(tasks):
    for e, emg in enumerate(emgs):
        
        cope = data[t,e]
        vmax = 0.05
        vmin = -0.05
        
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
            filename=f"plots/cmc_heat_diff/cmc_{task}_{emg}_diff.png",
        )
        
        pval = pvals[t,e]
        
        power.save(
            pval,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
            plot_kwargs={
                "cmap": "Greens_r",
                "bg_on_data": 1,
                "darkness": 0.3,
                "alpha": 1,
                "views": ["lateral"],
                "vmax":0.1,
                "vmin":0,
                
        
            },
            filename=f"plots/cmc_heat_diff/cmc_{task}_{emg}_pval.png",
        )