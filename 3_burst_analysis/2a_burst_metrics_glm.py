#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:16:06 2024

@author: mtrubshaw
"""

"""Fit a GLM and perform statistical significance testing.

"""

import numpy as np
import os
import pandas as pd
from scipy import stats

import glmtools as glm
from osl_dynamics.analysis import power

os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('plots/compiled_plots', exist_ok=True)
# Load target data



# Load regressor data
demographics = pd.read_csv("../demographics/task_demographics.csv")

grip_r = demographics["Grip_strength_r"].values
grip_l = demographics["Grip_strength_l"].values

category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "PLS"] = 3
category_list[category_list == "rPLS"] = 3
category_list[category_list == "FDR"] = 3
category_list[category_list == "rFDR"] = 3

grip = np.mean(np.vstack((grip_r,grip_l)),axis=0)
beta_rest = np.load('data/beta_rest.npy')
beta_15 = np.load('data/beta_15.npy')
burst_FOs = np.load('data/burst_FOs.npy')
burst_rates = np.load('data/burst_rates.npy')
burst_meanLTs = np.load('data/burst_meanLTs.npy',)
burst_amps = np.load('data/burst_amps.npy')


data = np.vstack((burst_FOs,burst_meanLTs,burst_amps,beta_15)).T





category_list = category_list

uniques = np.unique(category_list)

age = (demographics["Age"].values)
gender = []
for g in demographics["Sex"].values:
    if g == "M":
        gender.append(0)
    else:
        gender.append(1)
gender = np.array(gender)


missing_struc = (demographics["Missing_struc"].values)


# Create GLM dataset
data = glm.data.TrialGLMData(
    data=data,
    category_list=category_list,
    age=age,
    gender=gender,
    dim_labels=["Subjects", "Frequencies", "Parcels"],
    missing_struc=missing_struc,
    grip=grip,
)

# Design matrix
DC = glm.design.DesignConfig()
DC.add_regressor(name="HC", rtype="Categorical", codes=1)
DC.add_regressor(name="ALS", rtype="Categorical", codes=2)
# DC.add_regressor(name="FDR", rtype="Categorical", codes=3)
DC.add_regressor(name="Sex", rtype="Parametric", datainfo="gender", preproc="z")
DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
DC.add_regressor(name="Missing Structural", rtype="Parametric", datainfo="missing_struc", preproc="z")
# DC.add_regressor(name="Grip", rtype="Parametric", datainfo="grip", preproc="z")


DC.add_contrast(name="ALS-HC", values=[-1, 1, 0, 0, 0])
# DC.add_contrast(name="grip", values=[0, 0, 0, 0, 0,1])
# DC.add_contrast(name="FDR-HC", values=[-1, 0, 1, 0, 0, 0])
# DC.add_contrast(name="ALS-FDR", values=[0, 1, -1, 0, 0, 0])


design = DC.design_from_datainfo(data.info)
design.plot_summary(savepath="plots/glm_design.png", show=False)
design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

# Fit the GLM
model = glm.fit.OLSModel(design, data)

def do_stats(contrast_idx, metric="tstats"):
    # Max-stat permutations
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1),  # pool over channels
        nprocesses=16,
    )
    null_dist = perm.nulls

    # Calculate p-values
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = (1 - percentiles / 100)/2

    return pvalues

for i in range(model.copes.shape[0]):
    cope = model.copes[i]
    pvalues = do_stats(contrast_idx=i)
    tstats = model.tstats
    print(cope)
    print(pvalues)
    np.save(f"data/contrast_{i}.npy", cope)
    np.save(f"data/tstats.npy", tstats)
    np.save(f"data/contrast_{i}_pvalues.npy", pvalues)

for unique in uniques:
    count = np.count_nonzero(category_list==unique)
    print('Group',unique,' - ',count)
