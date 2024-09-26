#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:23:04 2024

@author: mtrubshaw
"""

import mne
import numpy as np
from osl.preprocessing import osl_wrappers
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt


# Directories
src_dir = Path("/home/mtrubshaw/Documents/ALS_task/data/src")
preproc_dir = Path("/home/mtrubshaw/Documents/ALS_task/data/preproc")
epoch_dir = Path("/ohba/pi/knobre/mtrubshaw/ALS_task/data/emg_grip_epoched_sensor")
os.makedirs('data',exist_ok=True)
os.makedirs(epoch_dir,exist_ok=True)

participants = pd.read_csv("../demographics/task_demographics.csv")

subjects = participants["Subject"].values
smris = participants["MRI_no"].values
missing_task2 = participants["Missing_task2"].values

preproc_files = []
smri_files = []
subject_list = []
parc_list = []
tasks = ['task1','left','right','task2']
tasks2 = ['task1','left','right']
tasks1 = ['left','right','task2']
n_subjects = len(subjects)
for n, subject in enumerate(subjects):
    if missing_task2[n] == "No":
        for task in tasks:
            subject_list.append(f'{task}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes":
        for task2 in tasks2:
            subject_list.append(f'{task2}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task2}')
    elif missing_task2[n] == "Mri":
        for task in tasks:
            subject_list.append(f'{task}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes_Mri":
        for task2 in tasks2:
            subject_list.append(f'{task2}_s0{subject}')
            parc_list.append(f'sub-{subject}_{task2}')

# from dask.distributed import Client

# if __name__ == "__main__":

#     client = Client(n_workers=10, threads_per_worker=1)
#     os.makedirs('data',exist_ok=True)

# # Fif files containined parcellated data
for subject, parc in zip(subject_list,parc_list):
    preproc_file = preproc_dir / f'{subject}_raw_tsss/{subject}_tsss_preproc_raw.fif'
    parc_file = src_dir / f'{parc}/sflip_parc-raw.fif'
    if not preproc_file.exists():
        print(f'File not found: {preproc_file}')
        continue
    # Read continuous parcellated data
    raw = mne.io.read_raw_fif(preproc_file, preload=True)
    # Find events
    events = mne.find_events(raw, stim_channel='STI101', min_duration=0.005)
    # Define event mappings
    event_id_dict = {f'Event_{int(event_id)}': int(event_id) for event_id in np.unique(events[:, 2])}
    # Epoching
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-1, tmax=5,  baseline=None, preload=True)

    # # Drop bad segments
    # epochs = osl_wrappers.drop_bad_epochs(epochs, picks='misc', metric='var')
    # epochs_parc = osl_wrappers.drop_bad_epochs(epochs_parc, picks='misc', metric='var')
    # Save epochs in a separate directory
    epoch_file = epoch_dir / f'{subject}_emg_grip_epo.fif'
    print(f'Saving: {epoch_file}')
    epochs.save(epoch_file, overwrite=True)
    
print(raw.info.ch_names.__contains__)
