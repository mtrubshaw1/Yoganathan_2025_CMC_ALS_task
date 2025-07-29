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


# Directories
src_dir = Path("../../data/src")
epoch_dir = Path("../../data/src_epoched")
epoch_dir.mkdir(exist_ok=True, parents=True)  # Create the directory if it doesn’t exist



participants = pd.read_csv("../../demographics/task_demographics.csv")

subjects = participants["Subject"].values
smris = participants["MRI_no"].values
missing_task2 = participants["Missing_task2"].values

preproc_files = []
smri_files = []
subject_list = []
tasks = ['task1','left','right','task2']
tasks2 = ['task1','left','right']
tasks1 = ['left','right','task2']
n_subjects = len(subjects)
for n, subject in enumerate(subjects):
    if missing_task2[n] == "No":
        for task in tasks:
            subject_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes":
        for task2 in tasks2:
            subject_list.append(f'sub-{subject}_{task2}')
    elif missing_task2[n] == "Mri":
        for task in tasks:
            subject_list.append(f'sub-{subject}_{task}')
    elif missing_task2[n] == "Yes_Mri":
        for task2 in tasks2:
            subject_list.append(f'sub-{subject}_{task2}')



# Fif files containined parcellated data
for subject in subject_list:
    parc_file = src_dir / f'{subject}/sflip_parc-raw.fif'
    if not parc_file.exists():
        print(f'File not found: {parc_file}')
        continue
    # Read continuous parcellated data
    raw = mne.io.read_raw_fif(parc_file, preload=True)
    # Find events
    events = mne.find_events(raw, stim_channel='STI101', min_duration=0.005)
    # Define event mappings
    event_id_dict = {f'Event_{int(event_id)}': int(event_id) for event_id in np.unique(events[:, 2])}
    picks = mne.pick_channels(raw.info['ch_names'], include=['parcel_%d' % i for i in range(52)] + ['STI101', 'EMG004', 'EMG005', 'MISC007', 'MISC008'])
    # Epoching
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-1, tmax=5, picks=picks, baseline=None, preload=True)
    # Drop bad segments
    epochs = osl_wrappers.drop_bad_epochs(epochs, picks='misc', metric='var')
    # Save epochs in a separate directory
    epoch_file = epoch_dir / f'{subject}_sflip_parc-epo.fif'
    print(f'Saving: {epoch_file}')
    # epochs.save(epoch_file, overwrite=True)
    
# # Fif files containined parcellated data
# subject = "sub-293_left"

# parc_file = src_dir / f'{subject}/sflip_parc-raw.fif'

# # Read continuous parcellated data
# raw = mne.io.read_raw_fif(parc_file, preload=True)

# # Find events
# events = mne.find_events(raw, stim_channel='STI101', min_duration=0.005)

# # Define event mappings
# event_id_dict = {f'Event_{int(event_id)}': int(event_id) for event_id in np.unique(events[:, 2])}

# # Define picks (include parcel channels and other necessary channels)
# picks = mne.pick_channels(raw.info['ch_names'], include=['parcel_%d' % i for i in range(52)] + ['STI101', 'EMG004', 'EMG005', 'MISC007', 'MISC008'])

# # Epoching
# epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-1, tmax=5, picks=picks, baseline=None, preload=True)

# # Drop bad segments (assuming osl_wrappers.drop_bad_epochs is correctly implemented)
# epochs = osl_wrappers.drop_bad_epochs(epochs, picks='misc', metric='var')

# # Plot epochs with specified picks and display events

# epochs.plot(picks=[9,20], n_epochs=3, events = events, event_id = event_id_dict)
