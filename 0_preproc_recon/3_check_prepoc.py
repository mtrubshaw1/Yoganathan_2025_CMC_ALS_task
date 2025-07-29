#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:39:24 2024

@author: mtrubshaw
"""
import numpy as np

import pandas as pd
import glob

preproc_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc/"

input_files = glob.glob('/home/mtrubshaw/Documents/ALS_task/data/preproc/*tsss', recursive = True) 



maxfilter_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc"
preproc_dir = "../../data/preproc"

participants = pd.read_csv("../../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values

expected_files = []
tasks = ['task1','left','right','task2']
tasks2 = ['task1','left','right']
n_subjects = len(subjects)
for n, subject in enumerate(subjects):
    if missing_task2[n] == "No":
        for task in tasks:
            mf_file = f'{task}_s0{subject}_raw_tsss'
            expected_files.append(f"{maxfilter_dir}/"+mf_file)
    elif missing_task2[n] == "Yes":
        for task2 in tasks2:
            mf_file = f'{task2}_s0{subject}_raw_tsss'
            expected_files.append(f"{maxfilter_dir}/"+mf_file)




set_expected = set(expected_files)
set_real = set(input_files)


# Find common elements
common_elements = set_expected & set_real  

# Find elements missing in list2
missing_in_real = set_expected - set_real  

print(missing_in_real)