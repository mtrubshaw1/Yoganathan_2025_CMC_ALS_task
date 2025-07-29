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

input_files = glob.glob('/home/mtrubshaw/Documents/ALS_task/data/smri/*', recursive = True) 
input_files = [s.replace('/home/mtrubshaw/Documents/ALS_task/data/smri/s0','')for s in input_files]


maxfilter_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc"
preproc_dir = "../../data/preproc"

participants = pd.read_csv("../../demographics/task_demographics.csv")

subjects = participants["Subject"].values
missing_task2 = participants["Missing_task2"].values

expected_files = []
for subject in subjects:
    expected_files.append(f'{subject}.nii')


set_expected = set(expected_files)
set_real = set(input_files)

# print(input_files)
# Find common elements
common_elements = set_expected & set_real  

# Find elements missing in list2
missing_in_real = set_expected - set_real  

print(missing_in_real)