"""Preprocess MaxFiltered sensor data.

"""

import pandas as pd
from dask.distributed import Client

from osl import preprocessing, utils
import os
import glob

os.makedirs(f'../../data/preproc',exist_ok=True)


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=6, threads_per_worker=1)

    config = """
        preproc:
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
        - bad_channels: {picks: grad, significance_level: 0.1}
        - interpolate_bads: {}
    """
    # %% preproc all data
    
    maxfilter_dir = "../../data/maxfiltered"
    preproc_dir = "../../data/preproc"

    participants = pd.read_csv("../../demographics/task_demographics.csv")

    subjects = participants["Subject"].values
    missing_task2 = participants["Missing_task2"].values

    maxfiltered_files = []
    tasks = ['task1','left','right','task2']
    tasks2 = ['task1','left','right']
    n_subjects = len(subjects)
    for n, subject in enumerate(subjects):
        if missing_task2[n] == "No":
            for task in tasks:
                mf_file = f'{task}_s0{subject}_raw_tsss.fif'
                maxfiltered_files.append(f"{maxfilter_dir}/"+mf_file)
            else:
                for task2 in tasks2:
                    mf_file = f'{task2}_s0{subject}_raw_tsss.fif'
                    maxfiltered_files.append(f"{maxfilter_dir}/"+mf_file)


    #%% Onlypreproc missing data
    # preproc_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc/"

    # input_files = glob.glob('/home/mtrubshaw/Documents/ALS_task/data/preproc/*tsss/', recursive = True) 



    # maxfilter_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc"
    # preproc_dir = "../../data/preproc"

    # participants = pd.read_csv("../../demographics/task_demographics.csv")

    # subjects = participants["Subject"].values
    # missing_task2 = participants["Missing_task2"].values

    # expected_files = []
    # tasks = ['task1','left','right','task2']
    # tasks2 = ['task1','left','right']
    # n_subjects = len(subjects)
    # for n, subject in enumerate(subjects):
    #     if missing_task2[n] == "No":
    #         for task in tasks:
    #             mf_file = f'{task}_s0{subject}_raw_tsss'
    #             expected_files.append(f"{maxfilter_dir}/"+mf_file)
    #         else:
    #             for task2 in tasks2:
    #                 mf_file = f'{task2}_s0{subject}_raw_tsss'
    #                 expected_files.append(f"{maxfilter_dir}/"+mf_file)




    # set_expected = set(expected_files)
    # set_real = set(input_files)


    # # Find common elements
    # common_elements = set_expected & set_real  

    # # Find elements missing in list2
    # missing_in_real = set_expected - set_real  
    
    # missing_in_real = [missing.replace('/home/mtrubshaw/Documents/ALS_task/data/preproc/','')for missing in missing_in_real]
    # missing_in_real = [missing.replace('tsss','tsss.fif')for missing in missing_in_real]
    # missing_in_real = missing_in_real
    # print(missing_in_real)
    
    # maxfiltered_files = []
    # max_dir = "/home/mtrubshaw/Documents/ALS_task/data/maxfiltered"
    # for t in range(len(missing_in_real)):
    #     maxfiltered_files.append(f'{max_dir}/{missing_in_real[t]}')

    #%%
    preprocessing.run_proc_batch(
        config,
        maxfiltered_files,
        outdir=preproc_dir,
        overwrite=True,
        dask_client=True,
    )
