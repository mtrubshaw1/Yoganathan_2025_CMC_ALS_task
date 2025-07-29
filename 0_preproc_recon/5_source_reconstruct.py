

"""Source reconstruction.

This includes beamforming, parcellation and orthogonalisation.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
from dask.distributed import Client
import pandas as pd

from osl import source_recon, utils

# Directories
preproc_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc"
coreg_dir = "/home/mtrubshaw/Documents/ALS_task/data/coreg"
src_dir = "/home/mtrubshaw/Documents/ALS_task/data/src"
fsl_dir = "/opt/ohba/fsl/6.0.5"  # this is where FSL is installed on hbaws

# Files
preproc_file = preproc_dir + "/{subject}_rest_tsss/{subject}_rest_tsss_preproc_raw.fif"  # {subject} will be replaced by the subject name



# Settings
config = """
    source_recon:
    - beamform_and_parcellate:
        freq_range: [1, 80]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""
#fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)

    # Copy directory containing the coregistration
    if not os.path.exists(src_dir):
        cmd = f"cp -r -u {coreg_dir} {src_dir}"
        print(cmd)
        os.system(cmd)

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
                preproc_files.append(f"{preproc_dir}/{task}_s0{subject}_raw_tsss/{task}_s0{subject}_tsss_preproc_raw.fif")
                subject_list.append(f'sub-{subject}_{task}')
        elif missing_task2[n] == "Yes":
            for task2 in tasks2:
                preproc_files.append(f"{preproc_dir}/{task2}_s0{subject}_raw_tsss/{task2}_s0{subject}_tsss_preproc_raw.fif")
                subject_list.append(f'sub-{subject}_{task2}')
        elif missing_task2[n] == "Mri":
            for task in tasks:
                preproc_files.append(f"{preproc_dir}/{task}_s0{subject}_raw_tsss/{task}_s0{subject}_tsss_preproc_raw.fif")
                subject_list.append(f'sub-{subject}_{task}')
        elif missing_task2[n] == "Yes_Mri":
            for task2 in tasks2:
                preproc_files.append(f"{preproc_dir}/{task2}_s0{subject}_raw_tsss/{task2}_s0{subject}_tsss_preproc_raw.fif")
                subject_list.append(f'sub-{subject}_{task2}')

    subject_list1 = []
    preproc_files1 = []
    for n, sub in enumerate(subject_list):
        if os.path.isfile(f'/home/mtrubshaw/Documents/ALS_task/data/src/{sub}/parc/parc-raw.fif') == False:
            subject_list1.append(subject_list[n])
            preproc_files1.append(preproc_files[n])
            
    print('***************')        
    print(f'Subjects to source reconstruct: ---- {len(subject_list1)}')
    print('***************')        
    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=10, threads_per_worker=1)

    # Source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subject_list1,
        preproc_files=preproc_files1,
        dask_client=False,
        )
