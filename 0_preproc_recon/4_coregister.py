"""Coregistration.

"""

import numpy as np
import pandas as pd
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

fsl_dir = "/opt/ohba/fsl/6.0.5"

## if using standard brain (missing structural) remember to add allow_smri_scaling: True to config under coregister

def fix_headshape_points(src_dir, subject, preproc_file, smri_file, epoch_file):
    filenames = source_recon.rhino.get_coreg_filenames(src_dir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(
        hs[0] < lpa[0] - 5,
        np.logical_or(
            hs[0] > rpa[0] + 5,
            hs[1] > nas[1] + 5,
        ),
    )
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)
    client = Client(n_workers=10, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - fix_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
        - forward_model:
            model: Single Layer
    """

    preproc_dir = "/home/mtrubshaw/Documents/ALS_task/data/preproc"
    smri_dir = "/home/mtrubshaw/Documents/ALS_task/data/smri"
    coreg_dir = "/home/mtrubshaw/Documents/ALS_task/data/coreg"
    
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
                smri_files.append(f'{smri_dir}/s0{smris[n]}.nii')
                subject_list.append(f'sub-{subject}_{task}')
        elif missing_task2[n] == "Yes":
            for task2 in tasks2:
                preproc_files.append(f"{preproc_dir}/{task2}_s0{subject}_raw_tsss/{task2}_s0{subject}_tsss_preproc_raw.fif")
                smri_files.append(f'{smri_dir}/s0{smris[n]}.nii')
                subject_list.append(f'sub-{subject}_{task2}')
        elif missing_task2[n] == "Mri":
            for task in tasks:
                preproc_files.append(f"{preproc_dir}/{task}_s0{subject}_raw_tsss/{task}_s0{subject}_tsss_preproc_raw.fif")
                smri_files.append(f'{smri_dir}/{smris[n]}.nii.gz')
                subject_list.append(f'sub-{subject}_{task}')
        elif missing_task2[n] == "Yes_Mri":
            for task2 in tasks2:
                preproc_files.append(f"{preproc_dir}/{task2}_s0{subject}_raw_tsss/{task2}_s0{subject}_tsss_preproc_raw.fif")
                smri_files.append(f'{smri_dir}/{smris[n]}.nii.gz')
                subject_list.append(f'sub-{subject}_{task2}')


    preproc_files = preproc_files[67:]
    smri_files = smri_files[67:]
    subject_list = subject_list[67:]

    source_recon.run_src_batch(
        config,
        src_dir=coreg_dir,
        subjects=subject_list,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
    )
