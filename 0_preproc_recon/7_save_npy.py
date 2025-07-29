"""Save source reconstructed data as numpy files.

"""

import os
import mne
import numpy as np
import pandas as pd
from glob import glob

from dask.distributed import Client

output_dir = f"../../data/src_npy"
os.makedirs(output_dir, exist_ok=True)


src_dir = "/home/mtrubshaw/Documents/ALS_task/data/src"

parc_paths = []
subjects = []
for path in sorted(glob(f"{src_dir}/*/sflip_parc-raw.fif")):
    subject = path.split("/")[-2]
    subjects.append(subject)
    parc_paths.append(path)

if __name__ == "__main__":
    client = Client(n_workers=6, threads_per_worker=1)

    for path, subject in zip(parc_paths,subjects):
        raw = mne.io.read_raw_fif(path, verbose=False)
        raw.pick("misc")
        data = raw.get_data(reject_by_annotation="omit", verbose=False).T
        np.save(f"{output_dir}/{subject}.npy", data)
        print(subject)