"""Example script for maxfiltering raw data recorded at Oxford.
Note: this script needs to be run on a computer with a MaxFilter license.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl.maxfilter import run_maxfilter_batch
import glob
import os


# raw_dir = "/home/mtrubshaw/Documents/ALS_task/data/raw/"

# input_files = glob.glob(raw_dir + "*.fif") 

input_files = ["/home/mtrubshaw/Documents/ALS_task/data/raw/task2_s05011_raw.fif",
"/home/mtrubshaw/Documents/ALS_task/data/raw/task2_s0336_raw.fif",
"/home/mtrubshaw/Documents/ALS_task/data/raw/task2_s0576_raw.fif",
"/home/mtrubshaw/Documents/ALS_task/data/raw/task2_s0293_raw.fif"]

# Directory to save the maxfiltered data to
os.makedirs('/home/mtrubshaw/Documents/ALS_task/data/maxfiltered',exist_ok=True)
output_directory = "/home/mtrubshaw/Documents/ALS_task/data/maxfiltered"

# Run MaxFiltering
run_maxfilter_batch(
    input_files,
    output_directory,

#Must specify --scanner below:
    "--maxpath /neuro/bin/util/maxfilter --scanner Neo --tsss --mode multistage --headpos --movecomp",


)

