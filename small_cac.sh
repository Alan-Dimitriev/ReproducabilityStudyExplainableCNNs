#!/bin/bash
#SBATCH -c 6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=80G        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=80_small_run_cac.out 
#SBATCH --partition=standard

echo "Starting to run"
source activate condaenv867 # use anaconda to load the library
echo "about to run cac"
python cac.py
echo "finished running"
