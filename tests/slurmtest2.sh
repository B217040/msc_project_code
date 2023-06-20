#!/bin/bash
# SBATCH --partition=gpu
# SBATCH --gres=gpu:1
# SBATCH --qos=gpu
# SBATCH --job-name=slurm_test
# SBATCH --gres=gpu:1
# SBATCH --time=12:00:00
# SBATCH --output=slurm_test_log-%A.txt
# SBATCH --account=tc046
module load python/3.10.8-gpu
echo Running test python script
python slurmtest.py
date
echo job done