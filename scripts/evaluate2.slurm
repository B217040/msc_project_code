#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --job-name=evaluate_tiny
#SBATCH --time=12:00:00
#SBATCH --output=/work/tc046/tc046/pchamp/slurm_logs/evaluate_tiny_%A.txt
#SBATCH --account=tc046

# =====================
# Logging information
# =====================

echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# Load the required modules
echo "modules are loaded"

python evaluation.py



# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"