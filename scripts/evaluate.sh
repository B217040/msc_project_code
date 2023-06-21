#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --job-name=evaluate_tiny
#SBATCH --gres=gpu:1
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
module load python/3.10.8-gpu
source /work/tc046/tc046/pchamp/diss310/bin/activate
echo "modules are loaded"

COMMAND="python evaluation.py"

echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"