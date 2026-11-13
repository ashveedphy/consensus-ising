#!/bin/bash
# Wolff runs, one array task per disorder realisation (T = 1.60 ... 2.40).
# Set delta in simulation/wolff.f90 and rebuild before submitting.
# Submit from the repository root (the folder that contains output/).
#SBATCH --job-name=wolff_L50
#SBATCH --array=1-200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/wolff_%A_%a.out
mkdir -p logs
./simulation/wolff_ising "$SLURM_ARRAY_TASK_ID"
