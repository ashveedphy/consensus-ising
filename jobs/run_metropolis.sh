#!/bin/bash
# Metropolis runs, one array task per disorder realisation (T = 2.44 ... 3.00).
# Set delta in simulation/run_metropolis.f90 and rebuild before submitting.
# Submit from the repository root (the folder that contains output/).
#SBATCH --job-name=metro_L50
#SBATCH --array=1-200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/metro_%A_%a.out
mkdir -p logs
./simulation/metropolis_ising "$SLURM_ARRAY_TASK_ID"
