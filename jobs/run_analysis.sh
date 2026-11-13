#!/bin/bash
# Full analysis chain for every disorder strength (M = 100 configurations per student).
# Run from the repository root after all Monte Carlo runs have finished.
set -e
M=100
for d in 0.000 0.200 0.400 0.600 0.800 0.999; do
    base=output/System_Size_L50/delta_$d
    python analysis/jhat_estimation_by_student.py --base "$base" --n_samples $M
    python analysis/teacher_quantities.py        --base "$base"
    python analysis/consensus_analysis.py \
        --students "$base/students_M${M}_identity.dat" \
        --teachers "$base/teachers.dat" --delta "$d"
done
