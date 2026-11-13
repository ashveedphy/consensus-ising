#!/bin/bash
# Compile the two Monte Carlo codes
gfortran -O3 -fopenmp -o wolff_ising wolff.f90
gfortran -O3 -fopenmp -o metropolis_ising run_metropolis.f90
