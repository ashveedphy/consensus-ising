# Consensus in Effective Model Inference for Disordered Ising Systems

Simulation and analysis code for

> A. Ashraf, *Consensus in Effective Model Inference for Disordered Ising Systems*, arXiv:2609.10731.

Independent students each infer a single uniform effective coupling from a
finite set of equilibrium configurations of one quenched realisation (teacher)
of the two-dimensional random-bond Ising model. The spread of the inferred
couplings (the consensus distribution) is decomposed into a finite-sampling
and a quenched-disorder contribution and compared with the Fisher information,
the Cramér–Rao bound, the bias and the mean squared error.

## Repository layout

```
consensus-ising/
├── bonds/
│   └── generate_bonds.py              quenched bond realisations Jx.dat, Jy.dat
├── simulation/
│   ├── wolff.f90                      Wolff cluster Monte Carlo   (T = 1.60 ... 2.40)
│   ├── run_metropolis.f90             Metropolis Monte Carlo      (T = 2.44 ... 3.00)
│   └── compile.sh                     compile commands
├── analysis/
│   ├── jhat_estimation_by_student.py  student inference of J_hat
│   ├── teacher_quantities.py          per-teacher J*, Fisher information, b'
│   └── consensus_analysis.py          variances, bias, CR bound, MSE
├── jobs/                              example SLURM / shell scripts
├── figures/                           figures of the paper
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Requirements

* Python ≥ 3.8 with NumPy (`pip install -r requirements.txt`)
* gfortran (any recent version)

## Workflow

All steps are run from the repository root; every step reads and writes the
folder `output/`.

### 1. Quenched disorder

```bash
python bonds/generate_bonds.py
```

Creates `output/System_Size_L50/delta_<δ>/realisation_<r>/Jx.dat, Jy.dat` for
δ ∈ {0.000, 0.200, 0.400, 0.600, 0.800, 0.999} and r = 1 … 200, with bonds
J_ij ~ U(J0 − δ, J0 + δ), J0 = 1 (Eq. 4). Each realisation has a fixed seed
derived from base seed 123456, δ and r, so the bonds are reproducible.
Jx(i,j) couples site (i,j) to (i+1,j); Jy(i,j) couples (i,j) to (i,j+1).

### 2. Monte Carlo sampling (Appendix A)

```bash
cd simulation && ./compile.sh && cd ..
./simulation/wolff_ising <r>          # one realisation r per job
./simulation/metropolis_ising <r>
```

Each program reads the bonds of realisation `r` and, for every temperature,
writes to `output/System_Size_L50/delta_<δ>/realisation_<r>/T_<T>/`:

| file | content |
|---|---|
| `output.dat` | one line per stored configuration: the 32 neighbourhood-pattern counts `n+(0) n-(0) … n+(15) n-(15)` (Appendix B) |
| `obs.dat` | energy and magnetisation of every stored configuration |
| `correlation.dat` | spin–spin correlation function |
| `meta.txt` | run parameters |

Pattern index x = 8·up + 4·down + 2·left + right, with spin −1 → 0, +1 → 1.

Run parameters used for L = 50:

| | Wolff (`wolff.f90`) | Metropolis (`run_metropolis.f90`) |
|---|---|---|
| temperatures | 1.60 – 2.40, ΔT = 0.04 (21 values) | 2.44 – 3.00, ΔT = 0.04 (15 values) |
| equilibration | 3 × 10⁴ cluster updates | 5 × 10⁴ sweeps |
| interval between stored configurations | 2.5 × 10⁴ cluster updates | 2 × 10³ sweeps |
| measurement length | 6.25 × 10⁸ cluster updates | 5 × 10⁷ sweeps |
| configurations stored, N_tot | 25 000 | 25 000 |

The disorder strength is set by the parameter `delta` at the top of each
`.f90` file; set it to the required value and rebuild before running each δ.
`jobs/run_wolff.sh` and `jobs/run_metropolis.sh` submit the 200 realisations
as a SLURM array.

### 3. Analysis

```bash
python analysis/jhat_estimation_by_student.py --base output/System_Size_L50/delta_0.500 --n_samples 100
python analysis/teacher_quantities.py         --base output/System_Size_L50/delta_0.500
python analysis/consensus_analysis.py \
       --students output/System_Size_L50/delta_0.500/students_M100_identity.dat \
       --teachers output/System_Size_L50/delta_0.500/teachers.dat --delta 0.5
```

`jobs/run_analysis.sh` runs these three steps for every δ.

| script | computes | paper |
|---|---|---|
| `jhat_estimation_by_student.py` | Splits the N_tot configurations of each teacher among students without replacement (M per student) and infers J_hat and the residual complexity of every student by kernel ridge regression | Eqs. (13)–(16), App. E |
| `teacher_quantities.py` | From all configurations of each teacher: J*, Fisher information β²Var(E), linear-response b′, bond mean, pattern weights N(x) | Eqs. (34), (37), (46) |
| `consensus_analysis.py` | Sampling, disorder and consensus variances; bias; b′; Cramér–Rao bound; efficiency; MSE; split of the disorder variance; characteristic temperatures | Eqs. (21)–(27), (31), (38), (41), (49), (56), (D19) |

The sample-size dependence (Figs. 3d and 5b) is obtained by repeating the
analysis with different `--n_samples`.

### 4. Figures

The figures of the paper are in `figures/`.

## Data

The bond realisations, the inferred couplings of all students and teachers and
the data underlying the figures are available on Zenodo:
https://doi.org/10.5281/zenodo.XXXXXXX.
The full set of configurations (about 140 GB) can be regenerated with this code
and is available from the author on reasonable request.

## Citation

Please cite the article above if you use this code (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
