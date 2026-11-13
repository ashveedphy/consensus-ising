#!/usr/bin/env python3
"""
jhat_estimation_by_student.py
=============================

Inference of the effective uniform coupling J_hat by an ensemble of students
(teacher-student framework, Sec. III of the paper).

Each teacher is one quenched bond realisation omega at one temperature T,
represented by its equilibrium configurations in
        realisation_XXXX/T_Y.YYY/output.dat
(N_tot rows; each row holds the 32 pattern counts of one configuration,
interleaved as n+(0) n-(0) n+(1) n-(1) ... n+(15) n-(15)).

The number of configurations a student reads is controlled by --n_samples (M).
The configurations of a teacher are split among the students WITHOUT
replacement: the rows are shuffled once and cut into disjoint blocks of M, so
no configuration is shared by two students.

For every student (Sec. III B):
  1. Sum the pattern counts of its M configurations -> n+(x), n-(x), N(x)
  2. Smoothed conditional probability (Jeffreys prior, alpha = 0.5)    Eq. (14)
        P(+1 | x) = (n+(x) + alpha) / (N(x) + 2 alpha)
  3. Empirical local field                                              Eq. (13)
        f~(x) = 0.5 ln[ P / (1 - P) ]
  4. Kernel ridge regression of f~ with pattern weights N(x)            App. E
        minimise  sum_x N(x) [f~(x) - f*(x)]^2 + lambda c^T K c,  f* = K c
  5. Projection onto the uniform Ising model                            Eq. (16)
        J_hat = (1/beta) sum_x N(x) S(x) f*(x) / sum_x N(x) S(x)^2
  6. Residual complexity                                                Eq. (E11)
        RC = sqrt( sum_x N(x) [f*(x) - beta J_hat S(x)]^2 / sum_x N(x) )

Output (one row per student):
    teacher  student  temperature  n_samples  J_hat  RC

Usage:
    python jhat_estimation_by_student.py --base delta_0.500 --n_samples 100
    python jhat_estimation_by_student.py --base delta_0.500 --n_samples 100 --kernel linear
"""

import argparse
import glob
import os
import re

import numpy as np

# ------------------------------------------------------------------ patterns
# phi[x, k] = spin (+1/-1) of neighbour k in pattern x;  S(x) = sum of neighbours
PHI = np.array([[2 * ((x >> k) & 1) - 1 for k in range(4)] for x in range(16)], float)
S = PHI.sum(axis=1)                                     # {-4, -2, 0, 2, 4}


def kernel_matrix(name):
    """16 x 16 kernel K(x, x') over the neighbourhood patterns (App. E)."""
    if name == "identity":                 # every pattern independent (used in the paper)
        return np.eye(16)
    if name == "linear":                   # K = phi(x) . phi(x')
        return PHI @ PHI.T
    if name == "isotropic":                # K = S(x) S(x')
        return np.outer(S, S)
    if name == "rbf":                      # K = exp(-|phi(x) - phi(x')|^2 / 2 l^2), l = 2
        d2 = ((PHI[:, None, :] - PHI[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * 2.0 ** 2))
    raise ValueError(f"unknown kernel {name}")


# ----------------------------------------------------------------- inference
def infer_coupling(counts, T, K, alpha=0.5, lam=0.0):
    """J_hat and RC from the summed counts (32,) of one student."""
    beta = 1.0 / T
    n_plus, n_minus = counts[0::2], counts[1::2]
    N = n_plus + n_minus                                        # pattern weights

    P = (n_plus + alpha) / (N + 2 * alpha)                      # Eq. (14)
    f_tilde = 0.5 * np.log(P / (1.0 - P))                       # Eq. (13)

    # weighted kernel ridge regression:  (K W K + lam K) c = K W f~
    W = np.diag(N)
    c = np.linalg.lstsq(K @ W @ K + lam * K, K @ W @ f_tilde, rcond=None)[0]
    f_star = K @ c

    beta_J = np.sum(N * S * f_star) / np.sum(N * S ** 2)        # Eq. (16)
    RC = np.sqrt(np.sum(N * (f_star - beta_J * S) ** 2) / np.sum(N))   # Eq. (E11)
    return beta_J / beta, RC


def split_without_replacement(n_tot, n_samples, n_students, rng):
    """Disjoint index blocks: student s gets rows blocks[s] (no overlaps)."""
    perm = rng.permutation(n_tot)
    return perm[: n_students * n_samples].reshape(n_students, n_samples)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Student inference of the effective coupling")
    ap.add_argument("--base", default=".", help="delta folder containing realisation_* folders")
    ap.add_argument("--n_samples", type=int, required=True,
                    help="number of configurations each student reads (M)")
    ap.add_argument("--n_students", type=int, default=None,
                    help="students per teacher (default: N_tot // n_samples)")
    ap.add_argument("--kernel", default="identity",
                    choices=["identity", "linear", "isotropic", "rbf"])
    ap.add_argument("--lam", type=float, default=0.0, help="ridge parameter lambda")
    ap.add_argument("--alpha", type=float, default=0.5, help="Jeffreys smoothing")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    K = kernel_matrix(a.kernel)
    out = a.out or os.path.join(a.base, f"students_M{a.n_samples}_{a.kernel}.dat")
    real_dirs = sorted(glob.glob(os.path.join(a.base, "realisation_*")))
    if not real_dirs:
        raise SystemExit(f"no realisation_* folders in {a.base}")

    with open(out, "w") as fh:
        fh.write(f"# kernel={a.kernel} lambda={a.lam} alpha={a.alpha} seed={a.seed}\n")
        fh.write("# split without replacement: no configuration is shared by two students\n")
        fh.write("# teacher  student  temperature  n_samples  J_hat  RC\n")

        for real_dir in real_dirs:
            teacher = int(re.sub(r"\D", "", os.path.basename(real_dir)))
            for T_dir in sorted(glob.glob(os.path.join(real_dir, "T_*"))):
                fp = os.path.join(T_dir, "output.dat")
                if not os.path.isfile(fp):
                    continue
                T = float(os.path.basename(T_dir)[2:])
                data = np.loadtxt(fp)                               # (N_tot, 32)
                n_tot = data.shape[0]
                n_students = a.n_students or n_tot // a.n_samples
                if n_students * a.n_samples > n_tot:
                    raise SystemExit(f"{fp}: {n_students} x {a.n_samples} > {n_tot} configurations")

                # reproducible split for this (teacher, T, M)
                rng = np.random.default_rng([a.seed, teacher, int(round(T * 1000)), a.n_samples])
                blocks = split_without_replacement(n_tot, a.n_samples, n_students, rng)

                for s, rows in enumerate(blocks, start=1):
                    J_hat, RC = infer_coupling(data[rows].sum(axis=0), T, K, a.alpha, a.lam)
                    fh.write(f"{teacher:5d}  {s:5d}  {T:.3f}  {a.n_samples:6d}  "
                             f"{J_hat:.10f}  {RC:.10f}\n")
            print(f"  done {os.path.basename(real_dir)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
