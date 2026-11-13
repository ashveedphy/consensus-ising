#!/usr/bin/env python3
"""
teacher_quantities.py
=====================

Quantities of each teacher (realisation omega, temperature T) computed from
ALL of its N_tot configurations and from its bonds. These are needed by
consensus_analysis.py for the bias, bias derivative, Fisher information,
Cramer-Rao bound and the split of the disorder variance.

Per teacher:
  J_star_full   coupling inferred from all N_tot configurations
                (infinite-data proxy J*_omega, Eq. 46), identity kernel
  fisher_I      Fisher information per configuration about the uniform
                coupling,  I_omega = beta^2 Var(E)                      Eq. (34)
                with E = sum_<ij> s_i s_j = 0.5 sum_x S(x) [n+(x) - n-(x)]
  bprime_LR     bias derivative b'(J0) by linear response               Eq. (37)
                A uniform shift eps of every bond changes any average by
                d<O>/d eps = beta Cov(O, E), so
                b' = d J_hat / d eps - 1
                   = T * sum_k (dphi/dn_k) * N_tot * beta * Cov(c_k, E) - 1
  bond_mean     mean of the bonds of the realisation (Jx.dat, Jy.dat);
                L_omega = bond_mean - J0 is the linear disorder response (Eq. 48)
  N(0..15)      pattern weights N(x) from all configurations (for Eqs. 31, D19)

Output (one row per teacher):
    teacher  temperature  n_configs  J_star_full  fisher_I  mean_E  var_E
    bprime_LR  bond_mean  N0 ... N15

Usage:
    python teacher_quantities.py --base delta_0.500
"""

import argparse
import glob
import os
import re

import numpy as np

PHI = np.array([[2 * ((x >> k) & 1) - 1 for k in range(4)] for x in range(16)], float)
S = PHI.sum(axis=1)


def coupling_and_gradient(total, T, alpha=0.5):
    """J_hat from total counts (identity kernel) and dphi/dn for (n+, n-)."""
    n_plus, n_minus = total[0::2], total[1::2]
    a_p, a_m = n_plus + alpha, n_minus + alpha
    N = n_plus + n_minus
    f = 0.5 * np.log(a_p / a_m)                                  # Eqs. (13)-(14)
    D = np.sum(N * S ** 2)
    phi = np.sum(N * S * f) / D                                  # beta * J_hat
    grad = np.empty(32)
    grad[0::2] = (S * f + N * S / (2 * a_p) - phi * S ** 2) / D  # d phi / d n+(x)
    grad[1::2] = (S * f - N * S / (2 * a_m) - phi * S ** 2) / D  # d phi / d n-(x)
    return T * phi, grad, N


def read_bond_mean(real_dir, header_lines=3):
    vals = []
    for name in ("Jx.dat", "Jy.dat"):
        fp = os.path.join(real_dir, name)
        if not os.path.isfile(fp):
            return np.nan
        vals.append(np.loadtxt(fp, skiprows=header_lines).ravel())
    return float(np.concatenate(vals).mean())


def main():
    ap = argparse.ArgumentParser(description="Per-teacher quantities from all configurations")
    ap.add_argument("--base", default=".", help="delta folder containing realisation_* folders")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(a.base, "teachers.dat")
    real_dirs = sorted(glob.glob(os.path.join(a.base, "realisation_*")))
    if not real_dirs:
        raise SystemExit(f"no realisation_* folders in {a.base}")

    with open(out, "w") as fh:
        fh.write("# teacher  temperature  n_configs  J_star_full  fisher_I  mean_E  var_E  "
                 "bprime_LR  bond_mean  " + "  ".join(f"N{x}" for x in range(16)) + "\n")

        for real_dir in real_dirs:
            teacher = int(re.sub(r"\D", "", os.path.basename(real_dir)))
            bond_mean = read_bond_mean(real_dir)
            for T_dir in sorted(glob.glob(os.path.join(real_dir, "T_*"))):
                fp = os.path.join(T_dir, "output.dat")
                if not os.path.isfile(fp):
                    continue
                T = float(os.path.basename(T_dir)[2:])
                beta = 1.0 / T
                data = np.loadtxt(fp)                                # (N_tot, 32)
                n_tot = data.shape[0]

                # coupling from all configurations
                J_star, grad, N = coupling_and_gradient(data.sum(axis=0), T, a.alpha)

                # Fisher information, Eq. (34)
                E = 0.5 * (data[:, 0::2] - data[:, 1::2]) @ S        # bond sum per config
                var_E = np.var(E, ddof=1)
                fisher_I = beta ** 2 * var_E

                # bias derivative by linear response, Eq. (37)
                cov_cE = (data - data.mean(0)).T @ (E - E.mean()) / (n_tot - 1)
                bprime_LR = T * grad @ (n_tot * beta * cov_cE) - 1.0

                fh.write(f"{teacher:5d}  {T:.3f}  {n_tot:6d}  {J_star:.10f}  {fisher_I:.8e}  "
                         f"{E.mean():.8e}  {var_E:.8e}  {bprime_LR:+.8e}  {bond_mean:.10f}  "
                         + "  ".join(f"{v:.0f}" for v in N) + "\n")
            print(f"  done {os.path.basename(real_dir)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
