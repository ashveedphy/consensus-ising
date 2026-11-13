#!/usr/bin/env python3
"""
consensus_analysis.py
=====================

Consensus distribution and its decomposition, per temperature, from the
outputs of jhat_estimation_by_student.py and teacher_quantities.py.
omega labels teachers (realisations), s labels students, M = n_samples.

Consensus (Sec. III C)
  Jbar_omega   = mean_s J_hat(omega, s)                                Eq. (21)
  V_omega      = Var_s  J_hat(omega, s)        (ddof = 1)              Eq. (22)
  V_samples    = mean_omega V_omega            sampling variance       Eq. (23)
  V_disorder   = Var_omega Jbar_omega          disorder variance       Eq. (24)
  V_consensus  = V_samples + V_disorder                                Eq. (26)
  V_pooled     = variance of all J_hat pooled (check of Eq. 26)        Eq. (25)
  W            = sqrt(V_consensus)             consensus width

Accuracy (Sec. IV)
  bias         = mean of all J_hat - J0                                Eq. (27)
  bias_theory  = -K_loc(T) delta^2,
                 K_loc = 4 sum N S tanh(beta J0 S) / (3 T sum N S^2)   Eq. (31)
  bprime_th    = -(4 delta^2 / 3T^2) sum N S^2 sech^2(beta J0 S) / sum N S^2   Eq. (D19)
  bprime_LR    = linear-response b' from teacher_quantities.py         Eq. (37)
  fisher_I     = mean_omega beta^2 Var(E)                              Eq. (34)
  CR_bound     = mean_omega (1 + b'_omega)^2 / (M I_omega)             Eq. (38)
  CR_unbiased  = mean_omega 1 / (M I_omega)
  efficiency   = CR_bound / V_samples                                  Eq. (41)
  MSE          = bias^2 + V_samples + V_disorder                       Eq. (56)

Split of the disorder variance (Sec. IV B), using J*_omega from all configs
  V_dis_inf    = Var_omega J*_omega                                    Eq. (46)
  L_omega      = mean bond - J0          (g_a = 1/N_b, Eq. C4)
  R_omega      = J*_omega - J0 - L_omega = Q + C + O(delta^4)
  Var_L        = Var_omega L_omega,      theory delta^2 / (3 N_b)      Eq. (C15)
  Var_Q        ~ Var_omega R_omega       (Var(C) is O(delta^6))        Eq. (C21)
  twoCov_LC    ~ 2 Cov_omega(L, R)       (Cov(L,Q) = 0 by symmetry)    Eq. (C25)
  Note: Var_L + Var_Q + twoCov_LC = V_dis_inf holds identically here.

Characteristic temperatures (grid minimum / maximum)
  T_I = argmax fisher_I, T_CR = argmin CR_bound, T_s = argmin V_samples,
  T_d = argmin V_disorder, T_V = argmin V_consensus, T_opt = argmin MSE

Usage:
    python consensus_analysis.py --students delta_0.500/students_M100_identity.dat \
                                 --teachers delta_0.500/teachers.dat --delta 0.5
"""

import argparse
import os

import numpy as np

J0 = 1.0
S = np.array([sum(2 * ((x >> k) & 1) - 1 for k in range(4)) for x in range(16)], float)


def bias_theory(N, T, delta):
    """Fixed-pattern bias, Eq. (31): b = -K_loc(T) delta^2."""
    K_loc = 4 * np.sum(N * S * np.tanh(J0 * S / T)) / (3 * T * np.sum(N * S ** 2))
    return -K_loc * delta ** 2


def bprime_theory(N, T, delta):
    """Bias derivative, Eq. (D19). N may be (16,) or (n_teachers, 16)."""
    sech2 = 1.0 / np.cosh(J0 * S / T) ** 2
    return (-(4 * delta ** 2 / (3 * T ** 2))
            * np.sum(N * S ** 2 * sech2, -1) / np.sum(N * S ** 2, -1))


def main():
    ap = argparse.ArgumentParser(description="Consensus variance decomposition and accuracy")
    ap.add_argument("--students", required=True, help="output of jhat_estimation_by_student.py")
    ap.add_argument("--teachers", required=True, help="output of teacher_quantities.py")
    ap.add_argument("--delta", type=float, required=True, help="disorder strength")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # students: teacher student T n_samples J_hat RC
    st = np.loadtxt(a.students, comments="#")
    s_teacher, s_T, s_M, s_J, s_RC = st[:, 0].astype(int), st[:, 2], st[:, 3], st[:, 4], st[:, 5]
    M = int(s_M[0])

    # teachers: teacher T n_configs J_star I mean_E var_E bprime_LR bond_mean N0..N15
    tq = np.loadtxt(a.teachers, comments="#")
    t_teacher, t_T, t_ntot = tq[:, 0].astype(int), tq[:, 1], tq[:, 2]
    t_Jstar, t_I, t_bpLR, t_bond = tq[:, 3], tq[:, 4], tq[:, 7], tq[:, 8]
    t_N = tq[:, 9:25]
    n_bonds = 2 * int(round(t_N[0].sum() / t_ntot[0]))          # N_b = 2 L^2

    rows = []
    for T in np.unique(np.round(s_T, 3)):
        sel = np.isclose(s_T, T)
        teachers = np.unique(s_teacher[sel])

        # ---- consensus: per-teacher mean and variance over students
        Jbar_w = np.array([s_J[sel & (s_teacher == w)].mean() for w in teachers])
        V_w = np.array([s_J[sel & (s_teacher == w)].var(ddof=1) for w in teachers])
        V_samples = V_w.mean()                                   # Eq. (23)
        V_disorder = Jbar_w.var(ddof=1)                          # Eq. (24)
        V_consensus = V_samples + V_disorder                     # Eq. (26)
        V_pooled = s_J[sel].var()                                # Eq. (25)
        bias = s_J[sel].mean() - J0                              # Eq. (27)

        # ---- teacher quantities at this T (same teachers, same order)
        tsel = np.isclose(t_T, T)
        idx = [np.where(tsel & (t_teacher == w))[0][0] for w in teachers]
        I_w, bpLR_w, Jstar_w = t_I[idx], t_bpLR[idx], t_Jstar[idx]
        N_w, bond_w = t_N[idx], t_bond[idx]

        bp_w = bprime_theory(N_w, T, a.delta)                    # Eq. (D19)
        CR = np.mean((1 + bp_w) ** 2 / (M * I_w))                # Eq. (38)
        CR_LR = np.mean((1 + bpLR_w) ** 2 / (M * I_w))
        CR_unb = np.mean(1.0 / (M * I_w))

        # ---- split of the disorder variance
        L_w = bond_w - J0
        R_w = Jstar_w - J0 - L_w
        Var_L = L_w.var(ddof=1)
        Var_Q = R_w.var(ddof=1)
        twoCov = 2 * np.cov(L_w, R_w, ddof=1)[0, 1]

        rows.append(dict(
            T=T, n_teachers=len(teachers), Jbar=s_J[sel].mean(),
            V_samples=V_samples, V_disorder=V_disorder, V_consensus=V_consensus,
            V_pooled=V_pooled, W=np.sqrt(V_consensus),
            bias=bias, bias_theory=bias_theory(N_w.mean(0), T, a.delta),
            bprime_th=bp_w.mean(), bprime_LR=bpLR_w.mean(),
            fisher_I=I_w.mean(), CR_bound=CR, CR_bound_LR=CR_LR, CR_unbiased=CR_unb,
            efficiency=CR / V_samples,
            MSE=bias ** 2 + V_samples + V_disorder,              # Eq. (56)
            V_dis_inf=Jstar_w.var(ddof=1), Var_L=Var_L,
            Var_L_theory=a.delta ** 2 / (3 * n_bonds), Var_Q=Var_Q, twoCov_LC=twoCov,
            RC=s_RC[sel].mean()))

    cols = list(rows[0])
    tab = {c: np.array([r[c] for r in rows]) for c in cols}
    Tg = tab["T"]
    temps = {
        "T_I": Tg[np.argmax(tab["fisher_I"])],
        "T_CR": Tg[np.argmin(tab["CR_bound"])],
        "T_s": Tg[np.argmin(tab["V_samples"])],
        "T_d": Tg[np.argmin(tab["V_disorder"])],
        "T_V": Tg[np.argmin(tab["V_consensus"])],
        "T_opt": Tg[np.argmin(tab["MSE"])],
    }

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.students)),
                                f"consensus_M{M}.dat")
    with open(out, "w") as fh:
        fh.write(f"# delta={a.delta}  M={M}  N_b={n_bonds}\n")
        fh.write("# " + "  ".join(f"{k}={v:.3f}" for k, v in temps.items()) + "\n")
        fh.write("# " + "  ".join(cols) + "\n")
        for r in rows:
            fh.write("  ".join(f"{r[c]:.3f}" if c == "T" else
                               f"{r[c]:d}" if c == "n_teachers" else
                               f"{r[c]:.8e}" for c in cols) + "\n")

    print(f"delta={a.delta}  M={M}  temperatures={len(rows)}")
    for k, v in temps.items():
        print(f"  {k:<6s} = {v:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
