#!/usr/bin/env python3
"""
generate_bonds.py
=================

Generate the quenched bond realisations (teachers) of the 2D random-bond
Ising model used in the paper, Eq. (4):

    J_ij ~ U(J0 - delta, J0 + delta),   J0 = 1,   i.i.d. for every bond,

on an L x L = 50 x 50 square lattice with periodic boundaries. Each site
(i, j) owns two bonds, as read by the Fortran codes: Jx(i, j) couples (i, j)
to (i+1, j) and Jy(i, j) couples (i, j) to (i, j+1), giving N_b = 2 L^2 = 5000
bonds. Since delta < 1, all bonds
are ferromagnetic and there is no frustration.

Paper settings:
    L = 50
    delta in {0.000, 0.200, 0.400, 0.600, 0.800, 0.999}
    N_omega = 200 realisations per delta

Creates
    output/System_Size_L50/delta_0.000/realisation_0001/Jx.dat
    output/System_Size_L50/delta_0.000/realisation_0001/Jy.dat
    ...
    output/System_Size_L50/delta_0.999/realisation_0200/Jy.dat

File format (read by the Fortran Monte Carlo code):
    # Jx
    # L = 50
    50
    <50 rows of 50 values, 10 decimals>

Reproducibility: each (delta, realisation) has its own fixed seed,
    seed = base_seed * 1_000_000 + round(1000 * delta) * 10_000 + realisation,
so rerunning the script regenerates exactly the bonds used for the paper.
Existing files are never overwritten, so bonds already used by the Monte
Carlo runs can't change by accident.

Usage
    python generate_bonds.py
"""

from pathlib import Path

import numpy as np

# ---------------- paper settings ----------------
L = 50
J0 = 1.0
DELTAS = [0.000, 0.200, 0.400, 0.600, 0.800, 0.999]
N_REALISATIONS = 200
BASE_SEED = 123456
OUT_ROOT = Path("output")


def write_matrix_ascii(path: Path, A: np.ndarray, tag: str) -> None:
    """Write an L x L bond array in the format read by the Fortran code."""
    n = A.shape[0]
    with path.open("w") as f:
        f.write(f"# {tag}\n# L = {n}\n{n}\n")
        for row in A:
            f.write(" ".join(f"{v:.10f}" for v in row) + "\n")


def make_bonds(delta: float, rng: np.random.Generator):
    """Jx, Jy ~ U(J0 - delta, J0 + delta); delta = 0 gives all bonds = J0."""
    Jx = rng.uniform(J0 - delta, J0 + delta, size=(L, L))
    Jy = rng.uniform(J0 - delta, J0 + delta, size=(L, L))
    return Jx, Jy


def main() -> None:
    sys_dir = OUT_ROOT / f"System_Size_L{L}"
    for d in DELTAS:
        assert 0.0 <= d < 1.0, "delta must satisfy 0 <= delta < 1"
        delta_dir = sys_dir / f"delta_{d:0.3f}"
        kept = 0
        for rid in range(1, N_REALISATIONS + 1):
            real_dir = delta_dir / f"realisation_{rid:04d}"
            if (real_dir / "Jx.dat").exists() and (real_dir / "Jy.dat").exists():
                kept += 1                      # never overwrite bonds already in use
                continue
            real_dir.mkdir(parents=True, exist_ok=True)

            seed = BASE_SEED * 1_000_000 + int(round(d * 1000)) * 10_000 + rid
            Jx, Jy = make_bonds(d, np.random.default_rng(seed))
            write_matrix_ascii(real_dir / "Jx.dat", Jx, "Jx")
            write_matrix_ascii(real_dir / "Jy.dat", Jy, "Jy")

        note = f"  ({kept} existing realisations kept)" if kept else ""
        print(f"delta={d:0.3f} -> {delta_dir}{note}")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
