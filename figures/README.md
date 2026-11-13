# Figures of "Consensus in Effective Model Inference for Disordered Ising Systems"

Each folder holds everything for one figure: the plotting script (`plot_figN.py`),
the data it reads (`figN_*.dat` / `.csv`) and the figure it produces (`FigN.pdf/png`).
Run a script from inside its own folder, e.g. `cd Fig3_bias_fisher_CR && python3 plot_fig3.py`.

Notation in file names: `delta0.600` = disorder strength δ = 0.6 (`delta0.999` is δ = 0.999,
the largest disorder); `T2.240` = temperature T = 2.24; L = 50 unless stated otherwise.

| Folder | Script | Input data | Output |
|---|---|---|---|
| `Fig1_schematic/` | – | – | `Fig1.pdf` (schematic of the consensus framework) |
| `Fig2_consensus_distribution/` | `plot_fig2a.py` | `fig2a_Jbar_vs_T.dat`: J̄ vs T, one column per δ | `Fig2a.pdf/png` |
| | `plot_fig2b.py` | `fig2b_students_T1.600.csv`, `…T2.240.csv`, `…T3.000.csv`: J and RC of every student (200 realisations × 250 students), δ = 0.5 | `Fig2b.pdf/png` |
| `Fig3_bias_fisher_CR/` | `plot_fig3.py` | `fig3_bias_theory_and_measured.dat`: bias, b′, theory vs measured, all δ<br>`fig3_across_delta<δ>.dat`: J̄, V_disorder, V_samples, Fisher information vs T<br>`fig3_sample_size_sweep_delta0.500.dat`: the same vs configurations per student | `Fig3.pdf/png/svg` |
| `Fig4_disorder_variance/` | `plot_fig4.py` | `fig4_across_delta<δ>.dat`, `fig4_pooled_delta<δ>.dat`: variance decomposition vs T<br>`fig4b_2CovLC_all_delta.csv`: 2 Cov(L, C) vs T, all δ<br>`fig4c_Vdisorder_terms_delta0.600.csv`: Var(L), Var(Q), 2 Cov(L, C) at δ = 0.6<br>`fig4f_variances_vs_L_delta0.500.dat`: variances for L = 2 … 256<br>`fig4f_R_vs_L_delta0.500.dat`: R(L) at δ = 0.5, T = 2.22 (extracted from the previous file) | `Fig4.pdf/png` |
| `Fig5_consensus_width_MSE/` | `plot_fig5.py` | `fig5_across_delta<δ>.dat`, `fig5_pooled_delta<δ>.dat`: consensus width and its two parts<br>`fig5_MSE_vs_N_delta<δ>.dat`: bias and variance vs configurations per student N | `Fig5.pdf/png/svg` |
| `Fig6_kernels_RC/` | `plot_fig6.py` | `fig6_Jbar_per_kernel.dat`, `fig6_RC_per_kernel.dat`: J̄ and RC vs T for each kernel and δ (plotted at T = 2.20) | `Fig6.svg` |

The `fig4_*` and `fig5_*` across/pooled files contain the same data; they are kept in
both folders so each figure folder is self-contained.

Requirements: Python 3 with NumPy, pandas and Matplotlib; `plot_fig2b.py` also needs
`scienceplots`, and `plot_fig2a.py` / `plot_fig2b.py` use LaTeX for text rendering
(set `usetex = False` / `USE_LATEX = False` to run without a LaTeX installation).
