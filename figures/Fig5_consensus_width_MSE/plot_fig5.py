#!/usr/bin/env python3
"""
2x6 figure, columns = disorder delta in {0, 0.2, 0.4, 0.6, 0.8, 0.999}:
  Row (a): sigma = sqrt(var_J_hat) with sqrt(V_sample), sqrt(V_disorder) vs T/Tc
  Row (b): MSE = bias^2 + var_J_hat vs T/Tc, one curve per N in {1,2,...,128}
The two rows share the same x-axis, so only the bottom row shows x-tick labels.

Reads fig5_pooled_delta<d>.dat, fig5_across_delta<d>.dat (row a) and
fig5_MSE_vs_N_delta<d>.dat (row b).
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, LogLocator, FixedLocator

# ============================================================
# CONTROLS  (edit these)
# ============================================================
FIGSIZE      = (28.0, 9.0)     # overall figure size (inches)
XLIM         = (0.8, 1.2)      # T/Tc range shown in every panel

# x-axis ticks for the BOTTOM row (which T/Tc values get labelled)
ROW2_XTICKS       = [0.9, 1.0, 1.1]
ROW2_XTICK_MINOR  = 0.05        # minor tick spacing (None to disable)

# y-axis controls
ROW1_YMAX_PAD = 1.08
ROW1_SCALE    = 1e3            # row (a) plotted in units of 10^-3
ROW1_YLIM     = (0, 11)        # row (a) y range (in 10^-3)
ROW1_YTICKS   = [0, 2, 4, 6, 8, 10]
FILL_ALPHA    = 0.15           # shading between sqrt(V_sample) and sqrt(V_disorder)
ROW2_YPAD_LO, ROW2_YPAD_HI = 0.7, 1.4

# font sizes
FS_LABEL   = 32
FS_TICK    = 26
FS_TITLE   = 29
FS_PANEL   = 35
FS_LEGEND  = 24

N_STUDENTS = 250

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
    "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 1.2, "lines.linewidth": 2.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
    "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

# ============================================================
# DATA MAPS
# ============================================================
deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
Tc = {0.0:2.28,0.2:2.28,0.4:2.24,0.6:2.20,0.8:2.12,1.0:2.04}
pooled = {0.0:'fig5_pooled_delta0.000.dat',0.2:'fig5_pooled_delta0.200.dat',0.4:'fig5_pooled_delta0.400.dat',
          0.6:'fig5_pooled_delta0.600.dat',0.8:'fig5_pooled_delta0.800.dat',1.0:'fig5_pooled_delta0.999.dat'}
across = {0.0:'fig5_across_delta0.000.dat',0.2:'fig5_across_delta0.200.dat',0.4:'fig5_across_delta0.400.dat',
          0.6:'fig5_across_delta0.600.dat',0.8:'fig5_across_delta0.800.dat',1.0:'fig5_across_delta0.999.dat'}
mse_f  = {0.0:'fig5_MSE_vs_N_delta0.000.dat',0.2:'fig5_MSE_vs_N_delta0.200.dat',0.4:'fig5_MSE_vs_N_delta0.400.dat',
          0.6:'fig5_MSE_vs_N_delta0.600.dat',0.8:'fig5_MSE_vs_N_delta0.800.dat',1.0:'fig5_MSE_vs_N_delta0.999.dat'}
Ns = [1, 2, 4, 8, 16, 32, 64, 128]

def dlabel(d): return r"$\delta=0.999$" if d==1.0 else (r"$\delta=0$" if d==0.0 else rf"$\delta={d}$")

COL_W  = "#1A1A1A"
COL_VS = "#2C6FB5"
COL_VD = "#E03020"
# one colour per N, dark purple (small N) -> orange (large N)
N_COLORS = plt.cm.plasma(np.linspace(0.05, 0.85, len(Ns)))
ncol = {n: N_COLORS[i] for i, n in enumerate(Ns)}
XLABEL = r"$T/T_{c}^{\mathrm{eff}}(\delta)$"

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(2, 6, figsize=FIGSIZE)
fig.subplots_adjust(left=0.05, right=0.93, top=0.94, bottom=0.10,
                    wspace=0.14, hspace=0.12)

def apply_row2_xaxis(ax):
    ax.xaxis.set_major_locator(FixedLocator(ROW2_XTICKS))
    ax.set_xticklabels([f"{t:g}" for t in ROW2_XTICKS])
    if ROW2_XTICK_MINOR:
        ax.xaxis.set_minor_locator(MultipleLocator(ROW2_XTICK_MINOR))

# ===== Row (a) =====
wmax = 0.0
R1 = {}
for d in deltas:
    p = np.loadtxt(pooled[d]); a = np.loadtxt(across[d])
    T = p[:, 0]; o = np.argsort(T)
    x = (T / Tc[d])[o]
    W   = np.sqrt(p[o, 4]) * ROW1_SCALE
    sVs = np.sqrt(a[o, 4] * N_STUDENTS) * ROW1_SCALE
    sVd = np.sqrt(a[o, 6]) * ROW1_SCALE
    R1[d] = (x, W, sVs, sVd)
    win = (x >= XLIM[0]) & (x <= XLIM[1])
    wmax = max(wmax, W[win].max(), sVs[win].max(), sVd[win].max())

for j, d in enumerate(deltas):
    ax = axes[0, j]
    x, W, sVs, sVd = R1[d]
    # shade the gap between the two components: blue where sampling dominates,
    # red where disorder dominates
    ax.fill_between(x, sVs, sVd, where=sVs >= sVd, interpolate=True,
                    color=COL_VS, alpha=FILL_ALPHA, lw=0, zorder=1)
    ax.fill_between(x, sVs, sVd, where=sVd >= sVs, interpolate=True,
                    color=COL_VD, alpha=FILL_ALPHA, lw=0, zorder=1)
    ax.plot(x, W, "-", color=COL_W, lw=1.4, zorder=4, label=r"$\mathcal{W}$")
    ax.plot(x, sVs, ls="none", marker="o", ms=5.0, mfc=COL_VS, mec=COL_VS,
            zorder=3, label=r"$\sqrt{V_{\mathrm{sample}}}$")
    ax.plot(x, sVd, ls="none", marker="o", ms=5.0, mfc=COL_VD, mec=COL_VD,
            zorder=3, label=r"$\sqrt{V_{\mathrm{disorder}}}$")
    ax.axvline(1.0, color="0.6", ls=(0, (4, 3)), lw=0.9, zorder=0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ROW1_YLIM)
    ax.set_yticks(ROW1_YTICKS)
    ax.set_title(dlabel(d), pad=6)
    apply_row2_xaxis(ax)
    ax.tick_params(labelbottom=False)
    ax.tick_params(top=True, right=True, which="both")
    if j == 0:
        ax.set_ylabel(r"$\sigma\ (10^{-3})$")
        ax.legend(loc="upper left", frameon=False, handlelength=1.3,
                  labelspacing=0.3, borderaxespad=0.2, fontsize=FS_LEGEND)
        ax.text(-0.40, 1.10, r"(a)", transform=ax.transAxes, fontsize=FS_PANEL,
                ha="left", va="bottom")
    else:
        ax.tick_params(labelleft=False)

# ===== Row (b) =====
mlo, mhi = np.inf, 0.0
MSE_data = {}
for d in deltas:
    a = np.loadtxt(mse_f[d]); MSE_data[d] = a
    for n in Ns:
        sel = a[:, 1] == n
        x = a[sel, 0] / Tc[d]; mse = a[sel, 2]**2 + a[sel, 4]
        win = (x >= XLIM[0]) & (x <= XLIM[1])
        if win.any():
            mlo = min(mlo, mse[win].min()); mhi = max(mhi, mse[win].max())

for j, d in enumerate(deltas):
    ax = axes[1, j]
    a = MSE_data[d]
    for n in Ns:
        sel = a[:, 1] == n
        T = a[sel, 0]; o = np.argsort(T)
        x = (T / Tc[d])[o]; mse = (a[sel, 2]**2 + a[sel, 4])[o]
        ax.plot(x, mse, "-", color=ncol[n], lw=1.9,
                label=(str(n) if j == 5 else None))
    ax.axvline(1.0, color="0.6", ls=(0, (4, 3)), lw=0.9, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(mlo * ROW2_YPAD_LO, mhi * ROW2_YPAD_HI)
    ax.set_xlabel(XLABEL)
    apply_row2_xaxis(ax)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.tick_params(top=True, right=True, which="both")
    if j == 0:
        ax.set_ylabel(r"MSE")
        ax.text(-0.40, 1.02, r"(b)", transform=ax.transAxes, fontsize=FS_PANEL,
                ha="left", va="top")
    else:
        ax.tick_params(labelleft=False)
    if j == 5:
        ax.legend(loc="lower right", frameon=False, title=r"$N$",
                  ncol=2, handlelength=1.3, columnspacing=1.0,
                  labelspacing=0.25, fontsize=FS_LEGEND-2, title_fontsize=FS_LEGEND)

fig.savefig("Fig5.pdf")
fig.savefig("Fig5.svg")
fig.savefig("Fig5.png", dpi=400)
print("ok")
