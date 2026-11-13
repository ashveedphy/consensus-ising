#!/usr/bin/env python3
"""
plot_fig4.py -- Fig. 4, 6-panel (2x3) disorder-variance summary figure
for the paper. Config-driven: everything to tweak lives in the CONFIG
block, nothing is hard-coded further down.

Layout
    Row 1:
        (a) V_dis vs T/Tc, one curve per delta, colorbar, circle markers
            on top of each curve.
        (b) cross term 2*Cov(L,dJ)/delta^2 vs T/Tc, one curve per delta,
            circle markers on top.
        (c) decomposition of V_disorder at delta=0.6 into
            Var(L) + Var(dJ) + 2*Cov(L,dJ) = sum (precomputed CSV).
    Row 2:
        (d) NEW: the T at which V_dis and the cross term each reach their
            minimum, plotted against delta, compared to the clean T_c.
            (A per-delta Hessian/Var(dJ) minimum is NOT included: at the
            one delta we have decomposition data for, Var(dJ) decreases
            monotonically across the whole T window -- its "minimum" is
            just the edge of the data range, not a real dip, so it isn't
            a meaningful third curve here. Supply a per-delta Var(dJ)
            file if you want it added properly.)
        (e) the delta=0.6 decomposition built from the *raw* across/pooled
            files: V_samples, V_disorder, their sum, and V_consensus vs T.
        (f) R(L) = V_dis / J_bar^2 vs system size L (log-log, L on a
            log2-scaled axis), using real data all the way to L=256.

Input files (all whitespace/'#'-comment .dat unless noted)
    fig4_across_delta<d>.dat : T J_star J_hat V_sample_Jstar V_sample_Jhat
                        V_dis_Jstar V_dis_Jhat sigma2_q_Jstar sigma2_q_Jhat
                        RC_mean Fisher n_realisations
    fig4_pooled_delta<d>.dat : T J_star J_hat var_J_star var_J_hat mean_RC
                        mean_Fisher n_students n_realisations
    fig4f_variances_vs_L_delta0.500.dat : L delta T V_sample J_bar V_dis n_real
    fig4b_2CovLC_all_delta.csv (comma-separated, header row)
                      : delta, T, TwoCov_over_delta2
    fig4c_Vdisorder_terms_delta0.600.csv (comma-separated, header row)
                      : T, Var_L, Var_dJ, TwoCov_L_dJ, Sum_Vdisorder

Run
    python3 plot_fig4.py
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, NullFormatter

# =====================================================================
#                             CONFIG
# =====================================================================
CONFIG = dict(

    # ---------- input ----------
    # None = the folder this script lives in (robust to being run from
    # anywhere); set to a string to point at a different data folder.
    data_dir = None,
    across_files = {0.0: "fig4_across_delta0.000.dat", 0.2: "fig4_across_delta0.200.dat",
                    0.4: "fig4_across_delta0.400.dat", 0.6: "fig4_across_delta0.600.dat",
                    0.8: "fig4_across_delta0.800.dat", 1.0: "fig4_across_delta0.999.dat"},
    pooled_files = {0.0: "fig4_pooled_delta0.000.dat", 0.2: "fig4_pooled_delta0.200.dat",
                    0.4: "fig4_pooled_delta0.400.dat", 0.6: "fig4_pooled_delta0.600.dat",
                    0.8: "fig4_pooled_delta0.800.dat", 1.0: "fig4_pooled_delta0.999.dat"},
    ldt_file    = "fig4f_variances_vs_L_delta0.500.dat",
    panelB_csv  = "fig4b_2CovLC_all_delta.csv",
    panelC_csv  = "fig4c_Vdisorder_terms_delta0.600.csv",
    rl_file     = "fig4f_R_vs_L_delta0.500.dat",   # panel (f): precomputed R(L) = V_dis/J_bar^2 vs L

    # column indices -- change these if a file layout changes
    across_cols = dict(T=0, J_star=1, J_hat=2, V_sample_Jstar=3, V_sample_Jhat=4,
                       V_dis_Jstar=5, V_dis_Jhat=6, sigma2_q_Jstar=7,
                       sigma2_q_Jhat=8, RC_mean=9, Fisher=10, n_real=11),
    pooled_cols = dict(T=0, J_star=1, J_hat=2, var_J_star=3, var_J_hat=4,
                       mean_RC=5, mean_Fisher=6, n_students=7, n_real=8),
    ldt_cols    = dict(L=0, delta=1, T=2, V_sample=3, J_bar=4, V_dis=5, n_real=6),
    rl_cols     = dict(L=0, delta=1, T=2, V_sample=3, J_bar=4, V_dis=5, n_real=6, R=7),

    # ---------- physics / series selection ----------
    deltas     = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],   # panel (a)
    Tc         = {0.0: 2.28, 0.2: 2.28, 0.4: 2.24, 0.6: 2.20, 0.8: 2.12, 1.0: 2.04},
    Tc_clean   = 2.27,     # single reference T_c used in panels (b)(c)(d)
    n_students = 250,      # rescales V_sample_Jhat back to unnormalised V_sample
    delta_show = 0.6,      # which delta panels (c) and (e) focus on
    b_show_deltas = [0.2, 0.4, 0.6, 0.8, 1.0],   # which curves panel (b)/(d) show
    xlim_a     = (0.8, 1.2),

    # ---------- marker density ----------
    # (every real data point is plotted -- no subsampling, no interpolation)

    # ---------- panel (f): R(L) ----------
    # fig4f_R_vs_L_delta0.500.dat is already a single (delta, T) slice with R(L)
    # precomputed as its own column, so no further filtering is needed here.

    # ---------- colours ----------
    viridis_range = (0.0, 0.85),    # slice of viridis shared by panels (a) and (b)

    col_sample = "#2C6FB5",    # panel (e): V_samples
    col_dis    = "#E03020",    # panel (e): V_disorder
    col_sum    = "black",      # panel (e): sum (open circles)
    col_cons   = "#1A1A1A",    # panel (e): V_consensus line

    col_VL   = "C2",     # panel (c): Var(L)
    col_VdJ  = "C0",     # panel (c): Var(dJ)
    col_cov  = "C3",     # panel (c): 2 Cov(L, dJ)
    col_sumC = "black",  # panel (c): sum

    col_Tdis = "black",    # panel (d): argmin V_dis curve
    col_Tcov = "C3",       # panel (d): argmin cross-term curve (matches panel c's cross-term color)

    # ---------- figure / layout ----------
    fig_w   = 20.5,
    fig_h   = 11.8,
    height_ratios = [1.0, 1.0],
    wspace  = 0.40,
    hspace  = 0.40,
    margins = dict(left=0.052, right=0.978, top=0.96, bottom=0.07),

    # ---------- fonts (points) ----------
    fs_label  = 28,
    fs_tick   = 24,
    fs_panel  = 30,
    fs_legend = 18,
    fs_cbar   = 26,
    fs_cbtick = 20,
    fs_annot  = 21,

    # ---------- lines / markers ----------
    lw_curve      = 2.6,
    lw_sum        = 2.8,
    lw_Vdis_thin  = 1.4,  # panel (c): thin V_disorder line
    ms_dot     = 7.5,
    ms_sum     = 7.5,
    mew_sum    = 1.7,
    ms_decomp  = 6.5,
    ms_sum_c   = 7.5,     # panel (c): solid circles for the separately-recomputed sum
    ms_cov     = 7.0,     # panel (b): circle markers on the cross-term curves
    ms_a       = 7.0,     # panel (a): circle markers on the V_dis curves
    ms_d       = 10.5,    # panel (d): argmin markers
    axes_lw    = 1.6,
    tick_lw    = 1.5,
    tick_major = 7.5,
    tick_minor = 4.2,

    # ---------- output ----------
    savename = "Fig4",
    formats  = ("pdf", "png"),
    dpi      = 400,
)


# =====================================================================
#                       STYLE / LOADING (no need to edit)
# =====================================================================
def apply_style(cfg):
    plt.rcParams.update({
        "mathtext.fontset": "cm", "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
        "axes.labelsize": cfg["fs_label"], "xtick.labelsize": cfg["fs_tick"],
        "ytick.labelsize": cfg["fs_tick"], "legend.fontsize": cfg["fs_legend"],
        "axes.linewidth": cfg["axes_lw"], "lines.linewidth": cfg["lw_curve"],
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": cfg["tick_major"], "ytick.major.size": cfg["tick_major"],
        "xtick.major.width": cfg["tick_lw"], "ytick.major.width": cfg["tick_lw"],
        "xtick.minor.size": cfg["tick_minor"], "ytick.minor.size": cfg["tick_minor"],
        "xtick.minor.width": cfg["tick_lw"] * 0.85, "ytick.minor.width": cfg["tick_lw"] * 0.85,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "savefig.dpi": cfg["dpi"], "savefig.bbox": "tight",
        "pdf.fonttype": 42, "svg.fonttype": "path",
    })


def _path(cfg, name):
    import os
    data_dir = cfg["data_dir"]
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(data_dir, name)


def load_across(cfg, d):
    a = np.loadtxt(_path(cfg, cfg["across_files"][d]))
    return a[np.argsort(a[:, cfg["across_cols"]["T"]])]


def load_pooled(cfg, d):
    p = np.loadtxt(_path(cfg, cfg["pooled_files"][d]))
    return p[np.argsort(p[:, cfg["pooled_cols"]["T"]])]


def load_ldt(cfg):
    return np.loadtxt(_path(cfg, cfg["ldt_file"]))


def load_rl(cfg):
    return np.loadtxt(_path(cfg, cfg["rl_file"]), comments="#")


def load_panelB(cfg):
    return pd.read_csv(_path(cfg, cfg["panelB_csv"]))


def load_panelC(cfg):
    return pd.read_csv(_path(cfg, cfg["panelC_csv"]))


def get_viridis_trimmed(cfg):
    """A viridis slice (trimmed to viridis_range) as a continuous colormap,
    so panels (a) and (b) can share one delta->colour mapping."""
    lo, hi = cfg["viridis_range"]
    stops = [plt.cm.viridis(lo + (hi - lo) * i / 255) for i in range(256)]
    return mcolors.LinearSegmentedColormap.from_list("viridis_trim", stops)


XLABEL_SCALED = r"$T/T_{c}^{\mathrm{eff}}(\delta)$"


def dlabel(d):
    """Legend label for a disorder strength; delta = 1.0 in the files is 0.999."""
    return r"$\delta=0.999$" if d == 1.0 else rf"$\delta={d:g}$"


def panel_tag(cfg, ax, tag):
    ax.text(-0.18, 1.04, tag, transform=ax.transAxes, fontsize=cfg["fs_panel"],
            ha="left", va="bottom")


# ------------------------------ panels ------------------------------
def draw_panel_a(cfg, fig, ax):
    """V_dis vs T/Tc, one curve per delta, with a colorbar and circle markers."""
    cmap = get_viridis_trimmed(cfg)
    norm = mpl.colors.Normalize(0.0, 1.0)
    c = cfg["across_cols"]
    ymax = 0.0
    for d in cfg["deltas"]:
        a = load_across(cfg, d)
        T = a[:, c["T"]]
        x = T / cfg["Tc"][d]
        y = a[:, c["V_dis_Jhat"]] / 1e-5
        col = cmap(norm(d))

        ax.plot(x, y, "-o", color=col, mfc=col, mec=col, ms=cfg["ms_a"],
                solid_capstyle="round", zorder=2)

        win = (x >= cfg["xlim_a"][0]) & (x <= cfg["xlim_a"][1])
        if win.any():
            ymax = max(ymax, y[win].max())
    ax.axvline(1.0, color="0.5", ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.set_xlim(*cfg["xlim_a"])
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xlabel(XLABEL_SCALED)
    ax.set_ylabel(r"$V_{\mathrm{disorder}}\,(\times 10^{-5})$")
    ax.set_xticks([0.9, 1.0, 1.1, 1.2])      # 0.8 omitted so it does not clash with y = 0
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    panel_tag(cfg, ax, "(a)")


def draw_panel_b(cfg, ax):
    """Cross term / delta^2 vs T/Tc, one curve per delta (fig4b_2CovLC_all_delta.csv)."""
    df = load_panelB(cfg)
    show = [d for d in cfg["b_show_deltas"] if d in df["delta"].unique()]
    cmap = get_viridis_trimmed(cfg)

    for d in show:
        col = cmap(d)
        sub = df[df["delta"] == d].sort_values("T")
        x = sub["T"].to_numpy() / cfg["Tc"][d]
        y = sub["TwoCov_over_delta2"].to_numpy() / 1e-5

        ax.plot(x, y, "-o", color=col, mfc=col, mec=col, ms=cfg["ms_cov"],
                zorder=2, label=dlabel(d))
    ax.axvline(1.0, color="0.5", ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.set_xlim(*cfg["xlim_a"])
    ax.set_xlabel(XLABEL_SCALED)
    # the plotted column is 2 Cov(L, C) = C_4^(3)(T) delta^4  (Eq. C25)
    ax.set_ylabel(r"$C_4^{(3)}(T)\,\delta^4\ \ (\times 10^{-5})$")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    panel_tag(cfg, ax, "(b)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4, labelspacing=0.28,
              fontsize=cfg["fs_legend"] * 0.9)


def draw_panel_c(cfg, ax):
    """Decomposition of V_disorder at delta=delta_show (precomputed CSV)."""
    df = load_panelC(cfg).sort_values("T")
    T = df["T"].to_numpy()
    VL = df["Var_L"].to_numpy() / 1e-5
    VdJ = df["Var_dJ"].to_numpy() / 1e-5
    twoCov = df["TwoCov_L_dJ"].to_numpy() / 1e-5
    total = df["Sum_Vdisorder"].to_numpy() / 1e-5

    def line_and_markers(y, color, marker, label):
        ax.plot(T, y, "-", marker=marker, color=color, mfc=color, mec=color,
                ms=cfg["ms_decomp"], lw=cfg["lw_curve"], zorder=2, label=label)

    line_and_markers(VL, cfg["col_VL"], "s", r"$\mathrm{Var(L)}$")
    line_and_markers(VdJ, cfg["col_VdJ"], "o", r"$\mathrm{Var(Q)} + \mathrm{O}(\delta^6)$")
    line_and_markers(twoCov, cfg["col_cov"], "^", r"$2\mathrm{Cov(L,C)} + \mathrm{O}(\delta^6)$")
    ax.plot(T, total, "-", color=cfg["col_sumC"], lw=cfg["lw_Vdis_thin"], zorder=3,
            label=r"$V_{\mathrm{disorder}}$")
    sum_recomputed = VL + VdJ + twoCov
    ax.plot(T, sum_recomputed, ls="none", marker="o", ms=cfg["ms_sum_c"],
            mfc=cfg["col_sumC"], mec=cfg["col_sumC"], zorder=4,
            label=r"$\mathrm{Var(L)} + \mathrm{Var(Q)} + 2\mathrm{Cov(L,C)} + \mathrm{O}(\delta^6)$")
    ax.axvline(cfg["Tc"][cfg["delta_show"]], color="0.5", ls=(0, (4, 3)), lw=1.0, zorder=0)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"variance $(\times 10^{-5})$")
    panel_tag(cfg, ax, "(c)")
    ax.legend(loc="center left", frameon=False, handlelength=1.2,
              labelspacing=0.3, fontsize=cfg["fs_legend"] * 0.85,
              bbox_to_anchor=(0.0, 0.52))


def draw_panel_d(cfg, ax):
    """NEW: the T at which V_dis and the cross term each reach their
    minimum, plotted against delta, alongside the clean T_c."""
    show = cfg["b_show_deltas"]
    ca = cfg["across_cols"]

    T_dis = []
    for d in show:
        a = load_across(cfg, d)
        T = a[:, ca["T"]]
        y = a[:, ca["V_dis_Jhat"]]
        T_dis.append(T[np.argmin(y)])

    df = load_panelB(cfg)
    T_cov = []
    for d in show:
        sub = df[df["delta"] == d].sort_values("T")
        T_cov.append(sub["T"].to_numpy()[np.argmin(sub["TwoCov_over_delta2"].to_numpy())])

    ax.plot(show, T_dis, "-s", ms=cfg["ms_d"], lw=cfg["lw_curve"], color=cfg["col_Tdis"],
            mfc=cfg["col_Tdis"], mec=cfg["col_Tdis"], zorder=3,
            label=r"$\arg\min_{T}\,\mathrm{V_{disorder}}$")
    ax.plot(show, T_cov, "-^", ms=cfg["ms_d"] * 1.55, lw=0, mfc="none",
            mec=cfg["col_Tcov"], mew=2.0, zorder=4,
            label=r"$\arg\min_{T}\,2\,\mathrm{Cov}(L,C)$")
    ax.plot(show, T_cov, "-", lw=cfg["lw_curve"], color=cfg["col_Tcov"], zorder=2)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$T$")
    ax.set_xlim(0.1, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    panel_tag(cfg, ax, "(d)")
    ax.legend(loc="lower left", frameon=False, handlelength=1.4, labelspacing=0.3,
              fontsize=cfg["fs_legend"] * 0.95)


def draw_panel_e(cfg, ax):
    """Raw-data variance decomposition at delta_show: V_sample, V_disorder,
    their sum, and V_consensus, built from the across/pooled files."""
    d = cfg["delta_show"]
    ca, cp = cfg["across_cols"], cfg["pooled_cols"]
    a = load_across(cfg, d)
    p = load_pooled(cfg, d)
    T = a[:, ca["T"]]
    V_samples = a[:, ca["V_sample_Jhat"]] * cfg["n_students"]
    V_dis = a[:, ca["V_dis_Jhat"]]
    V_cons = p[:, cp["var_J_hat"]]
    S = 1e-5

    ax.plot(T, V_samples / S, "-o", ms=cfg["ms_dot"], color=cfg["col_sample"],
            mfc=cfg["col_sample"], mec=cfg["col_sample"], zorder=5,
            label=r"$V_{\mathrm{samples}}$")
    ax.plot(T, V_dis / S, "-o", ms=cfg["ms_dot"], color=cfg["col_dis"],
            mfc=cfg["col_dis"], mec=cfg["col_dis"], zorder=5,
            label=r"$V_{\mathrm{disorder}}$")
    ax.plot(T, (V_samples + V_dis) / S, "-o", ms=cfg["ms_sum"], color=cfg["col_sum"],
            mfc="none", mec=cfg["col_sum"], mew=cfg["mew_sum"], zorder=6,
            label=r"$V_{\mathrm{samples}}+V_{\mathrm{disorder}}$")
    ax.plot(T, V_cons / S, "-", color=cfg["col_cons"], lw=cfg["lw_curve"] * 0.7,
            zorder=4, label=r"$V_{\mathrm{consensus}}$")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$(\times 10^{-5})$")
    ax.set_xlim(T.min(), T.max())
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, ylo + (yhi - ylo) * 1.22)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(MultipleLocator(0.125))
    ax.legend(loc="upper left", frameon=False, handlelength=1.5, labelspacing=0.3,
              bbox_to_anchor=(0.20, 1.0))
    ax.text(0.04, 0.90, rf"$\delta={d}$", transform=ax.transAxes,
            fontsize=cfg["fs_annot"], va="top")
    panel_tag(cfg, ax, "(e)")


def draw_panel_f(cfg, ax):
    """R(L) = V_dis / J_bar^2 vs L (log-log), real data all the way to L=256."""
    c = cfg["rl_cols"]
    data = load_rl(cfg)
    data = data[np.argsort(data[:, c["L"]])]
    L = data[:, c["L"]]
    R = data[:, c["R"]]

    p = np.polyfit(np.log2(L), np.log2(R), 1)
    slope = p[0]
    L_line = np.array([L.min(), L.max()])
    R_line = 2 ** (p[1] + slope * np.log2(L_line))

    ax.plot(L_line, R_line, "-", color="0.45", lw=1.8, zorder=1,
            label=rf"$\propto L^{{{slope:.2f}}}$")
    ax.plot(L, R, ls="none", marker="o", ms=11, mfc="none", mec="black",
            mew=1.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$R(L)$")
    ax.set_xticks(L)
    ax.set_xticklabels([f"{int(x)}" for x in L])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right", frameon=False, handlelength=1.5, labelspacing=0.35)
    panel_tag(cfg, ax, "(f)")


# ------------------------------ figure ------------------------------
def make_figure(cfg):
    apply_style(cfg)

    fig = plt.figure(figsize=(cfg["fig_w"], cfg["fig_h"]))
    gs = fig.add_gridspec(2, 3, height_ratios=cfg["height_ratios"],
                          wspace=cfg["wspace"], hspace=cfg["hspace"], **cfg["margins"])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_panel_a(cfg, fig, ax_a)
    draw_panel_b(cfg, ax_b)
    draw_panel_c(cfg, ax_c)
    draw_panel_d(cfg, ax_d)
    draw_panel_e(cfg, ax_e)
    draw_panel_f(cfg, ax_f)

    for ext in cfg["formats"]:
        fig.savefig(f"{cfg['savename']}.{ext}", dpi=cfg["dpi"])
    plt.close(fig)
    print(f"saved {cfg['savename']} as {', '.join(cfg['formats'])}")


if __name__ == "__main__":
    make_figure(CONFIG)
