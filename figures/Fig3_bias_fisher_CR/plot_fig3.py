#!/usr/bin/env python3
"""
plot_fig3.py -- Fig. 3, five-panel estimator figure, built in the same config-driven style
as plot.py: everything you might want to change lives in the CONFIG block,
nothing is hard-coded further down.

Panels
    (a) b(J_0)   vs T, one curve per delta          (line + filled circles)
    (b) b'(J_0)  vs T, one curve per delta          (line + smaller, denser
        filled circles)
    (c) I(J_0)   vs T, one curve per delta          (carries the delta legend)
    (d) Var(Jhat) and the two CR bounds vs sample size n, at selected T.
        Temperature is encoded by colour; quantity by line/marker shape.
    (e) a 1xN row of small panels, Var(Jhat) and CR bounds vs T/Tc,
        one panel per delta. Var(Jhat) is a black line with filled circles
        riding on top of it (denser than before).

Panels (a)-(c) are generated from the CONFIG["top_panels"] list, so adding,
removing or reordering them is a config edit, not a code edit. Each entry may
override "linestyle", "marker", "markersize", "markevery" and "marker_filled";
anything it omits falls back to the CONFIG["top_*"] defaults.

Input files (whitespace separated, '#' comments skipped)
    fig3_bias_theory_and_measured.dat : 0=disorder 1=T 2=delta 3=K_T 4=bias_O2 5=bias_O2O4
                        6=bias_measured 7=b_prime 8=J_star 9=J_eff
    fig3_across_delta*.dat : 0=T 1=mean_J_hat 2=V_disorder 3=V_sample
                        4=mean_Fisher 5=realisations
    fig3_sample_size_sweep_delta0.500.dat : 0=n 1=T 2=mean_J_hat 3=V_disorder 4=V_sample
                        5=mean_Fisher 6=realisations

Derived quantities
    V_samples     = V_sample
    CR(T)         = 1 / (n * I)            (unbiased Cramer-Rao bound)
    BCR(T)        = (1 + b')^2 / (n * I)   (biased Cramer-Rao bound, Eq. 38)

Run
    python3 fig3.py                 # uses CONFIG["palette"]
    python3 fig3.py viridis         # override the delta palette
    python3 fig3.py all             # one figure per palette in PALETTES
"""

import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.lines import Line2D

# =====================================================================
#                             CONFIG
# =====================================================================
CONFIG = dict(

    # ---------- input ----------
    bias_file   = "fig3_bias_theory_and_measured.dat",
    all_file    = "fig3_sample_size_sweep_delta0.500.dat",
    across_files = {0.0: "fig3_across_delta0.000.dat", 0.2: "fig3_across_delta0.200.dat",
                    0.4: "fig3_across_delta0.400.dat", 0.6: "fig3_across_delta0.600.dat",
                    0.8: "fig3_across_delta0.800.dat", 1.0: "fig3_across_delta0.999.dat"},

    # column indices, by name -- change these if the file layout changes
    bias_cols   = dict(disorder=0, T=1, delta=2, K_T=3, bias_O2=4,
                       bias_O2O4=5, bias_measured=6, b_prime=7,
                       J_star=8, J_eff=9),
    across_cols = dict(T=0, mean_J_hat=1, V_disorder=2, V_sample=3,
                       mean_Fisher=4, realisations=5),
    all_cols    = dict(n=0, T=1, mean_J_hat=2, V_disorder=3, V_sample=4,
                       mean_Fisher=5, realisations=6),

    # ---------- physics / series selection ----------
    deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    Tc     = {0.0: 2.28, 0.2: 2.28, 0.4: 2.24, 0.6: 2.20, 0.8: 2.16, 1.0: 2.04},
    n_sample   = 90,      # n used for the CR bounds in row (e)
    bias_delta = 1.0,     # which delta slice supplies b'(J_0) in panel (d)
    Tlim       = (1.6, 3.0),   # raw-T window for panels (a)-(c)

    # ---------- colours ----------
    palette   = "viridis",   # key into PALETTES below; sets the delta colours
    cmap_range = (0.05, 0.85), # trim the near-black/near-white ends so all 6
                                # delta curves stay legible against the page
    reverse_colors = False,
    manual_colors  = {},     # delta -> colour override, e.g. {1.0: "k"}

    col_var = "#1A1A1A",     # Var(Jhat)      in row (e)
    col_cru = "#1B6FE3",     # CR (unbiased)  in row (e)
    col_crb = "#D81B60",     # CR (biased)    in row (e)

    # ---------- panels (a)-(c) ----------
    # source: "bias" -> fig3_bias_theory_and_measured.dat, "across" -> fig3_across_delta*.dat
    # per-panel marker/line keys override the top_* defaults below.
    # (a) and (b) are drawn as filled circles with a connecting line; set
    # linestyle="none" on an entry if you want markers only.
    top_panels = [
        dict(tag="(a)", source="bias",   column="bias_O2",
             ylabel=r"$b(\delta,T)$",  ymajor=0.05, legend=False,
             linestyle="-", marker="o", marker_filled=True),
        dict(tag="(b)", source="bias",   column="b_prime",
             ylabel=r"$b'(J_0)$", ymajor=0.01, legend=False,
             linestyle="-", marker="o", marker_filled=True,
             markersize=5, markevery=1),
             # smaller than the top_markersize default (8 -> 5) and drawn at
             # every point (markevery 2 -> 1) so shrinking them is compensated
             # by roughly doubling how many are shown
        dict(tag="(c)", source="across", column="mean_Fisher",
             ylabel=r"$\langle \mathcal{I}_\omega(T)\rangle_\omega$",  ymajor=None, legend=True,
             legend_loc="upper right", linestyle="-", marker="none"),
    ],
    top_xlabel = r"$T$",
    top_ylabel_x = None,     # None = automatic; or an axes-fraction x position, e.g. -0.3
    top_ylabel_pad = 4,      # extra space (points) between y label and tick numbers
    abc_xmajor = 0.4,        # spacing between labelled x-ticks
    abc_xminor = 0.125,      # spacing between minor x-ticks

    # defaults for the top panels; individual entries may override any of these
    top_linestyle     = "-",
    top_marker        = "none",
    top_markersize    = 8,
    top_markevery     = 2,   # 1 = a circle on every data point
    top_marker_edge_w = 1.2,
    top_marker_filled = True,

    # ---------- panel (d) ----------
    d_tag        = "(d)",
    d_temps      = [1.60, 2.04, 3.00],
    d_temp_colors = {1.60: "#D62728", 2.04: "#1F5FD6", 3.00: "#2CA02C"},
    d_temp_fmt   = r"$T={:.2f}$",
    d_xlabel     = r"$n$",
    d_ylabel     = "",                      # no y label (as in the paper)
    d_log_y      = True,
    d_log_x      = False,
    d_marker_unb = "s",      # CR (unbiased)
    d_marker_bia = "o",      # CR (biased)
    d_legend_temp_loc = "upper right",
    d_legend_qty_loc  = "upper left",
    d_legend_qty_anchor = (0.14, 1.0),   # axes-fraction anchor of the quantity legend

    # ---------- row (e) ----------
    e_tag        = "(e)",
    e_tag_pos    = (-0.46, 1.12),   # axes-fraction, relative to the first panel
    e_xlabel     = r"$T/T_{c}^{\mathrm{eff}}(\delta)$",
    e_scale      = 1e-5,            # y values are divided by this
    e_ylabel     = r"$(\times 10^{-5})$",   # shown on the leftmost panel only
    e_xlim       = (0.83, 1.22),
    e_ylim       = (0.0, 2.7),      # in units of e_scale
    e_share_y    = True,            # False -> independent y-scale per panel
    e_xmajor     = 0.2,
    e_xminor     = 0.1,
    e_marker_bia = "o",
    e_box_aspect = 1.0,
    e_legend_panel = 0,             # index of the panel carrying the legend
    e_legend_loc   = "upper left",

    # filled circles riding on top of the black Var(Jhat) line
    e_marker_var     = "o",
    e_var_markevery  = 1,           # was 3; now every point (~doubled, since
                                     # matplotlib only steps in whole indices --
                                     # the nearest way to at least double the
                                     # count; drop to 2 if it looks too busy
                                     # on your real (denser) data)
    e_var_filled     = True,        # False -> hollow circles on the line

    # ---------- labels shared by (d) and (e) ----------
    lab_var = r"$V_{\mathrm{samples}}$",
    lab_cru = r"CR $(T)$",
    lab_crb = r"BCR $(T)$",

    # ---------- figure / layout ----------
    fig_w        = 20.5,     # inches
    fig_h        = 8.5,
    height_ratios = [1.15, 1.0],
    top_wspace   = 6.0,      # gap between panels (a)-(d)
    row_hspace   = 0.45,     # gap between the top row and row (e)
    e_wspace     = 0.00,     # gap between the row-(e) panels
    margins      = dict(left=0.055, right=0.985, top=0.95, bottom=0.12),
    usetex       = False,    # True needs a TeX install + dvipng + cm-super

    # ---------- font sizes (points) ----------
    fs_label   = 30,   # axis labels in (a)-(d)
    fs_tick    = 24,   # tick numbers in (a)-(d)
    fs_tag     = 28,   # the (a),(b),(c),(d) tags
    fs_tag_e   = 28,   # the (e) tag
    fs_legend_c = 17,
    fs_legend_d = 15,
    fs_legend_e = 17,
    fs_label_e = 28,   # axis labels in row (e)
    fs_tick_e  = 24,   # tick numbers in row (e)
    fs_title_e = 24,   # the delta=... titles in row (e)

    # ---------- lines ----------
    lw_top     = 2.6,   # curves in (a)-(c)
    lw_d_var   = 2.4,   # Var(Jhat) line in (d)
    lw_e_var   = 2.2,   # Var(Jhat) line in (e)
    lw_e_cru   = 2.2,   # CR (unbiased) line in (e)
    lw_legend  = 2.8,   # swatch width in the panel-(d) temperature legend

    # ---------- markers ----------
    ms_d        = 7,     # CR markers in (d)
    mew_d       = 1.7,
    ms_d_legend = 8,
    ms_e        = 6.0,   # CR (biased) markers in (e)
    mew_e       = 1.5,
    ms_e_var    = 5.5,   # circles on the Var(Jhat) line in (e)
    mew_e_var   = 1.2,
    marker_filled = False,   # True -> solid CR markers instead of open ones

    # ---------- axes furniture ----------
    axes_lw    = 1.4,
    tick_lw    = 1.3,
    tick_major = 7.0,
    tick_minor = 3.6,
    ticks_inside = True,
    minor_ticks  = True,
    legend_frame = False,

    # ---------- output ----------
    savename = "Fig3",
    formats  = ("pdf", "svg", "png"),
    dpi      = 400,
)

# =====================================================================
#                       PLOTTING (no need to edit)
# =====================================================================
# Named colour maps for the delta curves. Anything matplotlib knows by name
# also works; the tuples below are custom lists of stops.
PALETTES = {
    "redblue": ["#8B0000", "#E23B27", "#F09177", "#7AA6D2", "#2C6FB5", "#0A2F6B"],
    "viridis": "viridis",
    "plasma":  "plasma",
    "magma":   "magma",
}


def dlabel(d):
    """delta=0 without a trailing decimal; the largest disorder (stored as 1.0) is 0.999."""
    if d == 1.0:
        return r"$\delta=0.999$"
    if float(d).is_integer():
        return rf"$\delta={int(d)}$"
    return rf"$\delta={d}$"


def panel_tag(cfg, ax, tag):
    """(a), (b), ... in regular type above the top-left corner of a panel."""
    ax.text(-0.22, 1.03, tag, transform=ax.transAxes, fontsize=cfg["fs_tag"],
            ha="left", va="bottom")


def get_cmap(cfg):
    spec = PALETTES.get(cfg["palette"], cfg["palette"])
    if isinstance(spec, str):
        return plt.get_cmap(spec)
    return mcolors.LinearSegmentedColormap.from_list("delta_map", spec)


def get_colors(cfg):
    """delta -> RGBA, spread over the requested slice of the colormap."""
    cmap = get_cmap(cfg)
    deltas = cfg["deltas"]
    lo, hi = cfg["cmap_range"]
    n = len(deltas)
    frac = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
    frac = frac[::-1] if cfg["reverse_colors"] else frac
    cols = {d: cmap(f) for d, f in zip(deltas, frac)}
    cols.update(cfg.get("manual_colors") or {})
    return cols


def load(cfg):
    """Read every input file once; return a small bundle of arrays."""
    bias = np.loadtxt(cfg["bias_file"])
    allsw = np.loadtxt(cfg["all_file"])
    across = {d: np.loadtxt(f) for d, f in cfg["across_files"].items()}
    return dict(bias=bias, allsw=allsw, across=across)


def bias_series(cfg, data, d, column):
    """(T, y) from fig3_bias_theory_and_measured.dat for one disorder value, sorted by T."""
    c = cfg["bias_cols"]
    cb = data["bias"]
    m = np.isclose(cb[:, c["disorder"]], d)
    T, y = cb[m, c["T"]], cb[m, c[column]]
    o = np.argsort(T)
    return T[o], y[o]


def across_series(cfg, data, d, column):
    """(T, y) from fig3_across_delta*.dat for one disorder value, sorted by T."""
    c = cfg["across_cols"]
    a = data["across"][d]
    T, y = a[:, c["T"]], a[:, c[column]]
    o = np.argsort(T)
    return T[o], y[o]


def cr_bounds(n, fisher, b_prime):
    """Cramer-Rao bounds, unbiased and bias-corrected."""
    return 1.0 / (n * fisher), (1.0 + b_prime) ** 2 / (n * fisher)


def apply_style(cfg):
    plt.rcParams.update({
        "text.usetex": cfg["usetex"],
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.formatter.use_mathtext": True,
        "axes.titlesize": cfg["fs_tag"],
        "axes.labelsize": cfg["fs_label"],
        "xtick.labelsize": cfg["fs_tick"], "ytick.labelsize": cfg["fs_tick"],
        "legend.fontsize": cfg["fs_legend_e"],
        "axes.linewidth": cfg["axes_lw"], "lines.linewidth": cfg["lw_top"],
        "xtick.direction": "in" if cfg["ticks_inside"] else "out",
        "ytick.direction": "in" if cfg["ticks_inside"] else "out",
        "xtick.top": cfg["ticks_inside"], "ytick.right": cfg["ticks_inside"],
        "xtick.major.size": cfg["tick_major"], "ytick.major.size": cfg["tick_major"],
        "xtick.major.width": cfg["tick_lw"], "ytick.major.width": cfg["tick_lw"],
        "xtick.minor.size": cfg["tick_minor"], "ytick.minor.size": cfg["tick_minor"],
        "xtick.minor.visible": cfg["minor_ticks"],
        "ytick.minor.visible": cfg["minor_ticks"],
        "savefig.dpi": cfg["dpi"], "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    if cfg["usetex"]:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def style_axes(cfg, ax):
    ax.tick_params(which="both", top=cfg["ticks_inside"],
                   right=cfg["ticks_inside"])


def mfc(cfg, color):
    """Marker face colour: honours cfg['marker_filled']."""
    return color if cfg["marker_filled"] else "none"


# ------------------------------ panels ------------------------------
def draw_top_panel(cfg, data, ax, spec, colors):
    """One of the CONFIG['top_panels'] entries: a quantity vs T per delta."""
    reader = bias_series if spec["source"] == "bias" else across_series
    lo, hi = cfg["Tlim"]

    ls = spec.get("linestyle", cfg["top_linestyle"])
    mk = spec.get("marker", cfg["top_marker"])
    ms = spec.get("markersize", cfg["top_markersize"])
    mev = spec.get("markevery", cfg["top_markevery"])
    mew = spec.get("marker_edge_w", cfg["top_marker_edge_w"])
    filled = spec.get("marker_filled", cfg["top_marker_filled"])

    for d in cfg["deltas"]:
        T, y = reader(cfg, data, d, spec["column"])
        m = (T >= lo) & (T <= hi)
        ax.plot(T[m], y[m], color=colors[d], lw=cfg["lw_top"], ls=ls,
                marker=mk, ms=ms, markevery=mev,
                mfc=colors[d] if filled else "none", mec=colors[d], mew=mew,
                zorder=3, label=dlabel(d))

    ax.set_xlabel(spec.get("xlabel", cfg["top_xlabel"]))
    ax.set_ylabel(spec["ylabel"], labelpad=cfg["top_ylabel_pad"])
    if cfg["top_ylabel_x"] is not None:
        ax.yaxis.set_label_coords(cfg["top_ylabel_x"], 0.5)
    ax.set_xlim(*cfg["Tlim"])
    ax.xaxis.set_major_locator(MultipleLocator(cfg["abc_xmajor"]))
    ax.xaxis.set_minor_locator(MultipleLocator(cfg["abc_xminor"]))
    if spec.get("ymajor"):
        ax.yaxis.set_major_locator(MultipleLocator(spec["ymajor"]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    if spec.get("log_y"):
        ax.set_yscale("log")
    style_axes(cfg, ax)
    panel_tag(cfg, ax, spec["tag"])
    if spec.get("legend"):
        ax.legend(loc=spec.get("legend_loc", "upper right"),
                  frameon=cfg["legend_frame"], handlelength=1.3,
                  labelspacing=0.3, borderaxespad=0.4,
                  fontsize=cfg["fs_legend_c"])


def draw_panel_d(cfg, data, ax):
    """Var(Jhat) and the CR bounds vs n, one colour per temperature."""
    c = cfg["all_cols"]
    aall = data["allsw"]
    Tb, bp = bias_series(cfg, data, cfg["bias_delta"], "b_prime")

    for T in cfg["d_temps"]:
        m = np.isclose(aall[:, c["T"]], T)
        n = aall[m, c["n"]]
        var_jhat = aall[m, c["V_sample"]]
        fisher = aall[m, c["mean_Fisher"]]
        o = np.argsort(n)
        n, var_jhat, fisher = n[o], var_jhat[o], fisher[o]

        cru, crb = cr_bounds(n, fisher, np.interp(T, Tb, bp))
        col = cfg["d_temp_colors"][T]

        ax.plot(n, var_jhat, "-", color=col, lw=cfg["lw_d_var"], zorder=2)
        ax.plot(n, cru, ls="none", marker=cfg["d_marker_unb"], ms=cfg["ms_d"],
                mfc=mfc(cfg, col), mec=col, mew=cfg["mew_d"], zorder=3)
        ax.plot(n, crb, ls="none", marker=cfg["d_marker_bia"], ms=cfg["ms_d"],
                mfc=mfc(cfg, col), mec=col, mew=cfg["mew_d"], zorder=4)

    ax.set_xlabel(cfg["d_xlabel"])
    if cfg["d_ylabel"]:
        ax.set_ylabel(cfg["d_ylabel"])
    if cfg["d_log_y"]:
        ax.set_yscale("log")
    if cfg["d_log_x"]:
        ax.set_xscale("log")
    style_axes(cfg, ax)
    panel_tag(cfg, ax, cfg["d_tag"])

    # two-part legend: colour = temperature, marker = quantity
    temp_handles = [Line2D([0], [0], color=cfg["d_temp_colors"][T],
                           lw=cfg["lw_legend"], label=cfg["d_temp_fmt"].format(T))
                    for T in cfg["d_temps"]]
    qty_handles = [
        Line2D([0], [0], color="0.25", lw=cfg["lw_d_var"], ls="-",
               label=cfg["lab_var"]),
        Line2D([0], [0], color="0.25", ls="none", marker=cfg["d_marker_unb"],
               mfc=mfc(cfg, "0.25"), mec="0.25", mew=cfg["mew_d"],
               ms=cfg["ms_d_legend"], label=cfg["lab_cru"]),
        Line2D([0], [0], color="0.25", ls="none", marker=cfg["d_marker_bia"],
               mfc=mfc(cfg, "0.25"), mec="0.25", mew=cfg["mew_d"],
               ms=cfg["ms_d_legend"], label=cfg["lab_crb"]),
    ]
    leg = ax.legend(handles=temp_handles, loc=cfg["d_legend_temp_loc"],
                    frameon=cfg["legend_frame"], handlelength=1.3,
                    labelspacing=0.25, fontsize=cfg["fs_legend_d"])
    ax.add_artist(leg)
    ax.legend(handles=qty_handles, loc=cfg["d_legend_qty_loc"],
              bbox_to_anchor=cfg["d_legend_qty_anchor"],
              frameon=cfg["legend_frame"], handlelength=1.6,
              labelspacing=0.25, fontsize=cfg["fs_legend_d"])


def draw_row_e(cfg, data, axes):
    """One small panel per delta: Var(Jhat) and the CR bounds vs T/Tc."""
    c = cfg["across_cols"]
    s = cfg["e_scale"]
    N = cfg["n_sample"]

    for i, d in enumerate(cfg["deltas"]):
        ax = axes[i]
        a = data["across"][d]
        T = a[:, c["T"]]
        x = T / cfg["Tc"][d]
        var_jhat = a[:, c["V_sample"]]
        fisher = a[:, c["mean_Fisher"]]

        Tb, bp = bias_series(cfg, data, d, "b_prime")
        cru, crb = cr_bounds(N, fisher, np.interp(T, Tb, bp))

        o = np.argsort(x)
        # Var(Jhat): black line with circles riding on top of it
        ax.plot(x[o], var_jhat[o] / s, "-", color=cfg["col_var"],
                lw=cfg["lw_e_var"], marker=cfg["e_marker_var"],
                ms=cfg["ms_e_var"], markevery=cfg["e_var_markevery"],
                mfc=cfg["col_var"] if cfg["e_var_filled"] else "none",
                mec=cfg["col_var"], mew=cfg["mew_e_var"],
                zorder=5, label=cfg["lab_var"])
        ax.plot(x[o], cru[o] / s, "-", color=cfg["col_cru"],
                lw=cfg["lw_e_cru"], zorder=2, label=cfg["lab_cru"])
        ax.plot(x[o], crb[o] / s, ls="none", marker=cfg["e_marker_bia"],
                ms=cfg["ms_e"], mfc=mfc(cfg, cfg["col_crb"]), mec=cfg["col_crb"],
                mew=cfg["mew_e"], zorder=4, label=cfg["lab_crb"])

        ax.set_title(dlabel(d), pad=6, fontsize=cfg["fs_title_e"])
        ax.set_xlabel(cfg["e_xlabel"], fontsize=cfg["fs_label_e"])
        if cfg["e_xlim"]:
            ax.set_xlim(*cfg["e_xlim"])
        if cfg["e_ylim"]:
            ax.set_ylim(*cfg["e_ylim"])
        if i == 0:
            ax.set_ylabel(cfg["e_ylabel"], fontsize=cfg["fs_label_e"])
        ax.xaxis.set_major_locator(MultipleLocator(cfg["e_xmajor"]))
        ax.xaxis.set_minor_locator(MultipleLocator(cfg["e_xminor"]))
        ax.tick_params(axis="both", which="major", labelsize=cfg["fs_tick_e"])
        if cfg["e_box_aspect"]:
            ax.set_box_aspect(cfg["e_box_aspect"])
        style_axes(cfg, ax)

        if i == cfg["e_legend_panel"]:
            ax.legend(loc=cfg["e_legend_loc"], frameon=cfg["legend_frame"],
                      handlelength=1.4, labelspacing=0.3, borderaxespad=0.3,
                      fontsize=cfg["fs_legend_e"])
        if i == 0 and cfg["e_tag"]:
            ax.text(*cfg["e_tag_pos"], cfg["e_tag"], transform=ax.transAxes,
                    fontsize=cfg["fs_tag_e"], ha="left", va="bottom")


# ------------------------------ figure ------------------------------
def make_figure(cfg):
    apply_style(cfg)
    data = load(cfg)
    colors = get_colors(cfg)

    n_top = len(cfg["top_panels"]) + 1        # (a)-(c) plus panel (d)
    n_e = len(cfg["deltas"])
    ncol = 3 * n_top                          # 3 grid columns per top panel

    fig = plt.figure(figsize=(cfg["fig_w"], cfg["fig_h"]))
    gs = fig.add_gridspec(2, ncol, height_ratios=cfg["height_ratios"],
                          hspace=cfg["row_hspace"], wspace=cfg["top_wspace"],
                          **cfg["margins"])

    top_axes = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(n_top)]
    for ax, spec in zip(top_axes, cfg["top_panels"]):
        draw_top_panel(cfg, data, ax, spec, colors)
    draw_panel_d(cfg, data, top_axes[-1])

    gse = gs[1, :].subgridspec(1, n_e, wspace=cfg["e_wspace"])
    e_axes = []
    for i in range(n_e):
        if i == 0 or not cfg["e_share_y"]:
            ax = fig.add_subplot(gse[0, i])
        else:
            ax = fig.add_subplot(gse[0, i], sharey=e_axes[0])
            ax.tick_params(labelleft=False)
        e_axes.append(ax)
    draw_row_e(cfg, data, e_axes)

    for ext in cfg["formats"]:
        fig.savefig(f"{cfg['savename']}.{ext}", dpi=cfg["dpi"])
    plt.close(fig)
    print(f"saved {cfg['savename']} as {', '.join(cfg['formats'])}"
          f"   (palette: {cfg['palette']})")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if arg == "all":
        for p in PALETTES:
            make_figure(dict(CONFIG, palette=p,
                             savename=f"{CONFIG['savename']}_{p}"))
    elif arg in PALETTES:
        make_figure(dict(CONFIG, palette=arg))
    elif arg:
        sys.exit(f"unknown palette '{arg}'. choose from: {', '.join(PALETTES)}")
    else:
        make_figure(CONFIG)
