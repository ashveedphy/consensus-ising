#!/usr/bin/env python3
"""
plot_fig2a.py -- Fig. 2(a): plot one x column against many y columns from a text file.

File format (whitespace separated; a text header line and '#' comments are
skipped automatically):

    x  y1  y2  ...  yn

Everything you might want to change lives in the CONFIG block below.
Nothing is hard-coded further down.

Run:
    python3 plot.py                 # uses CONFIG["palette"]
    python3 plot.py magma           # override the palette from the command line
    python3 plot.py all             # write one figure per palette

Note on LaTeX: CONFIG["usetex"]=True needs a TeX install plus dvipng and
cm-super. If you see "File `type1ec.sty' not found", install cm-super
(apt install cm-super dvipng) or set usetex=False to use matplotlib's
built-in Computer Modern lookalike.

---------------------------------------------------------------------
This version has been re-styled to match a reference MATLAB script:
  - log-log axes, box on all sides, inward ticks
  - viridis colormap, thick (lw=2) lines
  - larger fonts (tick=20, xlabel=36, ylabel=42, legend=24)
  - per-curve marker shape (s, o, d, ^, ...) with optional hollow markers
  - one curve forced to a manual color (MATLAB's `p4.Color='k'`)
  - an optional horizontal reference line (MATLAB's `yline(1)`)
  - an optional annotation textbox (MATLAB's `annotation('textbox', ...)`)
Anything MATLAB expresses in *normalized/axes-fraction* units (like
TickLength) had to be approximated in points, since matplotlib ticks are
sized in points, not as a fraction of the axes box.
---------------------------------------------------------------------
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MaxNLocator, FormatStrFormatter, LogLocator

# =====================================================================
#                             CONFIG
# =====================================================================
CONFIG = dict(

    # ---------- input ----------
    filename = "fig2a_Jbar_vs_T.dat",
    xcol     = 0,            # column index of x (0 = first)
    ycols    = None,         # None = every other column, or e.g. [1, 3, 5]

    # ---------- colours ----------
    palette    = "viridis",     # matches MATLAB's viridis(40)
    cmap_range = (0.05, 0.85),  # slice of the map to use (trim near-black/white ends;
                                 # helps when spreading 6 curves across the colormap)
    reverse_colors = False,     # True -> first curve gets the light end
    manual_colors  = {},        # index -> color override, e.g. {5: "k"}; empty = none

    # ---------- labels ----------
    xlabel = r"$T$",
    ylabel = r"$\bar{J}$",           # disorder-averaged inferred coupling
    curve_names = [r"$\delta=0$",   r"$\delta=0.2$", r"$\delta=0.4$",
                   r"$\delta=0.6$", r"$\delta=0.8$", r"$\delta=0.999$"],
    title     = "",          # heading above the axes, "" for none
    panel_tag = "",          # corner tag, "" for none
    tag_pos   = (0.045, 0.945),
    tag_ha    = "left",
    tag_va    = "top",
    tag_size  = 26,

    # ---------- figure ----------
    fig_w      = 7.5,        # inches
    fig_h      = 6.0,
    box_aspect = 0.75,       # height/width of the plot box; suits this linear T-range plot
    usetex     = True,

    # ---------- font sizes (points) ----------
    fs_base    = 20,         # tick numbers (MATLAB: set(axes1,'FontSize',20))
    fs_label   = 36,         # xlabel size  (MATLAB: xlabel FontSize 36)
    ylabel_size = 42,        # ylabel size, only used if different from fs_label
    fs_title   = 20,
    fs_legend  = 24,         # MATLAB: l1.FontSize = 24

    # ---------- lines ----------
    linewidth  = 2.0,        # MATLAB: LineWidth 2 on every plot() call
    linestyle  = "-",
    line_alpha = 1.0,

    # ---------- markers ----------
    marker        = "o",
    cycle_markers = True,
    marker_list   = ["s", "o", "d", "^", "v", "p", "h", "*"],  # matches MATLAB s/o/d/^ order
    markersize    = 9,        # smaller than the MATLAB reference (12) since this dataset
                              # has 37 closely-spaced points per curve, not ~38 log-spaced ones
    markevery     = 2,        # thin out markers so they don't merge into a solid line
    marker_edge_w = 1.5,
    marker_filled = True,     # global default fill state
    marker_filled_overrides = {},   # per-curve override (0-based index -> bool); empty = all filled

    # ---------- axes ----------
    log_x = False,            # this dataset spans a narrow linear range (T = 1.6-3.0)
    log_y = False,
    xlim  = (1.6, 3.0),
    ylim  = (0.80, 1.01),
    ypad  = 0.02,             # ignored when ylim is set

    xticks = None,            # auto (linear locator below); set an explicit list if you prefer
    yticks = None,
    n_xticks  = 5,
    n_yticks  = 6,
    xtick_fmt = "%.1f",
    ytick_fmt = "%.2f",
    tick_mode = "nice",

    minor_ticks   = True,
    ticks_inside  = True,     # MATLAB box on all sides + inward ticks
    grid          = False,

    # ---------- line/tick weights ----------
    axes_lw = 2.0,            # MATLAB: set(axes1,'LineWidth',2) -> box outline
    tick_lw = 2.0,
    tick_major = 10.0,        # MATLAB TickLength is a *fraction of the axes box*
    tick_minor = 6.0,         # ([0.015 0.025]); these point-values are a visual
                              # approximation, since matplotlib has no direct
                              # equivalent unit.

    # ---------- legend ----------
    legend       = True,
    legend_loc   = "lower right",   # all 6 curves rise left-to-right, so lower-right is clearest
    legend_ncol  = 2,
    legend_frame = False,           # no box around the legend (as in the paper)

    # ---------- reference line (MATLAB: yline(1)) ----------
    hline = 1.0,              # y-value for a horizontal guide line, None to disable
    hline_style = dict(color="0.5", linestyle="-.", linewidth=2.0),

    # ---------- annotation textbox (MATLAB: annotation('textbox', ...)) ----------
    annotation_text = None,
        # e.g. r"$N=100,\ v=100$" "\n" r"$c=10^{7}$"
    annotation_pos  = (0.5, 0.02),   # axes-fraction (x, y) of the box's anchor corner
    annotation_size = 24,

    # ---------- output ----------
    savename = "Fig2a",
    formats  = ("pdf", "png"),
    dpi      = 300,
)

# =====================================================================
#                       PLOTTING (no need to edit)
# =====================================================================
PALETTES = ("viridis", "plasma", "magma")


def load(cfg):
    """Read the file, skipping '#' comments and a text header line."""
    data = np.genfromtxt(cfg["filename"], comments="#")
    if np.isnan(data[0]).any():      # a text header became NaNs
        data = data[1:]
    x = data[:, cfg["xcol"]]
    ycols = cfg["ycols"] or [i for i in range(data.shape[1])
                             if i != cfg["xcol"]]
    return x, data, ycols


def get_colors(cfg, n):
    cmap = plt.get_cmap(cfg["palette"])
    lo, hi = cfg["cmap_range"]
    cols = [cmap(lo + (hi - lo) * i / max(n - 1, 1)) for i in range(n)]
    cols = cols[::-1] if cfg["reverse_colors"] else cols
    manual = cfg.get("manual_colors") or {}
    for idx, color in manual.items():
        if 0 <= idx < len(cols):
            cols[idx] = color
    return cols


def apply_style(cfg):
    plt.rcParams.update({
        "text.usetex": cfg["usetex"],
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": cfg["fs_base"],
        "axes.labelsize": cfg["fs_label"],
        "axes.titlesize": cfg["fs_title"],
        "legend.fontsize": cfg["fs_legend"],
        "xtick.labelsize": cfg["fs_base"],
        "ytick.labelsize": cfg["fs_base"],
        "axes.linewidth": cfg["axes_lw"],
        "xtick.major.width": cfg["tick_lw"], "ytick.major.width": cfg["tick_lw"],
        "xtick.minor.width": cfg["tick_lw"] * 0.8,
        "ytick.minor.width": cfg["tick_lw"] * 0.8,
        "xtick.major.size": cfg["tick_major"], "ytick.major.size": cfg["tick_major"],
        "xtick.minor.size": cfg["tick_minor"], "ytick.minor.size": cfg["tick_minor"],
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    if cfg["usetex"]:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def make_figure(cfg):
    apply_style(cfg)
    x, data, ycols = load(cfg)
    colors = get_colors(cfg, len(ycols))
    fill_overrides = cfg.get("marker_filled_overrides") or {}

    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]))

    for i, c in enumerate(ycols):
        name = (cfg["curve_names"][i] if i < len(cfg["curve_names"])
                else f"column {c}")
        mk = (cfg["marker_list"][i % len(cfg["marker_list"])]
              if cfg["cycle_markers"] else cfg["marker"])
        filled = fill_overrides.get(i, cfg["marker_filled"])
        ax.plot(x, data[:, c], label=name, color=colors[i],
                linewidth=cfg["linewidth"], linestyle=cfg["linestyle"],
                alpha=cfg["line_alpha"],
                marker=mk, markersize=cfg["markersize"],
                markevery=cfg["markevery"],
                markerfacecolor=colors[i] if filled else "none",
                markeredgecolor=colors[i],
                markeredgewidth=cfg["marker_edge_w"])

    if cfg.get("hline") is not None:
        ax.axhline(cfg["hline"], **cfg["hline_style"])
        ax.get_lines()[-1].set_label("_nolegend_")

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])
    if cfg.get("ylabel_size"):
        ax.yaxis.label.set_fontsize(cfg["ylabel_size"])

    if cfg["log_x"]:
        ax.set_xscale("log")
    if cfg["log_y"]:
        ax.set_yscale("log")

    ax.margins(y=cfg["ypad"])
    if cfg["xlim"]:
        ax.set_xlim(cfg["xlim"])
    if cfg["ylim"]:
        ax.set_ylim(cfg["ylim"])

    def locator(n):
        if cfg["tick_mode"] == "exact":
            return LinearLocator(n)
        return MaxNLocator(nbins=n, steps=[1, 2, 2.5, 4, 5, 10])

    if cfg.get("xticks") is not None:
        ax.set_xticks(cfg["xticks"])
    elif not cfg["log_x"] and cfg["n_xticks"]:
        ax.xaxis.set_major_locator(locator(cfg["n_xticks"]))
        ax.xaxis.set_major_formatter(FormatStrFormatter(cfg["xtick_fmt"]))
    elif cfg["log_x"]:
        ax.xaxis.set_major_locator(LogLocator(base=10))

    if cfg.get("yticks") is not None:
        ax.set_yticks(cfg["yticks"])
    elif not cfg["log_y"] and cfg["n_yticks"]:
        ax.yaxis.set_major_locator(locator(cfg["n_yticks"]))
        ax.yaxis.set_major_formatter(FormatStrFormatter(cfg["ytick_fmt"]))
    elif cfg["log_y"]:
        ax.yaxis.set_major_locator(LogLocator(base=10))

    if cfg["minor_ticks"]:
        ax.minorticks_on()
    if cfg["ticks_inside"]:
        ax.tick_params(which="both", direction="in", top=True, right=True)
    if cfg["grid"]:
        ax.grid(True, alpha=0.3)
    if cfg["box_aspect"]:
        ax.set_box_aspect(cfg["box_aspect"])
    if cfg["title"]:
        ax.set_title(cfg["title"], pad=10)
    if cfg["panel_tag"]:
        ax.text(*cfg["tag_pos"], cfg["panel_tag"], transform=ax.transAxes,
                fontsize=cfg["tag_size"], ha=cfg["tag_ha"], va=cfg["tag_va"])

    if cfg.get("annotation_text"):
        ax.text(*cfg["annotation_pos"], cfg["annotation_text"],
                transform=ax.transAxes, fontsize=cfg["annotation_size"],
                ha="center", va="bottom",
                bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                          edgecolor="black", linewidth=2))

    if cfg["legend"]:
        ax.legend(loc=cfg["legend_loc"], ncol=cfg["legend_ncol"],
                  frameon=cfg["legend_frame"], handlelength=1.2,
                  labelspacing=0.35, borderaxespad=0.5)

    for ext in cfg["formats"]:
        fig.savefig(f"{cfg['savename']}.{ext}", dpi=cfg["dpi"])
    plt.close(fig)
    print(f"saved {cfg['savename']} as {', '.join(cfg['formats'])}"
          f"   (palette: {cfg['palette']})")


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if arg == "all":                       # one figure per palette
        for p in PALETTES:
            cfg = dict(CONFIG, palette=p, savename=f"{CONFIG['savename']}_{p}")
            make_figure(cfg)
    elif arg in PALETTES:
        make_figure(dict(CONFIG, palette=arg))
    elif arg:
        sys.exit(f"unknown palette '{arg}'. choose from: {', '.join(PALETTES)}")
    else:
        make_figure(CONFIG)
