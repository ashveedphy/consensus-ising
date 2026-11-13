"""
plot_fig2b.py -- Fig. 2(b): distribution of the estimator J-hat at three temperatures.
Reads fig2b_students_T1.600.csv, fig2b_students_T2.240.csv, fig2b_students_T3.000.csv
(columns: realisation, student_id, J_star, RC; J_hat = T * J_star).
Publication-grade figure (PRL / Nature, double-column ~7 in wide).

All visual knobs live in the STYLE CONTROLS block — nothing else needs editing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import scienceplots  # noqa: F401  (registers the 'science' style)


def tint(color, amount):
    """Blend a colour toward white. amount=0 keeps it, amount=1 -> white."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

# ============================================================
#  DATA CONFIG
# ============================================================
TEMPS = ["1.600", "2.240", "3.000"]
TEMP_LABELS = {
    "1.600": r"$T = 1.60 < T_c$",
    "2.240": r"$T = 2.24 \approx T_c$",
    "3.000": r"$T = 3.00 > T_c$",
}
PANEL_TAGS = []                             # e.g. ["(a)", "(b)", "(c)"]; none in the paper

# ---- Colour palette --------------------------------------------------------
# Specify ONE deep base colour per panel (used for the Gaussian curve and bar
# edges); the soft histogram fill is auto-derived from it via FILL_TINT, so the
# hues always stay perfectly consistent.
#
# All three presets below are colour-blind-safe and survive grayscale / CMYK
# printing. Just point PANEL_COLORS at the one you want.
PALETTES = {
    "tol_vibrant":  ["#CC3311", "#009988", "#0077BB"],  # crisp, high-contrast (recommended)
    "okabe_ito":    ["#D55E00", "#009E73", "#0072B2"],  # academic standard
    "seaborn_deep": ["#C44E52", "#55A868", "#4C72B0"],  # soft / understated
}
PANEL_COLORS = PALETTES["tol_vibrant"]
FILL_TINT    = 0.60        # histogram fill = base colour blended this far toward white

# Axis symbols (change once, propagates everywhere)
XLABEL = r"$\hat{J}_{\omega,s}$"
YLABEL = r"$P(\hat{J}_{\omega,s})$"

# ============================================================
#  STYLE CONTROLS  ←  edit freely
# ============================================================
USE_LATEX = True            # True = real LaTeX (matches your current fig).
                            # False = mathtext fallback (no LaTeX install needed).

# -- figure geometry --------------------------------------------------------
FIG_WIDTH   = 7.0           # inches  (~17.8 cm, PRL/Nature double column)
FIG_HEIGHT  = 3.5          # inches
DPI         = 600           # PNG export dpi (journals want >=300; PDF is vector)

# -- font sizes (pt) --------------------------------------------------------
XLABEL_FS   = 21            # x-axis label
YLABEL_FS   = 21            # y-axis label
XTICK_FS    = 18             # x tick numbers
YTICK_FS    = 18             # y tick numbers
TITLE_FS    = 18           # panel titles (T = ...)
ANNOT_FS    = 15           # mu / sigma text
PANELTAG_FS = 14           # (a) (b) (c)

# -- line widths ------------------------------------------------------------
SPINE_LW       = 1.3        # box around each panel
GAUSSIAN_LW    = 1.7        # Gaussian overlay curve
HIST_EDGE_LW   = 0.3        # histogram bar edges

# -- tick geometry ----------------------------------------------------------
TICK_DIR        = "in"      # 'in' | 'out' | 'inout'  (PRL convention = 'in')
MAJOR_TICK_LEN  = 4.0
MINOR_TICK_LEN  = 3.0
MAJOR_TICK_LW   = 1
MINOR_TICK_LW   = 0.7
TICKS_TOP_RIGHT = True       # mirror ticks on top & right spines (classic look)
N_XTICKS        = 3          # target number of major x ticks per panel
SHOW_MINOR      = True       # minor ticks on/off

# -- histogram / curve ------------------------------------------------------
N_BINS         = 55
X_RANGE_STD    = 4.0         # x-window = mean +/- X_RANGE_STD * sigma
HIST_ALPHA     = 0.55
Y_HEADROOM     = 1.10        # blank space above the tallest peak
SHOW_MEAN_LINE = False       # thin dashed vertical line at mu
MEAN_LINE_LW   = 0.8

# -- annotation -------------------------------------------------------------
ANNOT_BOX = False            # draw a subtle white box behind the mu/sigma text
# ============================================================




def load_data():
    """Read J_star (column 2) of all students from fig2b_students_T<T>.csv."""
    J = {}
    for T in TEMPS:
        arr = np.loadtxt(f"fig2b_students_T{T}.csv", delimiter=",", comments="#", skiprows=3)
        J[T] = arr[:, 2] * float(T)          # J_hat = T * J_star
        print(f"T={T}: N={len(J[T])}")
    return J



def main():
    J_data = load_data()
    style = ["science"] if USE_LATEX else ["science", "no-latex"]

    with plt.style.context(style):
        fig, axes = plt.subplots(
            1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=True
        )

        global_peak = 0.0
        for i, (ax, T) in enumerate(zip(axes, TEMPS)):
            data = J_data[T]
            line_c = PANEL_COLORS[i]
            fill_c = tint(line_c, FILL_TINT)

            if len(data) == 0:
                ax.set_visible(False)
                continue

            mu = float(np.mean(data))
            sd = float(np.std(data))
            xmin, xmax = mu - X_RANGE_STD * sd, mu + X_RANGE_STD * sd

            # --- histogram (density) ---
            counts, _, _ = ax.hist(
                data, bins=N_BINS, range=(xmin, xmax), density=True,
                color=fill_c, alpha=HIST_ALPHA,
                edgecolor=line_c, linewidth=HIST_EDGE_LW, zorder=2,
            )

            # --- Gaussian overlay ---
            x = np.linspace(xmin, xmax, 600)
            g = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(x, g, color=line_c, lw=GAUSSIAN_LW, solid_capstyle="round",
                    zorder=3)
            global_peak = max(global_peak, counts.max(), g.max())

            if SHOW_MEAN_LINE:
                ax.axvline(mu, color=line_c, lw=MEAN_LINE_LW,
                           ls=(0, (4, 2)), alpha=0.85, zorder=1)

            # --- mu / sigma annotation (top-right corner) ---
            bbox = (dict(boxstyle="round,pad=0.3", fc="white",
                         ec="0.8", lw=0.5, alpha=0.85) if ANNOT_BOX else None)
            ax.text(
                0.955, 0.95,
                f"$\\mu = {mu:.4f}$\n$\\sigma = {sd:.4f}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=ANNOT_FS, color="0.15", bbox=bbox, zorder=4,
            )

            # --- panel tag (a)/(b)/(c) (top-left corner) ---
            if PANEL_TAGS:
                ax.text(0.05, 0.94, PANEL_TAGS[i], transform=ax.transAxes,
                        ha="left", va="top", fontsize=PANELTAG_FS,
                        fontweight="bold", zorder=4)

            # --- title & labels ---
            ax.set_title(TEMP_LABELS[T], fontsize=TITLE_FS, pad=5)
            ax.set_xlim(xmin, xmax)
            ax.set_xlabel(XLABEL, fontsize=XLABEL_FS)
            if ax is axes[0]:
                ax.set_ylabel(YLABEL, fontsize=YLABEL_FS)

            # --- ticks ---
            ax.xaxis.set_major_locator(MaxNLocator(nbins=N_XTICKS, prune="both"))
            if SHOW_MINOR:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="both", which="major", direction=TICK_DIR,
                           length=MAJOR_TICK_LEN, width=MAJOR_TICK_LW,
                           top=TICKS_TOP_RIGHT, right=TICKS_TOP_RIGHT)
            ax.tick_params(axis="both", which="minor", direction=TICK_DIR,
                           length=MINOR_TICK_LEN, width=MINOR_TICK_LW,
                           top=TICKS_TOP_RIGHT, right=TICKS_TOP_RIGHT)
            ax.tick_params(axis="x", labelsize=XTICK_FS)
            ax.tick_params(axis="y", labelsize=YTICK_FS)

            for spine in ax.spines.values():
                spine.set_linewidth(SPINE_LW)

        # shared y-limit with headroom (sharey -> set once)
        axes[0].set_ylim(0, global_peak * Y_HEADROOM)

        fig.tight_layout(pad=0.4, w_pad=1.0)
        fig.savefig("Fig2b.pdf", bbox_inches="tight")
        fig.savefig("Fig2b.png", dpi=DPI, bbox_inches="tight")
        plt.show()

    print("Saved: Fig2b.pdf and Fig2b.png")


if __name__ == "__main__":
    main()
