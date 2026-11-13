#!/usr/bin/env python3
"""
1x2 figure at fixed temperature T = 2.20, disorder on x-axis (Fig. 6 of the paper):
  (a) Jbar vs delta  -- identity, isotropic and linear kernels coincide (Eq. E6);
                        inset shows deviation from kernel mean (x10^-5).
  (b) RC   vs delta  -- the three kernels separate (kernel-dependent).
Reads fig6_Jbar_per_kernel.dat and fig6_RC_per_kernel.dat.
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------- PRE style ----------
plt.rcParams.update({
    "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
    "axes.labelsize": 26, "xtick.labelsize": 19, "ytick.labelsize": 19,
    "legend.fontsize": 16, "axes.linewidth": 1.2, "lines.linewidth": 2.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 5.5, "ytick.major.size": 5.5,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

kernels = ["identity", "iso", "linear"]      # kernels shown in the paper
klabel = {"identity":"Identity","iso":"Isotropic","rbf":"RBF","linear":"Linear"}
kmark  = {"identity":"o","iso":"s","rbf":"^","linear":"D"}
kcol   = {"identity":"#1B6FE3","iso":"#E03020","rbf":"#2E8B57","linear":"#9B59B6"}
deltas = ["d0","d2","d4","d6","d8","d10"]
dval = [0.0,0.2,0.4,0.6,0.8,1.0]

# ---- evaluation mode ----
# "fixed" : all delta at one temperature T_FIX
# "tc"    : each delta at its own critical temperature Tc[delta]  (interpolated)
MODE  = "fixed"                # paper: T = 2.20 for every delta
T_FIX = 2.20
Tc    = {"d0":2.28,"d2":2.28,"d4":2.24,"d6":2.20,"d8":2.16,"d10":2.04}

def load_at_fixed(fn, Tt):
    with open(fn) as f: hdr = f.readline().split()
    arr = np.loadtxt(fn, skiprows=1); col = {n:i for i,n in enumerate(hdr)}
    T = arr[:, col["temperature"]]; it = np.argmin(np.abs(T - Tt))
    return {k: np.array([arr[it, col[f"{k}_{dk}"]] for dk in deltas]) for k in kernels}, T[it]

def load_at_tc(fn):
    """Each delta evaluated at its own Tc via linear interpolation in T."""
    with open(fn) as f: hdr = f.readline().split()
    arr = np.loadtxt(fn, skiprows=1); col = {n:i for i,n in enumerate(hdr)}
    T = arr[:, col["temperature"]]; o = np.argsort(T); T = T[o]
    out = {k: [] for k in kernels}
    for dk in deltas:
        tc = Tc[dk]
        for k in kernels:
            y = arr[o, col[f"{k}_{dk}"]]
            out[k].append(np.interp(tc, T, y))
    return {k: np.array(v) for k, v in out.items()}, None

if MODE == "tc":
    Jd, _   = load_at_tc("fig6_Jbar_per_kernel.dat")
    RCd, _  = load_at_tc("fig6_RC_per_kernel.dat")
    Tval = None
else:
    Jd, Tval = load_at_fixed("fig6_Jbar_per_kernel.dat", T_FIX)
    RCd, _   = load_at_fixed("fig6_RC_per_kernel.dat", T_FIX)
Jmean = np.mean([Jd[k] for k in kernels], axis=0)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.5, 5.9))
fig.subplots_adjust(left=0.08, right=0.985, wspace=0.28, bottom=0.15, top=0.93)

# ===== (a) Jhat vs delta : kernels coincide =====
axa.plot(dval, Jmean, "-", color="0.6", lw=1.4, zorder=1)
for k in kernels:
    axa.plot(dval, Jd[k], ls="none", marker=kmark[k], ms=11, mfc="none",
             mec=kcol[k], mew=1.8, zorder=3, label=klabel[k])
axa.set_xlabel(r"$\delta$")
axa.set_ylabel(r"$\bar{J}$")
axa.set_xlim(-0.05, 1.05)
axa.xaxis.set_major_locator(MultipleLocator(0.2))
axa.xaxis.set_minor_locator(MultipleLocator(0.1))
axa.tick_params(top=True, right=True, which="both")
axa.legend(loc="lower left", frameon=False, handlelength=1.0,
           labelspacing=0.35, title=r"Kernel", title_fontsize=14)
axa.text(-0.16, 1.02, "(a)", transform=axa.transAxes, fontsize=24, ha="left", va="bottom")

# inset: deviation from kernel mean
axin = axa.inset_axes([0.56, 0.62, 0.40, 0.33])
S = 1e-5
for k in kernels:
    axin.plot(dval, (Jd[k] - Jmean) / S, ls="none", marker=kmark[k], ms=6,
              mfc="none", mec=kcol[k], mew=1.4)
axin.axhline(0, color="0.5", lw=0.8)
axin.set_xlabel(r"$\delta$", fontsize=12, labelpad=1)
axin.set_ylabel(r"$\Delta\bar{J}\ (\times10^{-5})$", fontsize=11, labelpad=2)
axin.tick_params(labelsize=10, top=True, right=False, which="both")
axin.xaxis.set_major_locator(MultipleLocator(0.5))
axin.yaxis.set_major_locator(MultipleLocator(1))
axin.set_xlim(-0.05, 1.05); axin.set_ylim(-2.0, 2.0)

# ===== (b) RC vs delta : kernels separate =====
for k in kernels:
    axb.plot(dval, RCd[k], "-", color=kcol[k], lw=1.5, alpha=0.5, zorder=2)
    axb.plot(dval, RCd[k], ls="none", marker=kmark[k], ms=11, mfc="none",
             mec=kcol[k], mew=1.8, zorder=3, label=klabel[k])
axb.set_xlabel(r"$\delta$")
axb.set_ylabel(r"RC")
axb.set_xlim(-0.05, 1.05)
axb.set_ylim(-0.0005, None)
axb.xaxis.set_major_locator(MultipleLocator(0.2))
axb.xaxis.set_minor_locator(MultipleLocator(0.1))
axb.tick_params(top=True, right=True, which="both")
axb.text(-0.16, 1.02, "(b)", transform=axb.transAxes, fontsize=24, ha="left", va="bottom")

fig.savefig("Fig6.pdf")
fig.savefig("Fig6.svg")
fig.savefig("Fig6.png", dpi=400)
print("ok")
