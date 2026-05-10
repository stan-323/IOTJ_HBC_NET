"""
IEEE single-column (3.5") version of the 4-panel HFSS coefficients figure.
Designed for tight single-column layout with rotated abbreviated labels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

BASE  = os.environ.get("HBC_HFSS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FINAL = os.path.join(BASE, "outputs", "final")
OUT   = os.path.join(BASE, "outputs", "images", "main")
os.makedirs(OUT, exist_ok=True)

# ── IEEE single-column rcParams ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     7,
    "axes.titlesize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    5.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":    0.9,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "axes.grid":          True,
    "grid.linewidth":     0.3,
    "grid.alpha":         0.35,
    "grid.color":         "#999999",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── colours ─────────────────────────────────────────────────────────────────
C_SR  = "#2166AC"   # surface rest
C_SL  = "#74ADD1"   # surface moderate loose
C_SW  = "#4DAC26"   # surface sweat
C_IR  = "#D73027"   # implant rest
C_IS  = "#F46D43"   # implant stress

BAR_COLORS = [C_SR, C_SL, C_SW, C_IR, C_IS, C_IR, C_IS]

# ── load data ───────────────────────────────────────────────────────────────
sched = pd.read_csv(os.path.join(FINAL, "calibrated_library_sched.csv"))
main  = sched[sched["sched_include_main"] == True].copy()

ORDER = [
    "surface_rest", "surface_moderate_loose", "surface_sweat",
    "implant10_rest", "implant10_stress",
    "implant30_rest", "implant30_stress",
]
main_ord = pd.Categorical(main["label"], categories=ORDER, ordered=True)
main = main.assign(_ord=main_ord).sort_values("_ord").reset_index(drop=True)

x = np.arange(len(ORDER))

# ── short rotated labels ────────────────────────────────────────────────────
XLABELS_SHORT = [
    "S-Rest",
    "S-Loose",
    "S-Sweat",
    "I10-Rest",
    "I10-Str.",
    "I30-Rest",
    "I30-Str.",
]

# ── figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(3.5, 4.2))
axes = axes.flatten()

BAR_W = 0.52

PANELS = [
    ("g_norm",        r"$g_{\rm norm}$",                    True,  "(a) Energy-coupling proxy"),
    ("r_norm",        r"$r_{\rm norm}$",                    True,  "(b) Data-service proxy"),
    ("chi_norm",      r"$\chi_{\rm norm}$",                 False, "(c) Local-field burden proxy"),
    ("p_rx_cap_norm", r"$p^{\rm norm}_{\rm rx,cap}$",       True,  "(d) Rx-power cap proxy"),
]

for ax, (col, ylabel, use_log, title) in zip(axes, PANELS):
    vals = main[col].values
    ax.bar(x, vals, width=BAR_W, color=BAR_COLORS,
           edgecolor="white", linewidth=0.35, zorder=3)

    if use_log:
        ax.set_yscale("log")
    ax.axhline(1, color="#444444", lw=0.6, ls="--", zorder=2, alpha=0.6)

    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, fontsize=6.5, loc="left", pad=2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS_SHORT, fontsize=5.5,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)

    # value annotations – handle tall bars and reference-line overlap
    ymin, ymax = ax.get_ylim()
    for xi, v in enumerate(vals):
        if v < 5e-4:
            txt = f"{v:.1e}"
        elif v < 0.01:
            txt = f"{v:.3f}"
        elif v < 10:
            txt = f"{v:.2f}"
        else:
            txt = f"{v:.1f}"

        if use_log:
            proposed_y = v * 1.65
            # if annotation would exceed visible area, put inside bar
            if proposed_y > ymax * 0.85:
                ax.text(xi, v * 0.55, txt, ha="center", va="top",
                        fontsize=4.5, color="white", fontweight="bold",
                        clip_on=True)
                continue
            # avoid overlap with reference line at 1.0
            if 0.72 < proposed_y < 1.40:
                proposed_y = v * 3.0
            yt, va = proposed_y, "bottom"
        else:
            yt, va = v + 0.025, "bottom"

        ax.text(xi, yt, txt, ha="center", va=va, fontsize=4.5,
                color="#1a1a1a", clip_on=True)

# ── shared legend at bottom ─────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_SR, label="S-Rest (baseline)"),
    mpatches.Patch(color=C_SL, label="S-Loose"),
    mpatches.Patch(color=C_SW, label="S-Sweat"),
    mpatches.Patch(color=C_IR, label="Imp. rest"),
    mpatches.Patch(color=C_IS, label="Imp. stress"),
]
fig.legend(handles=legend_items, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.02),
           frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
           fontsize=5.5, handlelength=1.0, handleheight=0.8,
           columnspacing=0.6, borderpad=0.3)

fig.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.2, w_pad=0.9)

for fmt in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_main_A_single_col.{fmt}"))
plt.close(fig)
print("Saved fig_main_A_single_col")
print("Output:", OUT)
