"""
IEEE single-column 4×1 HFSS coefficients figure.
  - Width: 3.5"  (single column)
  - Height: ~5.4" (4 stacked panels, compressed)
  - Style: full box, inward ticks, x-labels only on bottom panel
"""

import os, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import warnings
warnings.filterwarnings("ignore")

BASE  = os.environ.get("HBC_HFSS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FINAL = os.path.join(BASE, "outputs", "final")
OUT   = os.path.join(BASE, "outputs", "images", "main")
os.makedirs(OUT, exist_ok=True)

# ── rcParams ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     7.5,
    "axes.titlesize":     7.5,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "legend.fontsize":    6.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":    0.9,
    "axes.linewidth":     0.6,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "axes.grid":          False,
})

# ── colours ──────────────────────────────────────────────────────────────────
C_SR  = "#2166AC"
C_SL  = "#74ADD1"
C_SW  = "#4DAC26"
C_IR  = "#D73027"
C_IS  = "#F46D43"
BAR_COLORS = [C_SR, C_SL, C_SW, C_IR, C_IS, C_IR, C_IS]

# ── load & sort ───────────────────────────────────────────────────────────────
sched = pd.read_csv(os.path.join(FINAL, "calibrated_library_sched.csv"))
main  = sched[sched["sched_include_main"] == True].copy()
ORDER = ["surface_rest", "surface_moderate_loose", "surface_sweat",
         "implant10_rest", "implant10_stress",
         "implant30_rest", "implant30_stress"]
main = (main
        .assign(_ord=pd.Categorical(main["label"], categories=ORDER, ordered=True))
        .sort_values("_ord").reset_index(drop=True))
x = np.arange(len(ORDER))

# x-axis tick labels (bottom panel only)
XLABELS = ["S-Rest", "S-Loose", "S-Sweat",
           "I10-Rest", "I10-Str.", "I30-Rest", "I30-Str."]
BAR_W = 0.55

# ── panel definitions (top → bottom) ─────────────────────────────────────────
PANELS = [
    ("g_norm",        r"$g_{\rm norm}$",              True,  "(a)"),
    ("r_norm",        r"$r_{\rm norm}$",              True,  "(b)"),
    ("chi_norm",      r"$\chi_{\rm norm}$",           False, "(c)"),
    ("p_rx_cap_norm", r"$p^{\rm norm}_{\rm rx,cap}$", True,  "(d)"),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.4),
                         gridspec_kw={"hspace": 0.18})

for idx, (ax, (col, ylabel, use_log, tag)) in enumerate(zip(axes, PANELS)):
    vals = main[col].values
    is_last = (idx == len(PANELS) - 1)

    # bars
    ax.bar(x, vals, width=BAR_W, color=BAR_COLORS,
           edgecolor="white", linewidth=0.35, zorder=3)

    # subtle horizontal grid
    ax.yaxis.grid(True, linewidth=0.3, color="#BBBBBB", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    if use_log:
        ax.set_yscale("log")
    ax.axhline(1, color="#333333", lw=0.65, ls="--", zorder=2, alpha=0.5)

    # y-axis label with panel tag
    ax.set_ylabel(f"{tag}  {ylabel}", labelpad=3)

    ax.set_xlim(-0.55, len(ORDER) - 0.45)
    ax.tick_params(axis="both", which="both", pad=2)

    # x-axis: labels only on bottom panel
    ax.set_xticks(x)
    if is_last:
        ax.set_xticklabels(XLABELS, fontsize=6.5,
                           rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", which="major", length=3)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="major", length=3)

    # ── value annotations ─────────────────────────────────────────────────
    trans_top = blended_transform_factory(ax.transData, ax.transAxes)
    ymin_ax, ymax_ax = ax.get_ylim()

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
            log_bot  = math.log10(max(ymin_ax, 1e-12))
            log_top  = math.log10(max(ymax_ax, 1e-11))
            bar_frac = (math.log10(max(v, 1e-12)) - log_bot) / (log_top - log_bot)

            if bar_frac > 0.80:
                # tall bar: badge pinned near top of axes
                ax.text(xi, 0.96, txt,
                        ha="center", va="top",
                        transform=trans_top,
                        fontsize=5.8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc=BAR_COLORS[xi], ec="none"))
            else:
                proposed = v * 1.55
                if 0.70 < proposed < 1.45:   # dodge y=1 reference line
                    proposed = v * 3.0
                ax.text(xi, proposed, txt,
                        ha="center", va="bottom",
                        fontsize=5.0, color="#111111", clip_on=True)
        else:
            span = ymax_ax - ymin_ax
            ax.text(xi, v + span * 0.025, txt,
                    ha="center", va="bottom",
                    fontsize=5.0, color="#111111", clip_on=True)

# ── shared legend below bottom panel ─────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_SR, label="S-Rest (baseline)"),
    mpatches.Patch(color=C_SL, label="S-Loose"),
    mpatches.Patch(color=C_SW, label="S-Sweat"),
    mpatches.Patch(color=C_IR, label="Implant rest"),
    mpatches.Patch(color=C_IS, label="Implant stress"),
]
fig.legend(handles=legend_items, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, -0.055),
           frameon=True, framealpha=0.95, edgecolor="#BBBBBB",
           fontsize=6.2, handlelength=1.0, handleheight=0.85,
           columnspacing=0.7, borderpad=0.4, handletextpad=0.4)

fig.tight_layout(rect=[0, 0.07, 1, 1])

for fmt in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_main_A_4x1.{fmt}"))
plt.close(fig)
print("Saved fig_main_A_4x1  →", OUT)
