"""
IEEE double-column 1×4 HFSS coefficients figure.
Style: full box border, inward ticks (top/right also tick), compressed height.
Target: 7.0" × 2.15", suitable for IEEE Trans double-column placement.
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

# ── rcParams ─────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     7,
    "axes.titlesize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":    0.9,
    # box + inward ticks
    "axes.linewidth":     0.55,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.width":  0.55,
    "ytick.major.width":  0.55,
    "xtick.major.size":   2.8,
    "ytick.major.size":   2.8,
    "xtick.minor.size":   1.5,
    "ytick.minor.size":   1.5,
    # grid off by default; we'll add manually where needed
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
main = (main.assign(_ord=pd.Categorical(main["label"], categories=ORDER, ordered=True))
            .sort_values("_ord").reset_index(drop=True))
x = np.arange(len(ORDER))

# ultra-short x labels (6-char max, so 45° rotation fits in tight subplot)
XLABELS = ["S-R", "S-L", "S-Sw", "I10-R", "I10-S", "I30-R", "I30-S"]
BAR_W = 0.52

# ── panel definitions ─────────────────────────────────────────────────────────
PANELS = [
    ("g_norm",        r"$g_{\rm norm}$",                True,
     r"(a) $g_{\rm norm}$"),
    ("r_norm",        r"$r_{\rm norm}$",                True,
     r"(b) $r_{\rm norm}$"),
    ("chi_norm",      r"$\chi_{\rm norm}$",             False,
     r"(c) $\chi_{\rm norm}$"),
    ("p_rx_cap_norm", r"$p^{\rm norm}_{\rm rx,cap}$",   True,
     r"(d) $p^{\rm norm}_{\rm rx,cap}$"),
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15),
                         gridspec_kw={"wspace": 0.38})

for ax, (col, ylabel, use_log, title) in zip(axes, PANELS):
    vals = main[col].values

    bars = ax.bar(x, vals, width=BAR_W, color=BAR_COLORS,
                  edgecolor="white", linewidth=0.3, zorder=3)

    # subtle horizontal grid lines only
    ax.yaxis.grid(True, linewidth=0.3, color="#AAAAAA", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if use_log:
        ax.set_yscale("log")
    ax.axhline(1, color="#333333", lw=0.65, ls="--", zorder=2, alpha=0.55)

    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=5.8,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", which="both", length=0, pad=1)
    ax.tick_params(axis="y", which="both", pad=1.5)
    ax.set_xlim(-0.55, len(ORDER) - 0.45)

    # ── value annotations ───────────────────────────────────────────────────
    # Use blended transform: x in data coords, y in axes (0–1) coords.
    # This decouples annotation y-position from data range, preventing
    # any clipping issues with tall bars.
    from matplotlib.transforms import blended_transform_factory
    trans_top = blended_transform_factory(ax.transData, ax.transAxes)

    ymin_ax, ymax_ax = ax.get_ylim()

    for xi, v in enumerate(vals):
        # format string
        if v < 5e-4:
            txt = f"{v:.1e}"
        elif v < 0.01:
            txt = f"{v:.3f}"
        elif v < 10:
            txt = f"{v:.2f}"
        else:
            txt = f"{v:.1f}"

        if use_log:
            import math
            # fractional height of bar in log-axis (0=bottom, 1=top of axes)
            log_bot = math.log10(max(ymin_ax, 1e-10))
            log_top = math.log10(ymax_ax)
            bar_frac = (math.log10(v) - log_bot) / (log_top - log_bot)

            if bar_frac > 0.80:
                # tall bar: place value at fixed axes-fraction position
                # (just below the top spine), coloured background box
                ax.text(xi, 0.96, txt,
                        ha="center", va="top",
                        transform=trans_top,
                        fontsize=5.5, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12",
                                  fc=BAR_COLORS[xi], ec="none"))
            else:
                # normal bar: annotate above bar in data coords
                proposed = v * 1.6
                if 0.68 < proposed < 1.45:      # dodge reference line
                    proposed = v * 3.2
                ax.text(xi, proposed, txt,
                        ha="center", va="bottom",
                        fontsize=4.2, color="#111111", clip_on=True)
        else:
            span = ymax_ax - ymin_ax
            ax.text(xi, v + span * 0.022, txt,
                    ha="center", va="bottom",
                    fontsize=4.2, color="#111111", clip_on=True)

# ── shared legend below all panels ───────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_SR, label="S-R: Surface rest (baseline)"),
    mpatches.Patch(color=C_SL, label="S-L: Surface mod. loose"),
    mpatches.Patch(color=C_SW, label="S-Sw: Surface sweat"),
    mpatches.Patch(color=C_IR, label="I10/I30-R: Implant rest"),
    mpatches.Patch(color=C_IS, label="I10/I30-S: Implant stress"),
]
fig.legend(handles=legend_items, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, -0.13),
           frameon=True, framealpha=0.95, edgecolor="#BBBBBB",
           fontsize=5.5, handlelength=0.9, handleheight=0.8,
           columnspacing=0.5, borderpad=0.4, handletextpad=0.4)

fig.tight_layout(rect=[0, 0.0, 1, 1])

for fmt in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_main_A_1x4.{fmt}"))
plt.close(fig)
print("Saved fig_main_A_1x4  →", OUT)
