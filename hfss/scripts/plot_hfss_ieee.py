"""
IEEE Transactions-style HFSS calibration figures  (v2 – revised per review)

Main text figures (main/):
  fig_main_A – 4-panel scheduling library coefficients
  fig_main_B – Surface loose air-gap scan

Supplementary figures (suppl/):
  fig_suppl_C – Field proxy stability (5 mm vs 10 mm cube)
  fig_suppl_D – Multi-frequency path-loss ordering (line plot)
  fig_suppl_E – Path loss + EM proxies summary
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = os.environ.get("HBC_HFSS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
FINAL  = os.path.join(BASE, "outputs", "final")
REVIEW = os.path.join(BASE, "outputs", "hfss_review")
MAIN   = os.path.join(BASE, "outputs", "images", "main")
SUPPL  = os.path.join(BASE, "outputs", "images", "suppl")
os.makedirs(MAIN,  exist_ok=True)
os.makedirs(SUPPL, exist_ok=True)

# ── IEEE rcParams ───────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     8,
    "axes.titlesize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "lines.linewidth":    1.1,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.40,
    "grid.color":         "#888888",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── colour palette ──────────────────────────────────────────────────────────
C_SURF_BASE  = "#2166AC"
C_SURF_DEG   = "#74ADD1"
C_SURF_WET   = "#4DAC26"
C_IMP_REST   = "#D73027"
C_IMP_STRESS = "#F46D43"

# ── load data ───────────────────────────────────────────────────────────────
sched = pd.read_csv(os.path.join(FINAL,  "calibrated_library_sched.csv"))
loose = pd.read_csv(os.path.join(REVIEW, "surface_loose_scan.csv"))
stab  = pd.read_csv(os.path.join(REVIEW, "field_proxy_stability.csv"))
freq  = pd.read_csv(os.path.join(REVIEW, "aux_frequency_ordering.csv"))

# main-library rows only
main = sched[sched["sched_include_main"] == True].copy()

# ── ordered labels ──────────────────────────────────────────────────────────
ORDER = [
    "surface_rest",
    "surface_moderate_loose",
    "surface_sweat",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
]
# concise x-axis labels (two lines, ≤12 chars each)
XLABELS = [
    "Surf.\nRest",
    "Surf.\nMod. Loose",
    "Surf.\nSweat",
    "Imp. 10\nRest",
    "Imp. 10\nStress",
    "Imp. 30\nRest",
    "Imp. 30\nStress",
]
BAR_COLORS = [
    C_SURF_BASE, C_SURF_DEG, C_SURF_WET,
    C_IMP_REST,  C_IMP_STRESS,
    C_IMP_REST,  C_IMP_STRESS,
]

main_ord = pd.Categorical(main["label"], categories=ORDER, ordered=True)
main = main.assign(_ord=main_ord).sort_values("_ord").reset_index(drop=True)
x    = np.arange(len(ORDER))
BAR_W = 0.52


# ═══════════════════════════════════════════════════════════════════════════
# fig_main_A  –  4-panel scheduling library coefficients
# ═══════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
axes = axes.flatten()

PANELS = [
    # (column, y-label, log-scale, panel title)
    ("g_norm",
     r"$g_{\rm norm}$",
     True,
     r"(a) Energy-coupling proxy"),
    ("r_norm",
     r"$r_{\rm norm}$",
     True,
     r"(b) Data-service proxy"),
    ("chi_norm",
     r"$\chi_{\rm norm}$",
     False,
     r"(c) Local-field burden proxy"),
    ("p_rx_cap_norm",
     r"$p^{\rm norm}_{\rm rx,cap}$",
     True,
     r"(d) Rx-power cap proxy"),
]

for ax, (col, ylabel, use_log, title) in zip(axes, PANELS):
    vals = main[col].values
    ax.bar(x, vals, width=BAR_W, color=BAR_COLORS,
           edgecolor="white", linewidth=0.4, zorder=3)

    if use_log:
        ax.set_yscale("log")
    ax.axhline(1, color="#444444", lw=0.8, ls="--", zorder=2, alpha=0.65)

    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, pad=3, fontsize=8, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=6.5, linespacing=1.15)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(ORDER) - 0.45)

    # value annotations above each bar
    for xi, v in enumerate(vals):
        if use_log:
            yt, va = v * 1.55, "bottom"
        else:
            yt, va = v + 0.018, "bottom"
        if v < 5e-4:
            txt = f"{v:.1e}"
        elif v < 0.01:
            txt = f"{v:.4f}"
        elif v < 10:
            txt = f"{v:.3f}"
        else:
            txt = f"{v:.1f}"
        ax.text(xi, yt, txt, ha="center", va=va, fontsize=5.2,
                color="#1a1a1a", clip_on=True)

    # sweat annotation on g_norm panel
    if col == "g_norm":
        ax.text(2, vals[2] * 0.35,
                "High-cond.\nwet contact",
                ha="center", va="top", fontsize=5.5,
                color="#2a6a10", style="italic")

# shared bottom legend
legend_items = [
    mpatches.Patch(color=C_SURF_BASE,  label="Surface / rest (baseline)"),
    mpatches.Patch(color=C_SURF_DEG,   label="Surface / mod. loose"),
    mpatches.Patch(color=C_SURF_WET,   label="Surface / sweat"),
    mpatches.Patch(color=C_IMP_REST,   label="Implant / rest"),
    mpatches.Patch(color=C_IMP_STRESS, label="Implant / stress"),
]
fig1.legend(handles=legend_items, ncol=5, loc="lower center",
            bbox_to_anchor=(0.5, -0.04), frameon=True,
            framealpha=0.92, edgecolor="#CCCCCC", fontsize=6.8,
            handlelength=1.2, handleheight=0.9, columnspacing=0.8)

fig1.tight_layout(rect=[0, 0.07, 1, 1], h_pad=1.6, w_pad=1.2)

for fmt in ("pdf", "png"):
    fig1.savefig(os.path.join(MAIN, f"fig_main_A_library_coefficients.{fmt}"))
plt.close(fig1)
print("Saved fig_main_A")


# ═══════════════════════════════════════════════════════════════════════════
# fig_main_B  –  Surface loose air-gap scan
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(3.5, 2.75))

gap = loose["loose_gap_mm"].astype(float).values
g_n = loose["g_norm"].values
r_n = loose["r_norm"].values

sel_mask = loose["selection_status"] == "selected_in_target_band"

# target band shading
ax2.axhspan(0.10, 0.80, color="#C6DBEF", alpha=0.40, zorder=1)
ax2.text(0.55, 0.38, "Target band\n(0.10–0.80)",
         ha="center", va="center", fontsize=6, color="#1a5276",
         transform=ax2.get_xaxis_transform(),
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# contact-failure region shading
ax2.axvspan(0.04, gap.max() * 2.0, color="#FADBD8", alpha=0.35, zorder=1)
ax2.text(0.85, 0.07, "Contact-\nfailure\nregion",
         ha="center", va="bottom", fontsize=5.5, color="#922b21",
         transform=ax2.get_xaxis_transform())

ax2.semilogy(gap, g_n, "o-",  color=C_IMP_REST,  ms=4.5, lw=1.1,
             label=r"$g_{\rm norm}$  (13.56 MHz)", zorder=4)
ax2.semilogy(gap, r_n, "s--", color=C_SURF_BASE,  ms=4.5, lw=1.1,
             label=r"$r_{\rm norm}$  (40 MHz)",    zorder=4)

# highlight selected point with open circle
for _, row in loose[sel_mask].iterrows():
    gv, rv = float(row["g_norm"]), float(row["r_norm"])
    gp = float(row["loose_gap_mm"])
    ax2.semilogy(gp, gv, "o", ms=9, mfc="none", mew=1.5,
                 color=C_IMP_REST, zorder=5)
    ax2.semilogy(gp, rv, "s", ms=9, mfc="none", mew=1.5,
                 color=C_SURF_BASE, zorder=5)
    # annotation in upper-left area, clear of all curves
    ax2.annotate(
        r"Selected ($\delta$=0.005 mm)",
        xy=(gp, gv),
        xytext=(0.0018, 1.4),
        fontsize=6.2, color="#2c2c2c",
        arrowprops=dict(arrowstyle="-|>", lw=0.75,
                        color="#555555",
                        connectionstyle="arc3,rad=-0.25"),
    )

ax2.set_xlabel(r"Air gap $\delta$ (mm)", labelpad=3)
ax2.set_ylabel(r"Normalized coefficient", labelpad=3)
ax2.set_title("Surface loose-contact air-gap scan", pad=4, fontsize=8, loc="left")
ax2.set_xscale("log")
ax2.set_xlim(gap.min() * 0.65, gap.max() * 2.2)
ax2.set_ylim(5e-5, 4.0)

ax2.legend(fontsize=6.5, loc="lower left",
           framealpha=0.92, edgecolor="#CCCCCC",
           handlelength=1.5, borderpad=0.5)

fig2.tight_layout()
for fmt in ("pdf", "png"):
    fig2.savefig(os.path.join(MAIN, f"fig_main_B_surface_loose_scan.{fmt}"))
plt.close(fig2)
print("Saved fig_main_B")


# ═══════════════════════════════════════════════════════════════════════════
# fig_suppl_C  –  Field proxy stability (5 mm vs 10 mm observation cube)
# ═══════════════════════════════════════════════════════════════════════════
stab5  = stab[stab["cube_mm"] == 5.0].copy()
stab10 = stab[stab["cube_mm"] == 10.0].copy()

STAB_ORDER  = ["surface_rest", "surface_moderate_loose", "implant30_stress"]
STAB_LABELS = ["Surface\nRest", "Surf.\nMod. Loose", "Imp. 30\nStress"]
STAB_COLORS = [C_SURF_BASE, C_SURF_DEG, C_IMP_STRESS]

fig3, ax3 = plt.subplots(figsize=(3.3, 2.65))

xs  = np.arange(len(STAB_ORDER))
BW3 = 0.27


def get_q(df, lbl):
    row = df[df["label"] == lbl]
    return np.nan if row.empty else row["Q_p95_norm_to_surface_rest_5mm"].values[0]


vals5  = [get_q(stab5,  l) for l in STAB_ORDER]
vals10 = [get_q(stab10, l) for l in STAB_ORDER]

ax3.bar(xs - BW3/2, vals5,  width=BW3, color=STAB_COLORS,
        edgecolor="white", lw=0.4, zorder=3)
ax3.bar(xs + BW3/2, vals10, width=BW3, color=STAB_COLORS,
        edgecolor="#444444", lw=0.5, hatch="///", zorder=3, alpha=0.72)

ax3.axhline(1, color="#444444", lw=0.8, ls="--", zorder=2, alpha=0.65)

for xi, (v5, v10) in enumerate(zip(vals5, vals10)):
    ax3.text(xi - BW3/2, v5  + 0.004, f"{v5:.3f}",
             ha="center", va="bottom", fontsize=5.5)
    ax3.text(xi + BW3/2, v10 + 0.004, f"{v10:.3f}",
             ha="center", va="bottom", fontsize=5.5)

ax3.set_xticks(xs)
ax3.set_xticklabels(STAB_LABELS, fontsize=7, linespacing=1.15)
ax3.tick_params(axis="x", length=0)
ax3.set_ylabel(r"$Q_{\rm p95,norm}$", labelpad=3)
ax3.set_title("Field proxy stability vs. observation-cube size",
              pad=3, fontsize=8, loc="left")
ax3.set_ylim(0.965, 1.165)

legend_c = [
    mpatches.Patch(color="#999999",               label="5-mm cube"),
    mpatches.Patch(color="#999999", hatch="///",  label="10-mm cube"),
]
ax3.legend(handles=legend_c, fontsize=6.5, loc="upper center",
           ncol=2, framealpha=0.92, edgecolor="#CCCCCC",
           handlelength=1.2, borderpad=0.4)

fig3.tight_layout()
for fmt in ("pdf", "png"):
    fig3.savefig(os.path.join(SUPPL, f"fig_suppl_C_field_proxy_stability.{fmt}"))
plt.close(fig3)
print("Saved fig_suppl_C")


# ═══════════════════════════════════════════════════════════════════════════
# fig_suppl_D  –  Multi-frequency path-loss ordering  (line + marker plot)
# ═══════════════════════════════════════════════════════════════════════════
FREQ_ORDER  = ["surface_rest", "implant10_rest", "implant30_rest", "implant30_stress"]
FREQ_LABELS = ["Surf.\nRest", "Imp. 10\nRest", "Imp. 30\nRest", "Imp. 30\nStress"]

PL_40 = {row["label"]: row["path_loss_data_db"]   for _, row in sched.iterrows()}
PL_13 = {row["label"]: row["path_loss_energy_db"] for _, row in sched.iterrows()}
PL_20 = {row["label"]: row["path_loss_dB"]         for _, row in freq.iterrows()}

pl13 = np.array([PL_13.get(l, np.nan) for l in FREQ_ORDER])
pl20 = np.array([PL_20.get(l, np.nan) for l in FREQ_ORDER])
pl40 = np.array([PL_40.get(l, np.nan) for l in FREQ_ORDER])

fig4, ax4 = plt.subplots(figsize=(3.5, 2.75))

xf = np.arange(len(FREQ_ORDER))

ax4.plot(xf, pl13, "o-",  color="#2166AC", ms=5.5, lw=1.2,
         label=r"$f_E$ = 13.56 MHz")
ax4.plot(xf, pl20, "s--", color="#4DAC26", ms=5.5, lw=1.2,
         label=r"$f_{\rm aux}$ = 20 MHz")
ax4.plot(xf, pl40, "^-.", color="#D73027", ms=5.5, lw=1.2,
         label=r"$f_D$ = 40 MHz")

# rank labels
for xi in xf:
    y_max = max(pl13[xi], pl20[xi], pl40[xi])
    ax4.text(xi, y_max + 0.4, f"Rank {xi+1}",
             ha="center", va="bottom", fontsize=5.5, color="#333333")

ax4.set_xticks(xf)
ax4.set_xticklabels(FREQ_LABELS, fontsize=7, linespacing=1.15)
ax4.tick_params(axis="x", length=0)
ax4.set_ylabel(r"Path loss (dB)", labelpad=3)
ax4.set_title("Multi-frequency path-loss ordering check",
              pad=3, fontsize=8, loc="left")
ax4.set_xlim(-0.4, len(FREQ_ORDER) - 0.6)
y_lo = min(pl13.min(), pl20.min(), pl40.min()) - 2
y_hi = max(pl13.max(), pl20.max(), pl40.max()) + 3
ax4.set_ylim(y_lo, y_hi)

ax4.legend(fontsize=6.5, loc="lower right",
           framealpha=0.92, edgecolor="#CCCCCC", handlelength=1.8)

fig4.tight_layout()
for fmt in ("pdf", "png"):
    fig4.savefig(os.path.join(SUPPL, f"fig_suppl_D_multifreq_ordering.{fmt}"))
plt.close(fig4)
print("Saved fig_suppl_D")


# ═══════════════════════════════════════════════════════════════════════════
# fig_suppl_E  –  Path loss + EM proxies summary  (2-panel)
# Path loss: positive bars going UPWARD (larger bar = more attenuation)
# ═══════════════════════════════════════════════════════════════════════════
fig5, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.65))

pl_e = main["path_loss_energy_db"].values   # positive dB
pl_d = main["path_loss_data_db"].values

BW5 = 0.30
# draw positive bars upward — larger bar means more path loss
axA.bar(x - BW5/2, pl_e, width=BW5, color=BAR_COLORS,
        edgecolor="white", lw=0.4, alpha=0.95,
        label=r"$f_E$ = 13.56 MHz", zorder=3)
axA.bar(x + BW5/2, pl_d, width=BW5, color=BAR_COLORS,
        edgecolor="#444444", lw=0.5, hatch="///", alpha=0.70,
        label=r"$f_D$ = 40 MHz", zorder=3)

# y-axis: 0 at bottom, higher = more loss
axA.set_ylim(0, max(pl_e.max(), pl_d.max()) * 1.15)
axA.set_xticks(x)
axA.set_xticklabels(XLABELS, fontsize=6.2, linespacing=1.15)
axA.tick_params(axis="x", length=0)
axA.set_ylabel(r"Path loss, $-|S_{21}|$ (dB)", labelpad=3)
axA.set_title(r"(a) Path loss at $f_E$ and $f_D$", pad=3, fontsize=8, loc="left")
axA.legend(fontsize=6.5, loc="upper left",
           framealpha=0.92, edgecolor="#CCCCCC", handlelength=1.5)

# right panel: chi_norm bars + p_rx_cap_norm line (log)
ax_chi = axB
ax_cap = ax_chi.twinx()
ax_cap.spines["right"].set_visible(True)

chi_vals = main["chi_norm"].values
cap_vals = main["p_rx_cap_norm"].values

ax_chi.bar(x, chi_vals, width=BAR_W, color=BAR_COLORS,
           edgecolor="white", lw=0.4, alpha=0.82, zorder=3)
ax_cap.semilogy(x, cap_vals, "D--", color="#333333",
                ms=4.5, lw=1.1, zorder=5,
                label=r"$p^{\rm norm}_{\rm rx,cap}$")

ax_chi.axhline(1, color="#444444", lw=0.8, ls=":", zorder=2, alpha=0.6)
ax_chi.set_xticks(x)
ax_chi.set_xticklabels(XLABELS, fontsize=6.2, linespacing=1.15)
ax_chi.tick_params(axis="x", length=0)
ax_chi.set_ylabel(r"$\chi_{\rm norm}$",            labelpad=3)
ax_cap.set_ylabel(r"$p^{\rm norm}_{\rm rx,cap}$",  labelpad=3)
axB.set_title(r"(b) Local-field burden and Rx-power cap",
              pad=3, fontsize=8, loc="left")

h1, l1 = ax_chi.get_legend_handles_labels()
h2, l2 = ax_cap.get_legend_handles_labels()
ax_chi.legend(h1 + h2, [r"$\chi_{\rm norm}$ (bars)"] + l2,
              fontsize=6.5, loc="upper left",
              framealpha=0.92, edgecolor="#CCCCCC")

fig5.tight_layout(w_pad=1.5)
for fmt in ("pdf", "png"):
    fig5.savefig(os.path.join(SUPPL, f"fig_suppl_E_path_loss_em_proxies.{fmt}"))
plt.close(fig5)
print("Saved fig_suppl_E")

print(f"\nMain figures   → {MAIN}")
print(f"Suppl. figures → {SUPPL}")
