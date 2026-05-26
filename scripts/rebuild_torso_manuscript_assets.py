from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator


WORKSPACE = Path(__file__).resolve().parents[1]
MAIN_EXPERIMENT = WORKSPACE / "main_experiment"
if str(MAIN_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(MAIN_EXPERIMENT))

from Experiment_results.config import ABLATION_METHODS, MAIN_METHODS, STRESS_METHODS, SimConfig
from Experiment_results.gates import evaluate_candidate_gates
from Experiment_results.plotting import build_fig4_mechanism_frames
from Experiment_results.reports import write_pass_fail_csv


TORSO_OUTPUTS = WORKSPACE / "hfss" / "torso_main" / "outputs"
HELDOUT_DIR = MAIN_EXPERIMENT / "output_0515_chest_phase1_heldout" / "output" / "candidates" / "heldout_final"
FIG_DIR = WORKSPACE / "manuscript" / "figures"
REV_DIR = WORKSPACE / "manuscript" / "revision_generated"
STYLE_FILE = MAIN_EXPERIMENT / "ieee_trans.mplstyle"

BODY_TEXT_SIZE = 8
PANEL_TITLE_SIZE = 9
HFSS_TOP_HEADROOM = 1.25
REGIME_LABEL_Y = 0.90
REGIME_LABEL_COLOR = "#111111"
FIG5_ALLOCATION_YMAX = 0.24
FIG5_LEGEND_LINE_X0 = 340
FIG5_LEGEND_LINE_X1 = 380
FIG5_LEGEND_TEXT_X = 395
FIG5_LEGEND_CRITICAL_Y = 0.118
FIG5_LEGEND_SURFACE_Y = 0.088
FIG6_SHORTAGE_YMAX = 1.22
FIG6_STRESS_WINDOW_LABEL_Y = 1.14

COLORS = {
    "Proposed": "#1B4F8A",
    "ADT-MAC": "#B03A2E",
    "Lyap.-DPP": "#1E8449",
    "w/o EM-bud.": "#6FA3D0",
    "w/o crit.-urg.": "#9DB8CE",
    "w/o implant-aware": "#7F7F7F",
    "Surface nodes": "#E67E22",
    "Oracle": "#333333",
}

LINESTYLES = {
    "Proposed": "-",
    "ADT-MAC": "--",
    "Lyap.-DPP": "-.",
    "w/o EM-bud.": ":",
    "w/o crit.-urg.": (0, (5, 2)),
    "w/o implant-aware": (0, (3, 1, 1, 1)),
    "Oracle": "-",
}

METHOD_TICK_LABELS = {
    "Proposed": "Proposed",
    "ADT-MAC": "SP\nbaseline",
    "Lyap.-DPP": "Lyap.\n-DPP",
    "w/o EM-bud.": "w/o EM\nbudget",
    "w/o crit.-urg.": "w/o crit.\nurg.",
    "w/o implant-aware": "w/o implant\naware",
}

NARROW_METHOD_TICK_LABELS = {
    **METHOD_TICK_LABELS,
    "w/o EM-bud.": "w/o\nEM\nbudget",
    "w/o crit.-urg.": "w/o\ncrit.\nurg.",
    "w/o implant-aware": "w/o\nimplant\naware",
}

HFSS_ORDER = [
    "surface_rest",
    "surface_sweat",
    "surface_moderate_loose",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
]

HFSS_LABELS = [
    "Surf.\nRest",
    "Surf.\nSweat",
    "Surf.\nLoose",
    "Imp10\nRest",
    "Imp10\nStress",
    "Imp30\nRest",
    "Imp30\nStress",
]

HFSS_COLORS = {
    "surface_rest": COLORS["Proposed"],
    "surface_sweat": COLORS["Lyap.-DPP"],
    "surface_moderate_loose": COLORS["w/o EM-bud."],
    "implant10_rest": COLORS["ADT-MAC"],
    "implant10_stress": COLORS["Surface nodes"],
    "implant30_rest": COLORS["ADT-MAC"],
    "implant30_stress": COLORS["Surface nodes"],
}


def apply_main_figure_style() -> None:
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": BODY_TEXT_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": BODY_TEXT_SIZE,
            "xtick.labelsize": BODY_TEXT_SIZE,
            "ytick.labelsize": BODY_TEXT_SIZE,
            "legend.fontsize": BODY_TEXT_SIZE,
            "legend.frameon": False,
            "legend.handlelength": 1.45,
            "legend.handletextpad": 0.35,
            "legend.columnspacing": 0.7,
            "legend.labelspacing": 0.25,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.4,
            "grid.linewidth": 0.32,
            "grid.alpha": 0.42,
            "grid.color": "#D6DCE2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def _save(fig: plt.Figure, stem: str, *, pad: float = 0.22, tight_rect: tuple[float, float, float, float] | None = None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad, rect=tight_rect)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600)
    stale_svg = FIG_DIR / f"{stem}.svg"
    if stale_svg.exists():
        stale_svg.unlink()
    plt.close(fig)


def _load_payload() -> dict[str, object]:
    return json.loads((HELDOUT_DIR / "candidate_metrics.json").read_text(encoding="utf-8"))


def _panel_title(ax: plt.Axes, letter: str, title: str, *, pad: float = 3.0) -> None:
    ax.set_title(f"{letter}. {title}", loc="center", pad=pad, fontsize=PANEL_TITLE_SIZE, fontweight="bold")


def _style_axes(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(y_grid, axis="y", linestyle=":", zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(direction="in", width=0.55, length=2.4, labelsize=BODY_TEXT_SIZE)
    ax.margins(x=0.04)


def _values_and_ci(frame: pd.DataFrame, metric: str, methods: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray([float(frame.loc[frame["method"].eq(method), metric].iloc[0]) for method in methods])
    ci_col = f"{metric}_ci95"
    if ci_col not in frame.columns:
        return values, None
    ci = np.asarray([float(frame.loc[frame["method"].eq(method), ci_col].fillna(0.0).iloc[0]) for method in methods])
    return values, ci


def _bar_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, methods: list[str]) -> None:
    values, ci = _values_and_ci(frame, metric, methods)
    xpos = np.arange(len(methods))
    ax.bar(
        xpos,
        values,
        yerr=ci,
        width=0.58,
        capsize=2.0 if ci is not None else 0,
        error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": "#222222"},
        color=[COLORS[m] for m in methods],
        edgecolor="#333333",
        linewidth=0.35,
        zorder=3,
    )
    zero_positions = [x for x, value in zip(xpos, values) if abs(value) < 1e-12]
    if zero_positions:
        ax.plot(
            zero_positions,
            [0.0] * len(zero_positions),
            linestyle="None",
            marker="_",
            markersize=6,
            color="#222222",
            markeredgewidth=0.75,
            zorder=4,
        )
    tick_labels = NARROW_METHOD_TICK_LABELS if len(methods) >= 4 else METHOD_TICK_LABELS
    ax.set_xticks(xpos, [tick_labels.get(method, method) for method in methods], rotation=0, ha="center")
    top_candidates = list(np.abs(values))
    if ci is not None:
        top_candidates.extend(np.abs(values + ci))
    top = max(top_candidates + [1e-9])
    ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    _style_axes(ax)


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _regime_label(label: object) -> str:
    return {
        "rest": "rest",
        "surface_sweat": "sweat",
        "surface_moderate_loose": "moderate loose",
        "recovery_rest": "recovery rest",
    }.get(str(label), str(label).replace("_", " "))


def _shade_regimes(ax: plt.Axes, frame: pd.DataFrame, *, labels: bool = False) -> None:
    timeline = frame[["t", "regime"]].drop_duplicates().sort_values("t")
    if timeline.empty:
        return
    rows = timeline.to_dict(orient="records")
    start = float(rows[0]["t"])
    current = rows[0]["regime"]
    spans: list[tuple[float, float, object]] = []
    for row in rows[1:]:
        if row["regime"] != current:
            spans.append((start, float(row["t"]), current))
            start = float(row["t"])
            current = row["regime"]
    unique_t = sorted(timeline["t"].unique())
    step = float(np.median(np.diff(unique_t))) if len(unique_t) > 1 else 1.0
    spans.append((start, float(timeline["t"].max()) + step, current))
    for left, right, name in spans:
        warm = "sweat" in str(name) or "loose" in str(name)
        ax.axvspan(left, right, color="#F4E6D7" if warm else "#EFEFEF", alpha=0.44, linewidth=0)
        if labels:
            ax.text(
                (left + right) / 2,
                REGIME_LABEL_Y,
                _regime_label(name),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=BODY_TEXT_SIZE,
                color=REGIME_LABEL_COLOR,
            )
    for _, right, _ in spans[:-1]:
        ax.axvline(right, color="#777777", linestyle="--", linewidth=0.45)


def _metric_frame(payload: dict[str, object], methods: list[str]) -> pd.DataFrame:
    metrics = payload["metrics"]
    return pd.DataFrame([{"method": method, **metrics[method]} for method in methods])


def _cell_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (values[1] - values[0]) / 2, mid, values[-1] + (values[-1] - values[-2]) / 2]


def refresh_gate_report(payload: dict[str, object]) -> None:
    phase = pd.DataFrame(payload["phase"])
    report = evaluate_candidate_gates(payload["metrics"], phase)
    (HELDOUT_DIR / "gate_result.json").write_text(
        json.dumps({"passed": report.passed, "failed_codes": report.failed_codes, "score": report.score}, indent=2),
        encoding="utf-8",
    )
    write_pass_fail_csv(HELDOUT_DIR / "candidate_pass_fail.csv", report.rows)
    (HELDOUT_DIR / "figure_gate_report.md").write_text(report.to_markdown(), encoding="utf-8")


def write_fig2() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    theta = np.linspace(0, 2 * np.pi, 300)
    layers = [
        (1.00, 0.68, "#F4B6A6", "skin/surface"),
        (0.88, 0.58, "#F7DFA6", "5 mm fat"),
        (0.72, 0.45, "#C86B6B", "muscle"),
    ]
    for width, height, color, label in layers:
        ax.fill(width * np.cos(theta), height * np.sin(theta), color=color, alpha=0.88, label=label)
    ax.scatter([-0.74, -0.64], [0.32, 0.32], marker="s", s=28, color="#444444")
    ax.text(-0.69, 0.42, "hub", ha="center", va="bottom", fontsize=7)
    ax.scatter([0.20, 0.42], [0.05, -0.18], s=42, color=COLORS["Proposed"])
    ax.text(0.20, 0.15, "10 mm", ha="center", fontsize=7)
    ax.text(0.42, -0.08, "30 mm", ha="center", fontsize=7)
    ax.annotate("fibrotic stress", xy=(0.20, 0.05), xytext=(0.48, 0.27), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=3, fontsize=6, frameon=False)
    _save(fig, "fig2_hfss_model")


def write_fig3() -> None:
    main = _fig3_proxy_frame()
    csv = main[["label", "path_loss_energy_db", "path_loss_data_db", "g_norm", "chi_norm", "p_rx_cap_norm"]].copy()
    csv.to_csv(FIG_DIR / "fig3_field_proxy_calibration.csv", index=False)

    x = np.arange(len(main))
    colors = [HFSS_COLORS[label] for label in main["label"]]
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    bar_w = 0.33
    ax_a.bar(
        x - bar_w / 2,
        main["path_loss_energy_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        alpha=0.94,
        zorder=3,
    )
    ax_a.bar(
        x + bar_w / 2,
        main["path_loss_data_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        hatch="///",
        alpha=0.58,
        zorder=3,
    )
    _panel_title(ax_a, "a", r"Path loss at $f_E$ and $f_D$")
    ax_a.set_ylabel("Path loss (dB)")
    ax_a.set_ylim(0, max(float(main["path_loss_energy_db"].max()), float(main["path_loss_data_db"].max())) * HFSS_TOP_HEADROOM)
    _style_axes(ax_a)
    ax_a.legend(
        handles=[
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, label=r"$f_E$ 13.56 MHz"),
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, hatch="///", alpha=0.58, label=r"$f_D$ 40 MHz"),
        ],
        loc="upper left",
        ncol=2,
        frameon=False,
    )

    ax_cap = ax_b.twinx()
    ax_b.bar(x, main["chi_norm"], width=0.52, color=colors, edgecolor="#333333", linewidth=0.30, alpha=0.86, zorder=3)
    ax_b.axhline(1.0, color="#444444", linewidth=0.70, linestyle=":", alpha=0.78, zorder=2)
    ax_cap.semilogy(
        x,
        main["p_rx_cap_norm"],
        marker="D",
        linestyle="--",
        color="#222222",
        markersize=3.4,
        linewidth=0.92,
        markerfacecolor="white",
        markeredgewidth=0.72,
        zorder=5,
    )
    _panel_title(ax_b, "b", "Field-load proxy and Rx-power cap")
    ax_b.set_ylabel(r"$\chi_{\rm norm}$")
    ax_cap.set_ylabel(r"$p^{\rm norm}_{\rm rx,cap}$")
    ax_b.set_ylim(0, max(1.15, float(main["chi_norm"].max()) * 1.55))
    ax_cap.set_ylim(0.05, 2.0)
    ax_cap.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.25, 0.5, 1.0]))
    ax_cap.yaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.25", "0.5", "1"]))
    ax_cap.yaxis.set_minor_locator(FixedLocator([]))
    ax_b.set_xticks(x, HFSS_LABELS, rotation=0, ha="center")
    ax_b.tick_params(axis="x", labelsize=BODY_TEXT_SIZE, pad=1)
    ax_cap.tick_params(direction="in", width=0.55, length=2.4, labelsize=BODY_TEXT_SIZE)
    _style_axes(ax_b)
    ax_cap.grid(False)
    ax_b.legend(
        handles=[
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, label=r"$\chi_{\rm norm}$"),
            Line2D([0], [0], marker="D", color="#222222", linestyle="--", markersize=3.4, markerfacecolor="white", label=r"$p^{\rm norm}_{\rm rx,cap}$"),
        ],
        loc="upper right",
        ncol=2,
        fontsize=BODY_TEXT_SIZE - 0.3,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    fig.align_ylabels([ax_a, ax_b])
    _save(fig, "fig3_field_proxy_calibration", pad=0.18)


def _fig3_proxy_frame() -> pd.DataFrame:
    sched = pd.read_csv(TORSO_OUTPUTS / "torso_scheduler_library.csv")
    main = sched[sched["sched_include_main"].astype(str).str.lower().isin(["true", "1"])].copy()
    main["label"] = pd.Categorical(main["label"], categories=HFSS_ORDER, ordered=True)
    main = main.sort_values("label")
    main["label"] = main["label"].astype(str)
    return main


def write_fig3_path_loss_only() -> None:
    main = _fig3_proxy_frame()
    x = np.arange(len(main))
    colors = [HFSS_COLORS[label] for label in main["label"]]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 1.70))
    bar_w = 0.33
    ax.bar(
        x - bar_w / 2,
        main["path_loss_energy_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        alpha=0.94,
        zorder=3,
    )
    ax.bar(
        x + bar_w / 2,
        main["path_loss_data_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        hatch="///",
        alpha=0.58,
        zorder=3,
    )
    ax.set_title(r"Path loss at $f_E$ and $f_D$", pad=3.0, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
    ax.set_ylabel("Path loss (dB)")
    ax.set_ylim(0, max(float(main["path_loss_energy_db"].max()), float(main["path_loss_data_db"].max())) * HFSS_TOP_HEADROOM)
    ax.set_xticks(x, HFSS_LABELS, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=BODY_TEXT_SIZE, pad=1)
    _style_axes(ax)
    ax.legend(
        handles=[
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, label=r"$f_E$ 13.56 MHz"),
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, hatch="///", alpha=0.58, label=r"$f_D$ 40 MHz"),
        ],
        loc="upper left",
        ncol=2,
        frameon=False,
    )
    _save(fig, "fig3_path_loss_variation", pad=0.12)


def write_fig4(payload: dict[str, object]) -> None:
    frame = _metric_frame(payload, MAIN_METHODS)
    frame.to_csv(FIG_DIR / "fig4_main_method_comparison.csv", index=False)
    panels = [
        ("stress_shortage_severity", "Implant shortage"),
        ("surface_served_ratio", "Surface service"),
        ("p95_backlog", "p95 backlog"),
        ("recovery_time", "Recovery time"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.18))
    for index, (ax, (metric, title)) in enumerate(zip(axes.ravel(), panels)):
        _bar_panel(ax, frame, metric, MAIN_METHODS)
        _panel_title(ax, chr(ord("a") + index), title)
    _save(fig, "fig4_main_method_comparison")


def write_fig5() -> None:
    cfg = SimConfig()
    raw = pd.read_csv(HELDOUT_DIR / "raw" / "condition_switch_per_slot.csv")
    frames = build_fig4_mechanism_frames(raw, cfg)
    csv = frames["csv"]
    protected_margin_floor = cfg.protected_margin_frac
    floor = csv[csv["panel"].eq("margin")][["t", "regime"]].drop_duplicates()
    floor["panel"] = "margin"
    floor["method"] = "Protected floor"
    floor["curve"] = "Protected margin floor"
    floor["value"] = protected_margin_floor
    csv = pd.concat([csv, floor[["panel", "method", "curve", "t", "regime", "value"]]], ignore_index=True)
    csv.to_csv(FIG_DIR / "fig5_medium_condition_response.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.65), sharex=True)
    for index, ax in enumerate(axes):
        _shade_regimes(ax, csv, labels=index == 0)
        _style_axes(ax)

    allocation = csv[csv["panel"].eq("allocation")]
    for curve, color in [("Critical implant nodes", COLORS["Proposed"]), ("Surface nodes", COLORS["Surface nodes"])]:
        sub = allocation[allocation["curve"].eq(curve)].sort_values("t")
        axes[0].plot(sub["t"], _smooth(sub["value"], 18), color=color, linewidth=0.95, label=curve)
    axes[0].set_ylim(-0.005, FIG5_ALLOCATION_YMAX)

    em = csv[csv["panel"].eq("em")]
    proposed_em = em[em["method"].eq("Proposed")].sort_values("t")
    axes[1].plot(proposed_em["t"], _smooth(proposed_em["value"], 18), color=COLORS["Proposed"], linewidth=0.95, label=METHOD_TICK_LABELS["Proposed"])
    axes[1].axhline(1.0, color="#333333", linestyle=":", linewidth=0.70, label="Budget")

    margin = csv[csv["panel"].eq("margin")]
    for method in ["Proposed", "w/o crit.-urg."]:
        sub = margin[margin["method"].eq(method)].sort_values("t")
        axes[2].plot(
            sub["t"],
            _smooth(sub["value"], 18),
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=0.95,
            label=METHOD_TICK_LABELS.get(method, method).replace("\n", " "),
        )
    floor_sub = margin[margin["curve"].eq("Protected margin floor")].sort_values("t")
    if not floor_sub.empty:
        axes[2].plot(floor_sub["t"], floor_sub["value"], color="#333333", linestyle=":", linewidth=0.70, label="Protected floor")

    _panel_title(axes[0], "a", "Energy-support allocation")
    _panel_title(axes[1], "b", "EM-budget usage")
    _panel_title(axes[2], "c", "Critical implant margin")
    axes[0].set_ylabel(r"$u_i^E$")
    axes[1].set_ylabel("EM util.")
    axes[2].set_ylabel("Margin")
    axes[2].set_xlabel("Slot")
    axes[0].plot(
        [FIG5_LEGEND_LINE_X0, FIG5_LEGEND_LINE_X1],
        [FIG5_LEGEND_CRITICAL_Y, FIG5_LEGEND_CRITICAL_Y],
        color=COLORS["Proposed"],
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG5_LEGEND_TEXT_X, FIG5_LEGEND_CRITICAL_Y, "Critical implant nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    axes[0].plot(
        [FIG5_LEGEND_LINE_X0, FIG5_LEGEND_LINE_X1],
        [FIG5_LEGEND_SURFACE_Y, FIG5_LEGEND_SURFACE_Y],
        color=COLORS["Surface nodes"],
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG5_LEGEND_TEXT_X, FIG5_LEGEND_SURFACE_Y, "Surface nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    axes[1].legend(fontsize=BODY_TEXT_SIZE, loc="upper right", frameon=False)
    axes[2].legend(fontsize=BODY_TEXT_SIZE, loc="lower right", frameon=False)
    _save(fig, "fig5_medium_condition_response")


def write_fig6() -> None:
    cfg = SimConfig()
    raw = pd.read_csv(HELDOUT_DIR / "raw" / "stress_per_slot.csv")
    affected = raw[raw["node"].isin(cfg.stress_nodes)]
    csv = affected.groupby(["method", "t"], as_index=False).agg(
        e_next=("e_next", "mean"),
        xi=("xi", "mean"),
        severity=("xi", lambda s: float((s / cfg.E_min_critical).mean())),
    )
    csv.to_csv(FIG_DIR / "fig6_implant_stress_window.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)
    for ax in axes:
        ax.axvspan(cfg.stress_start, cfg.stress_end, color="#D8D8D8", alpha=0.46, linewidth=0)
        ax.axvline(cfg.stress_start, color="#777777", linestyle="--", linewidth=0.45)
        ax.axvline(cfg.stress_end, color="#777777", linestyle="--", linewidth=0.45)
        _style_axes(ax)
    for method in STRESS_METHODS:
        sub = csv[csv["method"].eq(method)].sort_values("t")
        if sub.empty:
            continue
        label = METHOD_TICK_LABELS.get(method, method).replace("\n", " ")
        axes[0].plot(sub["t"], _smooth(sub["e_next"], 10), label=label, color=COLORS.get(method), linestyle=LINESTYLES.get(method, "-"), linewidth=0.95)
        axes[1].plot(sub["t"], _smooth(sub["severity"], 4), label=label, color=COLORS.get(method), linestyle=LINESTYLES.get(method, "-"), linewidth=0.95)
    axes[0].axhline(cfg.E_min_critical, color="#333333", linestyle=":", linewidth=0.70, label=r"$E_i^{min}$")
    axes[0].set_ylim(-0.05, 1.32)
    axes[1].set_ylim(-0.05, FIG6_SHORTAGE_YMAX)
    axes[1].text(
        (cfg.stress_start + cfg.stress_end) / 2,
        FIG6_STRESS_WINDOW_LABEL_Y,
        "stress window",
        ha="center",
        va="center",
        fontsize=BODY_TEXT_SIZE,
        color=REGIME_LABEL_COLOR,
    )
    _panel_title(axes[0], "a", "Stressed implant energy")
    _panel_title(axes[1], "b", "Normalized shortage")
    axes[0].set_ylabel("Energy")
    axes[1].set_ylabel("Shortage")
    axes[1].set_xlabel("Slot")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        fontsize=BODY_TEXT_SIZE - 0.6,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.65,
        labelspacing=0.22,
    )
    _save(fig, "fig6_implant_stress_window")


def write_fig7(payload: dict[str, object]) -> None:
    frame = _metric_frame(payload, ABLATION_METHODS)
    frame.to_csv(FIG_DIR / "fig7_ablation_interface_check.csv", index=False)
    panels = [
        ("stress_shortage_severity", "Implant shortage"),
        ("implant_energy_p05", "Energy p05"),
        ("em_violation_rate", "EM violation"),
        ("served_workload_ratio", "Served ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.18))
    for index, (ax, (metric, title)) in enumerate(zip(axes.ravel(), panels)):
        _bar_panel(ax, frame, metric, ABLATION_METHODS)
        _panel_title(ax, chr(ord("a") + index), title)
    _save(fig, "fig7_ablation_interface_check")


def write_fig8(payload: dict[str, object]) -> None:
    phase = pd.DataFrame(payload["phase"])
    phase.to_csv(FIG_DIR / "fig8_operating_region.csv", index=False)
    pivot = phase.pivot(index="P_H_multiplier", columns="B_EM_multiplier", values="stress_shortage_severity").sort_index()
    b_values = np.asarray(pivot.columns, dtype=float)
    p_values = np.asarray(pivot.index, dtype=float)
    b_grid, p_grid = np.meshgrid(b_values, p_values)
    b_edges = _cell_edges(b_values)
    p_edges = _cell_edges(p_values)

    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    vmax = max(0.20, float(phase["stress_shortage_severity"].max()))
    image = ax.pcolormesh(b_edges, p_edges, pivot.to_numpy(), cmap="Blues", vmin=0.0, vmax=vmax, shading="flat")
    handles: list[Line2D] = []
    data = pivot.to_numpy()
    if float(np.nanmin(data)) <= 0.05 <= float(np.nanmax(data)):
        ax.contour(b_grid, p_grid, data, levels=[0.05], colors="#111111", linestyles="--", linewidths=0.72)
        handles.append(Line2D([0], [0], color="#111111", linestyle="--", linewidth=0.72, label="0.05 boundary"))
    marked = phase[phase["is_final_workpoint"].astype(bool)] if "is_final_workpoint" in phase.columns else phase.iloc[0:0]
    if not marked.empty:
        row = marked.iloc[0]
        star_x = float(row["B_EM_multiplier"])
        star_y = float(row["P_H_multiplier"])
        star_label = "final workpoint"
    else:
        star_x = float(b_values[-1])
        star_y = float(p_values[-1])
        star_label = "final (1,1)"
    ax.scatter([star_x], [star_y], marker="*", s=58, color=COLORS["ADT-MAC"], edgecolor="white", linewidth=0.55, zorder=3)
    handles.append(Line2D([0], [0], marker="*", markersize=6.5, markerfacecolor=COLORS["ADT-MAC"], markeredgecolor="white", color="none", label=star_label))
    ax.set_xlabel(r"$B_{\rm EM}$ multiplier")
    ax.set_ylabel(r"$P_H$ multiplier")
    ax.set_xticks(b_values)
    ax.set_yticks(p_values)
    ax.set_ylim(p_edges[0], p_edges[-1] + 0.035)
    ax.tick_params(axis="x", rotation=0)
    _style_axes(ax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    cbar.set_label("Shortage severity")
    cbar.ax.tick_params(labelsize=BODY_TEXT_SIZE)
    ax.legend(
        handles=handles,
        fontsize=BODY_TEXT_SIZE - 0.2,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.85,
    )
    _save(fig, "fig8_operating_region")


def write_fig9() -> None:
    robustness = MAIN_EXPERIMENT / "output_0515_chest_tier1_label_robustness" / "output" / "label_robustness_summary.csv"
    if robustness.exists():
        frame = pd.read_csv(robustness)
        frame.to_csv(FIG_DIR / "fig9_state_label_robustness.csv", index=False)
        panels = [
            ("random_flip", "Random label flip", "Perturbation rate"),
            ("stress_false_negative", "Stress false negative", r"$p_{\rm FN}$"),
            ("stuck_mixed_burst", "Stuck mixed burst", "Burst length"),
            ("rssi_calibrated", "Calibrated RSSI estimator", "Estimator"),
        ]
        metrics = [
            ("stress_shortage_severity", "Shortage", COLORS["ADT-MAC"], "o"),
            ("surface_served_ratio", "Surface service", COLORS["Proposed"], "s"),
            ("em_violation_rate", "EM violation", COLORS["w/o implant-aware"], "^"),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(3.5, 4.55), sharey=True)
        for index, (ax, (panel, title, xlabel)) in enumerate(zip(axes, panels)):
            sub = frame[frame["panel"].eq(panel)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            if panel == "rssi_calibrated":
                sub = sub.sort_values("perturbation")
                x = np.arange(len(sub), dtype=float)
                ax.set_xticks(x, [str(value) for value in sub["perturbation"]], rotation=0, ha="center")
            else:
                sub["x"] = pd.to_numeric(sub["perturbation"], errors="coerce")
                sub = sub.dropna(subset=["x"]).sort_values("x")
                x = sub["x"].to_numpy()
                ax.set_xticks(x)
            for metric, label, color, marker in metrics:
                ax.plot(
                    x,
                    sub[metric].astype(float).to_numpy(),
                    marker=marker,
                    linewidth=0.95,
                    markersize=3.1,
                    label=label,
                    color=color,
                )
            _panel_title(ax, chr(ord("a") + index), title)
            ax.set_ylim(-0.04, 1.06)
            ax.set_ylabel("Metric")
            ax.set_xlabel(xlabel)
            _style_axes(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=BODY_TEXT_SIZE, bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0)
        _save(fig, "fig9_state_label_robustness", tight_rect=(0.0, 0.0, 1.0, 0.94))
        return

    summary = MAIN_EXPERIMENT / "output_0515_chest_state_err" / "output" / "estimator_prototype_summary.csv"
    if not summary.exists():
        stale_svg = FIG_DIR / "fig9_state_label_robustness.svg"
        if stale_svg.exists():
            stale_svg.unlink()
        return
    frame = pd.read_csv(summary)
    metrics = ["label_accuracy", "stress_false_negative_rate", "stress_shortage_severity", "em_violation_rate", "rx_cap_violation_rate"]
    values = [float(frame[metric].iloc[0]) for metric in metrics]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(range(len(metrics)), values, color=COLORS["Proposed"], edgecolor="#333333", linewidth=0.35, zorder=3)
    ax.set_xticks(range(len(metrics)), [metric.replace("_", "\n") for metric in metrics], rotation=0, ha="center")
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("Value")
    _style_axes(ax)
    _save(fig, "fig9_state_label_robustness")


def write_revision_generated() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    materials = pd.read_csv(TORSO_OUTPUTS / "material_table.csv").rename(columns={"epsilon_r": "relative_permittivity", "sigma_S_per_m": "conductivity_S_per_m"})
    materials.to_csv(REV_DIR / "tissue_materials.csv", index=False)

    setup_rows = [
        {"item": "model", "setting": "chest/torso surrogate cross-section with surface electrodes and implant receivers"},
        {"item": "layers", "setting": "5 mm fat shell over muscle; 10 mm implant resides in muscle"},
        {"item": "frequencies", "setting": "f_E=13.56 MHz, f_D=40 MHz; auxiliary sweep at 1/10/21/40 MHz where solved"},
        {"item": "states", "setting": "surface rest/sweat/moderate-loose plus implant 10/30 mm rest/stress"},
        {"item": "stress", "setting": "fibrotic/degraded local material state in muscle for implant stress cases"},
        {"item": "scheduler calibration", "setting": "implant LP-facing g_norm uses 2x scheduler energy scale; raw S21 gain retained as g_norm_raw_s21"},
    ]
    pd.DataFrame(setup_rows).to_csv(REV_DIR / "hfss_setup_table.csv", index=False)

    fields = pd.read_csv(TORSO_OUTPUTS / "field_proxy_samples.csv")
    ref = fields.loc[fields["case"].eq("t_surf_rest")].iloc[0]
    chi_rows = []
    for _, row in fields.iterrows():
        label = row["condition"]
        if row["case"] == "t_surf_loose":
            label = "surface_moderate_loose"
        values = {
            "chi_p90": row["Q_p90_W_per_m3"] / ref["Q_p90_W_per_m3"],
            "chi_p95": row["Q_p95_W_per_m3"] / ref["Q_p95_W_per_m3"],
            "chi_p99": row["Q_p99_W_per_m3"] / ref["Q_p99_W_per_m3"],
            "chi_mean": row["Q_mean_W_per_m3"] / ref["Q_mean_W_per_m3"],
        }
        span = 100.0 * (max(values.values()) - min(values.values())) / max(values["chi_p95"], 1e-12)
        chi_rows.append({"label": label, **values, "relative_span_pct": span})
    pd.DataFrame(chi_rows).to_csv(REV_DIR / "chi_sensitivity.csv", index=False)

    freq = pd.read_csv(TORSO_OUTPUTS / "torso_frequency_summary.csv")
    fd = freq.rename(columns={"S21_dB": "s21_db", "S21_mag2_norm_to_13p56": "r_norm"})
    fd[["condition", "freq_MHz", "s21_db", "r_norm"]].rename(columns={"condition": "label"}).to_csv(REV_DIR / "fd_sensitivity.csv", index=False)


def copy_supplementary_outputs() -> None:
    mapping = {
        "necessity_ablation.csv": MAIN_EXPERIMENT / "output_0515_chest_necessity" / "output" / "necessity_ablation.csv",
        "lyap_v_sweep.csv": MAIN_EXPERIMENT / "output_0515_chest_lyap_v" / "output" / "lyap_v_sweep.csv",
        "estimator_closed_loop_summary.csv": MAIN_EXPERIMENT / "output_0515_chest_tier1_estimator" / "output" / "estimator_closed_loop_summary.csv",
        "label_robustness_summary.csv": MAIN_EXPERIMENT / "output_0515_chest_tier1_label_robustness" / "output" / "label_robustness_summary.csv",
        "network_scale_summary.csv": MAIN_EXPERIMENT / "output_0515_chest_tier1_network_scale" / "output" / "network_scale_summary.csv",
        "estimator_prototype_summary.csv": MAIN_EXPERIMENT / "output_0515_chest_state_err" / "output" / "estimator_prototype_summary.csv",
        "small_network_summary.csv": MAIN_EXPERIMENT / "output_0515_chest_small_network" / "output" / "small_network_summary.csv",
    }
    for name, src in mapping.items():
        if src.exists():
            pd.read_csv(src).to_csv(REV_DIR / name, index=False)


def main() -> None:
    apply_main_figure_style()
    payload = _load_payload()
    refresh_gate_report(payload)
    write_fig3()
    write_fig3_path_loss_only()
    write_fig4(payload)
    write_fig5()
    write_fig6()
    write_fig7(payload)
    write_fig8(payload)
    write_fig9()
    write_revision_generated()
    copy_supplementary_outputs()
    print(f"rebuilt torso manuscript assets in {FIG_DIR}")


if __name__ == "__main__":
    main()
