from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT
PACKAGE_DIR = REPO_ROOT
PAPER_DIR = REPO_ROOT
FIGURE_DIR = Path(os.environ.get("IOTJ_FIGURE_DIR", REPO_ROOT / "figures"))
LEGACY_FIGURE_DIR = FIGURE_DIR
FIGURE_DATA_DIR = Path(os.environ.get("IOTJ_FIGURE_DATA_DIR", FIGURE_DIR))
ROBUSTNESS_SUMMARY = Path(os.environ.get("IOTJ_ROBUSTNESS_SUMMARY", REPO_ROOT / "data" / "state_misclassification_summary.csv"))
ROBUSTNESS_LEADERBOARD = Path(os.environ.get("IOTJ_ROBUSTNESS_LEADERBOARD", REPO_ROOT / "data" / "iter10_leaderboard.csv"))
HFSS_LIBRARY = Path(os.environ.get("HFSS_SCHED_LIBRARY", REPO_ROOT / "data" / "calibrated_library_sched.csv"))
STYLE_FILE = Path(os.environ.get("IOTJ_STYLE_FILE", REPO_ROOT / "ieee_trans.mplstyle"))
HFSS_MODEL_BASE = FIGURE_DIR / "fig2_hfss_model_legacy_base.png"

BODY_TEXT_SIZE = 8
PANEL_TITLE_SIZE = 9
REGIME_LABEL_Y = 0.90
REGIME_LABEL_COLOR = "#111111"
HFSS_TOP_HEADROOM = 1.25
FIG4_ALLOCATION_YMAX = 0.24
FIG4A_LEGEND_LINE_X0 = 340
FIG4A_LEGEND_LINE_X1 = 380
FIG4A_LEGEND_TEXT_X = 395
FIG4A_LEGEND_CRITICAL_Y = 0.118
FIG4A_LEGEND_SURFACE_Y = 0.088
FIG5_SHORTAGE_YMAX = 1.22
FIG5_STRESS_WINDOW_LABEL_Y = 1.14
FIG8_A_YMAX = 0.275
FIG8_A_ANNOTATION_AXY = 0.86
FIG8_B_MARGIN_LABEL_X = 0.055
FIG8_B_MARGIN_LABEL_Y = 0.68
FIG8_B_FN_LABEL_X = 0.650
FIG8_B_FN_LABEL_Y = 0.84
FIG8_C_SHORTAGE_THRESHOLD = 0.05

STYLE_OVERRIDES = {
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
}

SCHEME_A = {
    "Proposed": "#1B4F8A",
    "ADT-MAC": "#B03A2E",
    "Lyap.-DPP": "#1E8449",
    "w/o EM-bud.": "#6FA3D0",
    "w/o crit.-urg.": "#9DB8CE",
    "w/o implant-aware": "#7F7F7F",
    "Surface nodes": "#E67E22",
}

LINESTYLES = {
    "Proposed": "-",
    "ADT-MAC": "--",
    "Lyap.-DPP": "-.",
    "w/o EM-bud.": ":",
    "w/o crit.-urg.": (0, (5, 2)),
    "w/o implant-aware": (0, (3, 1, 1, 1)),
}

METHOD_LABELS = {
    "Proposed": "Proposed",
    "ADT-MAC": "ADT-MAC",
    "Lyap.-DPP": "Lyap.-DPP",
    "w/o EM-bud.": "w/o EM-bud.",
    "w/o crit.-urg.": "w/o crit.-urg.",
    "w/o implant-aware": "w/o implant-aware",
}

METHOD_TICK_LABELS = {
    "Proposed": "Proposed",
    "ADT-MAC": "ADT-MAC",
    "Lyap.-DPP": "Lyap.\n-DPP",
    "w/o EM-bud.": "w/o\nEM\nbudget",
    "w/o crit.-urg.": "w/o\ncrit.\nurg.",
    "w/o implant-aware": "w/o\nimplant\naware",
}

MAIN_METHODS = ["Proposed", "ADT-MAC", "Lyap.-DPP"]
ABLATION_METHODS = ["Proposed", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware"]
STRESS_METHODS = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o crit.-urg."]

HFSS_ORDER = [
    "surface_rest",
    "surface_moderate_loose",
    "surface_sweat",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
]

HFSS_LABELS = [
    "Surf.\nRest",
    "Surf.\nLoose",
    "Surf.\nSweat",
    "Imp10\nRest",
    "Imp10\nStress",
    "Imp30\nRest",
    "Imp30\nStress",
]

CASE_COLORS = {
    "surface_rest": "#1B4F8A",
    "surface_moderate_loose": "#6FA3D0",
    "surface_sweat": "#1E8449",
    "implant10_rest": "#B03A2E",
    "implant10_stress": "#E67E22",
    "implant30_rest": "#B03A2E",
    "implant30_stress": "#E67E22",
}


def panel_title(letter: str, title: str) -> str:
    return f"{letter}. {title}"


def default_output_dir(timestamp: str | None = None) -> Path:
    if timestamp:
        return PAPER_DIR / f"figures_standard_{timestamp}"
    return FIGURE_DIR


def apply_style() -> None:
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    plt.rcParams.update(STYLE_OVERRIDES)


def set_panel_title(ax: plt.Axes, letter: str, title: str, *, pad: float = 3) -> None:
    ax.set_title(panel_title(letter, title), loc="center", pad=pad, fontsize=PANEL_TITLE_SIZE, fontweight="bold")


def legend(ax: plt.Axes, *args, **kwargs):
    kwargs.setdefault("frameon", False)
    return ax.legend(*args, **kwargs)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, pad: float = 0.25, tight: bool = True, svg: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=pad)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    if svg:
        fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


def copy_source_csv(out_dir: Path, stem: str) -> None:
    source = FIGURE_DATA_DIR / f"{stem}.csv"
    destination = out_dir / f"{stem}.csv"
    if source.exists() and source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def values_and_ci(frame: pd.DataFrame, metric: str, methods: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray([float(frame.loc[frame["method"].eq(method), metric].iloc[0]) for method in methods])
    ci_col = f"{metric}_ci95"
    if ci_col not in frame.columns:
        return values, None
    ci = np.asarray([float(frame.loc[frame["method"].eq(method), ci_col].fillna(0.0).iloc[0]) for method in methods])
    return values, ci


def style_axes(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.grid(y_grid, axis="y", linestyle=":", zorder=0)
    ax.grid(False, axis="x")
    ax.tick_params(direction="in", width=0.55, length=2.4, labelsize=BODY_TEXT_SIZE)
    ax.margins(x=0.04)


def set_data_ylim(ax: plt.Axes, values: list[float] | np.ndarray | pd.Series, *, lower: float | None = None, pad_frac: float = 0.14) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    y_min = float(finite.min()) if lower is None else lower
    y_max = float(finite.max())
    span = max(y_max - y_min, abs(y_max) * 0.08, 1e-6)
    ax.set_ylim(y_min, y_max + span * pad_frac)


def fig8_margin_axis_limits(
    values: list[float] | np.ndarray | pd.Series,
    errors: list[float] | np.ndarray | pd.Series | None = None,
    *,
    lower_floor: float = 0.0,
    pad_frac: float = 0.18,
    min_span: float = 0.01,
) -> tuple[float, float]:
    center = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(center)
    center = center[finite_mask]
    if center.size == 0:
        return lower_floor, lower_floor + min_span
    if errors is None:
        err = np.zeros_like(center)
    else:
        err = np.asarray(errors, dtype=float)[finite_mask]
        err = np.where(np.isfinite(err), err, 0.0)
    low = float(np.min(center - err))
    high = float(np.max(center + err))
    span = max(high - low, min_span)
    lower = max(lower_floor, low - span * pad_frac)
    upper = high + span * pad_frac
    return lower, upper


def fig8_shortage_axis_values(severity: list[float] | np.ndarray | pd.Series) -> tuple[np.ndarray, str, float]:
    raw = np.asarray(severity, dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return raw, "", 1.0
    max_abs = float(np.max(np.abs(finite)))
    if max_abs < 1e-3:
        scale = 1e4
        scale_label = r"$\times10^{-4}$"
    elif max_abs < 1e-2:
        scale = 1e3
        scale_label = r"$\times10^{-3}$"
    else:
        scale = 1.0
        scale_label = ""
    scaled = raw * scale
    scaled_finite = scaled[np.isfinite(scaled)]
    top = float(np.max(scaled_finite)) if scaled_finite.size else 0.0
    upper = max(top * 1.32, 0.8 if scale > 1.0 else 0.08)
    return scaled, scale_label, upper


def fig8_stress_fn_coarse_sweep(leaderboard: pd.DataFrame) -> pd.DataFrame:
    frame = leaderboard.copy()
    frame["param_value_numeric"] = pd.to_numeric(frame["param_value"], errors="coerce")
    boundary = frame[
        frame["model"].eq("stress_false_negative")
        & frame["param_name"].eq("p_fn")
        & frame["iteration"].eq(2)
    ].copy()
    if boundary.empty:
        raise ValueError("Missing Fig.8b stress-FN coarse sweep rows")
    boundary = boundary.dropna(subset=["param_value_numeric"])
    return boundary.sort_values("param_value_numeric")


def bar_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, methods: list[str]) -> None:
    values, ci = values_and_ci(frame, metric, methods)
    xpos = np.arange(len(methods))
    ax.bar(
        xpos,
        values,
        yerr=ci,
        width=0.58,
        capsize=2.0 if ci is not None else 0,
        error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": "#222222"},
        color=[SCHEME_A[m] for m in methods],
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
    ax.set_xticks(xpos, [METHOD_TICK_LABELS.get(m, METHOD_LABELS.get(m, m)) for m in methods], rotation=0, ha="center")
    top_candidates = list(np.abs(values))
    if ci is not None:
        top_candidates.extend(np.abs(values + ci))
    top = max(top_candidates + [1e-9])
    ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
    style_axes(ax)


def render_bar_grid(
    out_dir: Path,
    stem: str,
    methods: list[str],
    panels: list[tuple[str, str]],
    *,
    figsize: tuple[float, float] = (3.5, 3.18),
) -> None:
    frame = pd.read_csv(FIGURE_DATA_DIR / f"{stem}.csv")
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for idx, (ax, (metric, title)) in enumerate(zip(axes.ravel(), panels)):
        bar_panel(ax, frame, metric, methods)
        set_panel_title(ax, chr(ord("a") + idx), title)
    save_figure(fig, out_dir, stem)
    copy_source_csv(out_dir, stem)


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def regime_label(label: object) -> str:
    return {
        "rest": "rest",
        "surface_sweat": "sweat",
        "surface_moderate_loose": "moderate loose",
        "recovery_rest": "recovery rest",
    }.get(str(label), str(label).replace("_", " "))


def shade_regimes(ax: plt.Axes, frame: pd.DataFrame, *, labels: bool = False) -> None:
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
                regime_label(name),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=BODY_TEXT_SIZE,
                color=REGIME_LABEL_COLOR,
            )
    for _, right, _ in spans[:-1]:
        ax.axvline(right, color="#777777", linestyle="--", linewidth=0.45)


def render_fig4(out_dir: Path) -> None:
    frame = pd.read_csv(FIGURE_DATA_DIR / "fig4_condition_switching_response.csv")
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.55), sharex=True)
    for idx, ax in enumerate(axes):
        shade_regimes(ax, frame, labels=idx == 0)
        style_axes(ax)

    allocation = frame[frame["panel"].eq("allocation")]
    for curve, color in [("Critical implant nodes", SCHEME_A["Proposed"]), ("Surface nodes", SCHEME_A["Surface nodes"])]:
        sub = allocation[allocation["curve"].eq(curve)].sort_values("t")
        axes[0].plot(sub["t"], smooth(sub["value"], 18), color=color, linewidth=0.95, label=curve)
    axes[0].set_ylim(-0.005, FIG4_ALLOCATION_YMAX)

    em = frame[frame["panel"].eq("em")]
    proposed_em = em[em["method"].eq("Proposed")].sort_values("t")
    axes[1].plot(proposed_em["t"], smooth(proposed_em["value"], 18), color=SCHEME_A["Proposed"], linewidth=0.95, label="Proposed")
    axes[1].axhline(1.0, color="#333333", linestyle=":", linewidth=0.70, label="Budget")

    margin = frame[frame["panel"].eq("margin")]
    for method in ["Proposed", "w/o crit.-urg."]:
        sub = margin[margin["method"].eq(method)].sort_values("t")
        axes[2].plot(
            sub["t"],
            smooth(sub["value"], 18),
            color=SCHEME_A[method],
            linestyle=LINESTYLES[method],
            linewidth=0.95,
            label=METHOD_LABELS[method],
        )
    axes[2].axhline(0.0, color="#333333", linestyle=":", linewidth=0.70)

    set_panel_title(axes[0], "a", "Energy-support allocation")
    set_panel_title(axes[1], "b", "EM-budget usage")
    set_panel_title(axes[2], "c", "Critical implant margin")
    axes[0].set_ylabel(r"$u_i^E$")
    axes[1].set_ylabel("EM util.")
    axes[2].set_ylabel("Margin")
    axes[2].set_xlabel("Slot")
    axes[0].plot(
        [FIG4A_LEGEND_LINE_X0, FIG4A_LEGEND_LINE_X1],
        [FIG4A_LEGEND_CRITICAL_Y, FIG4A_LEGEND_CRITICAL_Y],
        color=SCHEME_A["Proposed"],
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG4A_LEGEND_TEXT_X, FIG4A_LEGEND_CRITICAL_Y, "Critical implant nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    axes[0].plot(
        [FIG4A_LEGEND_LINE_X0, FIG4A_LEGEND_LINE_X1],
        [FIG4A_LEGEND_SURFACE_Y, FIG4A_LEGEND_SURFACE_Y],
        color=SCHEME_A["Surface nodes"],
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG4A_LEGEND_TEXT_X, FIG4A_LEGEND_SURFACE_Y, "Surface nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    legend(axes[1], fontsize=BODY_TEXT_SIZE, loc="best")
    legend(axes[2], fontsize=BODY_TEXT_SIZE, loc="best")
    save_figure(fig, out_dir, "fig4_condition_switching_response", pad=0.22)
    copy_source_csv(out_dir, "fig4_condition_switching_response")


def stress_window(frame: pd.DataFrame) -> tuple[float, float] | None:
    by_t = frame.groupby("t")["severity"].max()
    active = by_t[by_t > 1e-12]
    if active.empty:
        return None
    t_values = sorted(by_t.index.astype(float))
    step = float(np.median(np.diff(t_values))) if len(t_values) > 1 else 1.0
    return float(active.index.min()), float(active.index.max()) + step


def render_fig5(out_dir: Path) -> None:
    frame = pd.read_csv(FIGURE_DATA_DIR / "fig5_implant_stress_response.csv")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)
    window = stress_window(frame)
    for ax in axes:
        if window is not None:
            start, end = window
            ax.axvspan(start, end, color="#D8D8D8", alpha=0.46, linewidth=0)
            ax.axvline(start, color="#777777", linestyle="--", linewidth=0.45)
            ax.axvline(end, color="#777777", linestyle="--", linewidth=0.45)
        style_axes(ax)
    for method in STRESS_METHODS:
        sub = frame[frame["method"].eq(method)].sort_values("t")
        if sub.empty:
            continue
        axes[0].plot(
            sub["t"],
            smooth(sub["e_next"], 10),
            color=SCHEME_A[method],
            linestyle=LINESTYLES[method],
            linewidth=0.95,
            label=METHOD_LABELS.get(method, method),
        )
        axes[1].plot(
            sub["t"],
            smooth(sub["severity"], 4),
            color=SCHEME_A[method],
            linestyle=LINESTYLES[method],
            linewidth=0.95,
            label=METHOD_LABELS.get(method, method),
        )
    axes[0].axhline(0.35, color="#333333", linestyle=":", linewidth=0.70, label=r"$E_i^{min}$")
    energy_with_floor = frame["e_next"].tolist() + [0.35]
    set_data_ylim(axes[0], energy_with_floor, lower=-0.02, pad_frac=0.18)
    axes[1].set_ylim(-0.05, FIG5_SHORTAGE_YMAX)
    if window is not None:
        start, end = window
        axes[1].text((start + end) / 2, FIG5_STRESS_WINDOW_LABEL_Y, "stress window", ha="center", va="center", fontsize=BODY_TEXT_SIZE, color=REGIME_LABEL_COLOR)
    set_panel_title(axes[0], "a", "Stressed implant energy")
    set_panel_title(axes[1], "b", "Normalized shortage")
    axes[0].set_ylabel("Energy")
    axes[1].set_ylabel("Shortage")
    axes[1].set_xlabel("Slot")
    legend(axes[0], fontsize=BODY_TEXT_SIZE, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.0)
    save_figure(fig, out_dir, "fig5_implant_stress_response", pad=0.22)
    copy_source_csv(out_dir, "fig5_implant_stress_response")


def render_fig7(out_dir: Path) -> None:
    phase = pd.read_csv(FIGURE_DATA_DIR / "fig7_phase_diagram_implant_shortage.csv")
    pivot = phase.pivot(index="P_H_multiplier", columns="B_EM_multiplier", values="stress_shortage_severity").sort_index()
    b_values = np.asarray(pivot.columns, dtype=float)
    p_values = np.asarray(pivot.index, dtype=float)
    b_grid, p_grid = np.meshgrid(b_values, p_values)
    b_edges = np.r_[b_values[0] - (b_values[1] - b_values[0]) / 2, (b_values[:-1] + b_values[1:]) / 2, b_values[-1] + (b_values[-1] - b_values[-2]) / 2]
    p_edges = np.r_[p_values[0] - (p_values[1] - p_values[0]) / 2, (p_values[:-1] + p_values[1:]) / 2, p_values[-1] + (p_values[-1] - p_values[-2]) / 2]

    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    vmax = max(0.20, float(phase["stress_shortage_severity"].max()))
    image = ax.pcolormesh(b_edges, p_edges, pivot.to_numpy(), cmap="Blues", vmin=0.0, vmax=vmax, shading="flat")
    handles: list[Line2D] = []
    if float(pivot.to_numpy().min()) <= 0.05 <= float(pivot.to_numpy().max()):
        ax.contour(b_grid, p_grid, pivot.to_numpy(), levels=[0.05], colors="#111111", linestyles="--", linewidths=0.72)
        handles.append(Line2D([0], [0], color="#111111", linestyle="--", linewidth=0.72, label="0.05 boundary"))
    if "is_final_workpoint" in phase.columns:
        marked = phase[phase["is_final_workpoint"].astype(bool)]
        if not marked.empty:
            row = marked.iloc[0]
            ax.scatter(
                [row["B_EM_multiplier"]],
                [row["P_H_multiplier"]],
                marker="*",
                s=58,
                color=SCHEME_A["ADT-MAC"],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            handles.append(Line2D([0], [0], marker="*", markersize=6.5, markerfacecolor=SCHEME_A["ADT-MAC"], markeredgecolor="white", color="none", label="final workpoint"))
    ax.set_xlabel(r"$B_{\rm EM}$ multiplier")
    ax.set_ylabel(r"$P_H$ multiplier")
    ax.set_xticks(b_values)
    ax.set_yticks(p_values)
    ax.tick_params(axis="x", rotation=0)
    style_axes(ax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    cbar.set_label("Shortage severity")
    cbar.ax.tick_params(labelsize=BODY_TEXT_SIZE)
    if handles:
        legend(ax, handles=handles, fontsize=BODY_TEXT_SIZE, loc="upper right")
    save_figure(fig, out_dir, "fig7_phase_diagram_implant_shortage", pad=0.22)
    copy_source_csv(out_dir, "fig7_phase_diagram_implant_shortage")


def select_robustness_case(frame: pd.DataFrame, model: str, param_name: str, param_value: float | str) -> pd.Series:
    mask = frame["model"].eq(model) & frame["param_name"].eq(param_name)
    if isinstance(param_value, str):
        mask &= frame["param_value"].astype(str).eq(param_value)
    else:
        numeric = pd.to_numeric(frame["param_value"], errors="coerce")
        mask &= np.isclose(numeric, float(param_value))
    selected = frame[mask]
    if selected.empty:
        raise ValueError(f"Missing robustness case: {model} {param_name}={param_value}")
    return selected.iloc[0]


def render_fig8(out_dir: Path) -> None:
    summary = pd.read_csv(ROBUSTNESS_SUMMARY).sort_values("p_err")
    leaderboard = pd.read_csv(ROBUSTNESS_LEADERBOARD)
    leaderboard["param_value_numeric"] = pd.to_numeric(leaderboard["param_value"], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.92))
    blue = SCHEME_A["Proposed"]
    red = SCHEME_A["ADT-MAC"]
    orange = SCHEME_A["Surface nodes"]

    x_a = summary["p_err"]
    y_a = summary["stress_affected_margin_min"]
    err_a = summary["stress_affected_margin_min_ci95"]
    axes[0].errorbar(x_a, y_a, yerr=err_a, color=blue, marker="o", capsize=2.0, linewidth=0.95, markersize=3.4)
    axes[0].axvline(0.20, color=red, linestyle=(0, (2, 3)), linewidth=0.72)
    final_a = summary[np.isclose(summary["p_err"], 0.20)].iloc[0]
    axes[0].scatter([0.20], [final_a["stress_affected_margin_min"]], s=22, color=red, zorder=4)
    axes[0].text(0.58, FIG8_A_ANNOTATION_AXY, "severity = 0\nat 20%", transform=axes[0].transAxes, color=red, fontsize=BODY_TEXT_SIZE, ha="left", va="top")
    axes[0].set_ylim(float((y_a - err_a).min()) - 0.003, FIG8_A_YMAX)
    axes[0].set_xlim(-0.01, 0.21)
    axes[0].set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])
    axes[0].set_ylabel("Margin min")
    axes[0].set_xlabel(r"$p_{\rm err}$ (label-error rate)")
    set_panel_title(axes[0], "a", "Random label-noise sanity check")
    style_axes(axes[0])

    boundary = fig8_stress_fn_coarse_sweep(leaderboard)
    x_b = boundary["param_value_numeric"]
    y_b = boundary["stress_affected_margin_min"]
    err_b = boundary["stress_affected_margin_min_ci95"]
    axes[1].errorbar(x_b, y_b, yerr=err_b, color=blue, marker="o", capsize=2.0, linewidth=0.95, markersize=3.4)
    ax_b_right = axes[1].twinx()
    ax_b_right.plot(x_b, boundary["stress_false_negative_rate"], color=red, marker="s", linewidth=0.92, markersize=3.1)
    axes[1].text(
        FIG8_B_MARGIN_LABEL_X,
        FIG8_B_MARGIN_LABEL_Y,
        "Margin min",
        transform=axes[1].transAxes,
        color=blue,
        fontsize=BODY_TEXT_SIZE,
        ha="left",
        va="bottom",
    )
    ax_b_right.text(
        FIG8_B_FN_LABEL_X,
        FIG8_B_FN_LABEL_Y,
        "Observed FN",
        transform=ax_b_right.transAxes,
        color=red,
        fontsize=BODY_TEXT_SIZE,
        ha="left",
        va="center",
    )
    axes[1].set_ylim(*fig8_margin_axis_limits(y_b, err_b, min_span=0.018))
    fn_err = boundary["stress_false_negative_rate_ci95"] if "stress_false_negative_rate_ci95" in boundary.columns else None
    _, fn_upper = fig8_margin_axis_limits(boundary["stress_false_negative_rate"], fn_err, min_span=0.05)
    ax_b_right.set_ylim(0.0, max(0.22, fn_upper))
    axes[1].set_xlim(-0.015, 0.515)
    axes[1].set_ylabel("Margin min")
    axes[1].set_xlabel(r"$p_{\rm FN}$ (stress-FN rate)")
    ax_b_right.set_ylabel("Observed FN", color=red)
    ax_b_right.tick_params(direction="in", width=0.55, length=2.4, labelsize=BODY_TEXT_SIZE, colors=red)
    ax_b_right.spines["right"].set_color(red)
    ax_b_right.grid(False)
    set_panel_title(axes[1], "b", "Stress false-negative boundary")
    style_axes(axes[1])

    cases = [
        ("random_symmetric", "p_err", 0.2, "Random\n20%"),
        ("bursty_false_negative", "burst_len", 50.0, "Burst\n50"),
        ("estimator_delay", "delay", 80.0, "Delay\n80"),
        ("stuck_at_rest", "ratio", 0.5, "Stuck\n0.5"),
        ("mixed_realistic", "mixed_id", "mixed_boundary", "Mixed"),
    ]
    selected = [select_robustness_case(leaderboard, model, param, value) for model, param, value, _ in cases]
    labels = [label for _, _, _, label in cases]
    severity = np.asarray([float(row["stress_shortage_severity"]) for row in selected])
    margin = np.asarray([float(row["stress_affected_margin_min"]) for row in selected])
    scaled_severity, severity_scale_label, severity_ymax = fig8_shortage_axis_values(severity)
    xpos = np.arange(len(cases))
    colors = [blue, SCHEME_A["w/o crit.-urg."], red, orange, SCHEME_A["w/o implant-aware"]]
    axes[2].bar(xpos, scaled_severity, width=0.42, color=colors, edgecolor="#333333", linewidth=0.30, zorder=3)
    nonzero_severity = np.abs(severity) > 0
    scale_factor = float(np.median(scaled_severity[nonzero_severity] / severity[nonzero_severity])) if np.any(nonzero_severity) else 1.0
    if severity_scale_label:
        threshold_scaled = FIG8_C_SHORTAGE_THRESHOLD * scale_factor
    else:
        threshold_scaled = FIG8_C_SHORTAGE_THRESHOLD
    if np.isfinite(threshold_scaled) and threshold_scaled <= severity_ymax:
        axes[2].axhline(threshold_scaled, color=red, linestyle=(0, (1.5, 2.5)), linewidth=0.72)
    for x, y, m in zip(xpos, scaled_severity, margin):
        axes[2].text(x, y + severity_ymax * 0.032, f"m={m:.3f}", ha="center", va="bottom", fontsize=BODY_TEXT_SIZE)
    axes[2].set_ylim(0.0, severity_ymax)
    axes[2].set_xticks(xpos, labels)
    ylabel = f"Shortage severity ({severity_scale_label})" if severity_scale_label else "Shortage severity"
    axes[2].set_ylabel(ylabel)
    set_panel_title(axes[2], "c", "Representative estimator failures")
    style_axes(axes[2])

    fig.align_ylabels(axes)
    save_figure(fig, out_dir, "fig8_state_estimator_robustness", pad=0.24, svg=True)


def render_hfss_proxies(out_dir: Path) -> None:
    rows = pd.read_csv(HFSS_LIBRARY)
    rows = rows[rows["sched_include_main"].astype(str).str.lower().eq("true")]
    by_label = rows.set_index("label")
    missing = [label for label in HFSS_ORDER if label not in by_label.index]
    if missing:
        raise ValueError(f"Missing HFSS labels: {missing}")
    ordered = by_label.loc[HFSS_ORDER].reset_index()
    x = np.arange(len(ordered))
    colors = [CASE_COLORS[label] for label in ordered["label"]]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 2.88), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    bar_w = 0.33
    ax_a.bar(
        x - bar_w / 2,
        ordered["path_loss_energy_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        alpha=0.94,
        zorder=3,
    )
    ax_a.bar(
        x + bar_w / 2,
        ordered["path_loss_data_db"],
        width=bar_w,
        color=colors,
        edgecolor="#333333",
        linewidth=0.30,
        hatch="///",
        alpha=0.58,
        zorder=3,
    )
    set_panel_title(ax_a, "a", r"Path loss at $f_E$ and $f_D$")
    ax_a.set_ylabel("Path loss (dB)")
    ax_a.set_ylim(0, max(float(ordered["path_loss_energy_db"].max()), float(ordered["path_loss_data_db"].max())) * HFSS_TOP_HEADROOM)
    style_axes(ax_a)
    legend(
        ax_a,
        handles=[
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, label=r"$f_E$ 13.56 MHz"),
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, hatch="///", alpha=0.58, label=r"$f_D$ 40 MHz"),
        ],
        loc="upper left",
        ncol=2,
    )

    ax_cap = ax_b.twinx()
    ax_b.bar(x, ordered["chi_norm"], width=0.52, color=colors, edgecolor="#333333", linewidth=0.30, alpha=0.86, zorder=3)
    ax_b.axhline(1.0, color="#444444", linewidth=0.70, linestyle=":", alpha=0.78, zorder=2)
    ax_cap.semilogy(
        x,
        ordered["p_rx_cap_norm"],
        marker="D",
        linestyle="--",
        color="#222222",
        markersize=3.4,
        linewidth=0.92,
        markerfacecolor="white",
        markeredgewidth=0.72,
        zorder=5,
    )
    set_panel_title(ax_b, "b", "Local-field burden and Rx-power cap")
    ax_b.set_ylabel(r"$\chi_{\rm norm}$")
    ax_cap.set_ylabel(r"$p^{\rm norm}_{\rm rx,cap}$")
    ax_b.set_ylim(0, max(1.15, float(ordered["chi_norm"].max()) * HFSS_TOP_HEADROOM))
    ax_cap.set_ylim(0.05, 1.25)
    ax_cap.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.25, 0.5, 1.0]))
    ax_cap.yaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.25", "0.5", "1"]))
    ax_cap.yaxis.set_minor_locator(FixedLocator([]))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(HFSS_LABELS, linespacing=1.0)
    ax_b.tick_params(axis="x", labelsize=BODY_TEXT_SIZE, pad=1)
    ax_cap.tick_params(direction="in", width=0.55, length=2.4, labelsize=BODY_TEXT_SIZE)
    style_axes(ax_b)
    ax_cap.grid(False)
    legend(
        ax_b,
        handles=[
            Patch(facecolor="#9DB8CE", edgecolor="#333333", linewidth=0.30, label=r"$\chi_{\rm norm}$"),
            Line2D([0], [0], marker="D", color="#222222", linestyle="--", markersize=3.4, markerfacecolor="white", label=r"$p^{\rm norm}_{\rm rx,cap}$"),
        ],
        loc="upper right",
        ncol=2,
    )
    fig.align_ylabels([ax_a, ax_b])
    save_figure(fig, out_dir, "fig5_path_loss_and_em_proxies", pad=0.18)


def render_hfss_model(out_dir: Path) -> None:
    if not HFSS_MODEL_BASE.exists():
        source = LEGACY_FIGURE_DIR / "fig2_hfss_model.png"
        if not source.exists():
            raise FileNotFoundError(f"Missing legacy HFSS model figure: {source}")
        shutil.copy2(source, HFSS_MODEL_BASE)
    legacy_path = HFSS_MODEL_BASE
    if not legacy_path.exists():
        raise FileNotFoundError(f"Missing legacy HFSS model figure: {legacy_path}")
    image = plt.imread(legacy_path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(3.5, 3.5 * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    overlay_legacy_panel_titles(ax)
    save_figure(fig, out_dir, "fig2_hfss_model", tight=False)


def overlay_legacy_panel_titles(ax: plt.Axes) -> None:
    masks = [
        (0.250, 0.945, 0.560, 0.050),
        (0.000, 0.944, 0.090, 0.050),
        (0.170, 0.455, 0.700, 0.045),
        (0.000, 0.455, 0.090, 0.045),
    ]
    for x, y, w, h in masks:
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="white", edgecolor="none", zorder=3))
    title_box = {"facecolor": "white", "edgecolor": "none", "pad": 1.4}
    ax.text(
        0.5,
        0.972,
        panel_title("a", "HFSS tissue-block model"),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        bbox=title_box,
        zorder=4,
    )
    ax.text(
        0.5,
        0.474,
        panel_title("b", "Cross-section and receiver depths"),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        bbox=title_box,
        zorder=4,
    )


def write_manifest(out_dir: Path) -> None:
    lines = [
        "# main_figure_redesign_manifest",
        "",
        f"- created: {datetime.now().isoformat(timespec='seconds')}",
        "- scope: main manuscript figures only; supplementary figures unchanged",
        "- style: IEEE/IOTJ restrained single-column; Times-family typography",
        "- typography: centered 9 pt multi-panel titles; approximately 8 pt plot text",
        "- panel titles: only on multi-panel figures",
        "- legends: frameless",
        "- Fig.1/HFSS model: legacy HFSS model PNG used as the base with overlaid centered panel titles",
        "- palette: Scheme A role hierarchy",
        "",
        "## Outputs",
        "- fig2_hfss_model.pdf/png",
        "- fig5_path_loss_and_em_proxies.pdf/png",
        "- fig3_main_comparison.pdf/png",
        "- fig4_condition_switching_response.pdf/png",
        "- fig5_implant_stress_response.pdf/png",
        "- fig6_ablation_hfss.pdf/png",
        "- fig7_phase_diagram_implant_shortage.pdf/png",
        "- fig8_state_estimator_robustness.pdf/png/svg",
        "",
        "Fig.8 is rendered by the Python/matplotlib main-figure renderer for visual consistency.",
    ]
    (out_dir / "redesign_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all(out_dir: Path | None = None, *, timestamp: str | None = None) -> Path:
    apply_style()
    target = out_dir or default_output_dir(timestamp)
    target.mkdir(parents=True, exist_ok=True)
    render_hfss_model(target)
    render_hfss_proxies(target)
    render_bar_grid(
        target,
        "fig3_main_comparison",
        MAIN_METHODS,
        [
            ("stress_shortage_severity", "Implant shortage"),
            ("surface_served_ratio", "Surface service"),
            ("p95_backlog", "p95 backlog"),
            ("recovery_time", "Recovery time"),
        ],
    )
    render_fig4(target)
    render_fig5(target)
    render_bar_grid(
        target,
        "fig6_ablation_hfss",
        ABLATION_METHODS,
        [
            ("stress_shortage_severity", "Implant shortage"),
            ("implant_energy_p05", "Energy p05"),
            ("em_violation_rate", "EM violation"),
            ("served_workload_ratio", "Served ratio"),
        ],
    )
    render_fig7(target)
    render_fig8(target)
    write_manifest(target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render main manuscript figures using the main-figure standard.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to the configured figure directory.")
    parser.add_argument("--timestamp", default=None, help="Optional timestamp suffix for a preview figures_standard_<timestamp> directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = render_all(args.out_dir, timestamp=args.timestamp)
    print(out_dir)


if __name__ == "__main__":
    main()
