from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfss_paper_results.config import ABLATION_METHODS, MAIN_METHODS, METHOD_COLORS, METHOD_LINESTYLES, STRESS_METHODS, SimConfig


SCHEME_A_METHOD_COLORS = {
    "Proposed": "#1B4F8A",
    "ADT-MAC": "#B03A2E",
    "Lyap.-DPP": "#1E8449",
    "w/o EM-bud.": "#6FA3D0",
    "w/o crit.-urg.": "#9DB8CE",
    "w/o implant-aware": "#7F7F7F",
    "Oracle": "#333333",
}

SCHEME_A_LINESTYLES = {
    "Proposed": "-",
    "ADT-MAC": "--",
    "Lyap.-DPP": "-.",
    "w/o EM-bud.": ":",
    "w/o crit.-urg.": (0, (5, 2)),
    "w/o implant-aware": (0, (3, 1, 1, 1)),
    "Oracle": "-",
}

REDESIGN_FIGURE_STEMS = [
    "fig3_main_comparison",
    "fig4_condition_switching_response",
    "fig5_implant_stress_response",
    "fig6_ablation_hfss",
    "fig7_phase_diagram_implant_shortage",
]

METHOD_SHORT_LABELS = {
    "Proposed": "Proposed",
    "ADT-MAC": "ADT-MAC",
    "Lyap.-DPP": "Lyap.-DPP",
    "w/o EM-bud.": "w/o EM\nbudget",
    "w/o crit.-urg.": "w/o crit.-urg.",
    "w/o implant-aware": "w/o implant\naware",
}

BODY_TEXT_SIZE = 8
PANEL_TITLE_SIZE = 9
REGIME_LABEL_Y = 0.90
REGIME_LABEL_COLOR = "#111111"
FIG4_ALLOCATION_YMAX = 0.24
FIG4A_LEGEND_LINE_X0 = 340
FIG4A_LEGEND_LINE_X1 = 380
FIG4A_LEGEND_TEXT_X = 395
FIG4A_LEGEND_CRITICAL_Y = 0.118
FIG4A_LEGEND_SURFACE_Y = 0.088
FIG5_SHORTAGE_YMAX = 1.22
FIG5_STRESS_WINDOW_LABEL_Y = 1.14

FIG4_TITLES = {
    "allocation": "Proposed energy-support allocation by node class",
    "em": "Proposed EM-budget usage",
    "margin": "Critical implant energy margin",
}

FIG6_PANELS = [
    ("stress_shortage_severity", "Stress-window implant shortage severity"),
    ("implant_energy_p05", "Implant energy p05"),
    ("em_violation_rate", "EM violation rate"),
    ("served_workload_ratio", "Served workload ratio"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, final_csv: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    plt.close(fig)
    if final_csv is not None:
        final_csv.to_csv(out_dir / f"{stem}.csv", index=False)


def apply_ieee_scheme_a_style(style_path: Path | None = None) -> None:
    if style_path is None:
        package_path = Path(__file__).resolve()
        candidates = [
            package_path.parents[2] / "ieee_trans.mplstyle",
            package_path.parents[1] / "ieee_trans.mplstyle",
        ]
        style_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not style_path.exists():
        raise FileNotFoundError(f"Missing Matplotlib style file: {style_path}")
    plt.style.use(str(style_path))
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
            "grid.alpha": 0.45,
            "grid.linewidth": 0.35,
            "grid.color": "#D6DCE2",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def bar_values_and_ci95(frame: pd.DataFrame, metric: str, methods: list[str]) -> tuple[list[float], list[float] | None]:
    values = [float(frame.loc[frame["method"].eq(method), metric].iloc[0]) for method in methods]
    ci_col = f"{metric}_ci95"
    if ci_col not in frame.columns:
        return values, None
    ci95 = [float(frame.loc[frame["method"].eq(method), ci_col].fillna(0.0).iloc[0]) for method in methods]
    return values, ci95


def _bar(ax, frame: pd.DataFrame, metric: str, title: str, methods: list[str]) -> None:
    vals, ci95 = bar_values_and_ci95(frame, metric, methods)
    ax.bar(
        range(len(methods)),
        vals,
        yerr=ci95,
        capsize=3 if ci95 is not None else 0,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": "#222222"},
        color=[METHOD_COLORS[m] for m in methods],
        edgecolor="#333333",
        linewidth=0.45,
    )
    ax.set_xticks(range(len(methods)), methods, rotation=18, ha="right")
    ax.set_title(title)
    ax.margins(y=0.14)


def quick_fig3(metrics: dict[str, dict[str, float]], out_dir: Path) -> None:
    apply_style()
    rows = [{"method": method, **metrics[method]} for method in MAIN_METHODS if method in metrics]
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for ax, metric, title in [
        (axes[0, 0], "stress_shortage_severity", "Stress shortage severity"),
        (axes[0, 1], "surface_served_ratio", "Surface served ratio"),
        (axes[1, 0], "p95_backlog", "p95 backlog"),
        (axes[1, 1], "recovery_time", "Recovery time"),
    ]:
        _bar(ax, frame, metric, title, MAIN_METHODS)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "quick_fig3.png", dpi=220)
    plt.close(fig)


def quick_fig5(stress_raw: pd.DataFrame, cfg: SimConfig, out_dir: Path) -> None:
    apply_style()
    affected = stress_raw[stress_raw["node"].isin(cfg.stress_nodes)]
    data = affected.groupby(["method", "t"], as_index=False).agg(e_next=("e_next", "mean"), xi=("xi", "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 3.6), sharex=True)
    for ax in axes:
        ax.axvspan(cfg.stress_start, cfg.stress_end, color="#888888", alpha=0.16, linewidth=0)
    for method in [m for m in STRESS_METHODS if m in set(data["method"])]:
        sub = data[data["method"].eq(method)]
        axes[0].plot(sub["t"], sub["e_next"].rolling(8, min_periods=1).mean(), color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], label=method)
        axes[1].plot(sub["t"], sub["xi"], color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], label=method)
    axes[0].axhline(cfg.E_min_critical, color="#333333", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Energy")
    axes[1].set_ylabel("Shortfall")
    axes[1].set_xlabel("Slot")
    axes[0].legend(fontsize=6.5, ncol=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "quick_fig5.png", dpi=220)
    plt.close(fig)


def plot_fig3(metrics: dict[str, dict[str, float]], final_dir: Path) -> None:
    apply_style()
    frame = pd.DataFrame([{"method": method, **metrics[method]} for method in MAIN_METHODS])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    panels = [
        ("stress_shortage_severity", "Stress-window implant shortage severity"),
        ("surface_served_ratio", "Surface served workload ratio"),
        ("p95_backlog", "p95 backlog"),
        ("recovery_time", "Recovery time (slots)"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        _bar(ax, frame, metric, title, MAIN_METHODS)
    _save(fig, final_dir, "fig3_main_comparison", final_csv=frame)


def build_fig4_mechanism_frames(condition_raw: pd.DataFrame, cfg: SimConfig) -> dict[str, pd.DataFrame]:
    raw = condition_raw.copy()
    raw["margin"] = np.where(raw["is_critical"], (raw["e_next"] - raw["E_min"]) / (raw["E_max"] - raw["E_min"]), np.nan)
    grouped = raw.groupby(["method", "t", "node_type", "is_critical", "regime"], as_index=False).agg(
        uE=("uE", "mean"),
        em_utilization=("em_utilization", "first"),
        margin=("margin", "mean"),
    )

    proposed = grouped[grouped["method"].eq("Proposed")]
    allocation_parts = []
    crit = proposed[proposed["is_critical"].astype(bool)].groupby(["t", "regime"], as_index=False)["uE"].mean()
    crit["curve"] = "Critical implant nodes"
    surf = proposed[proposed["node_type"].eq("surface")].groupby(["t", "regime"], as_index=False)["uE"].mean()
    surf["curve"] = "Surface nodes"
    allocation_parts.extend([crit, surf])
    allocation = pd.concat(allocation_parts, ignore_index=True).rename(columns={"uE": "value"})
    allocation["method"] = "Proposed"
    allocation["panel"] = "allocation"

    em = raw[raw["method"].eq("Proposed")].groupby(["method", "t", "regime"], as_index=False)["em_utilization"].first()
    em = em.rename(columns={"em_utilization": "value"})
    em["curve"] = "Proposed raw EM utilization"
    em["panel"] = "em"

    budget = em[["t", "regime"]].copy()
    budget["method"] = "Budget limit = 1"
    budget["value"] = 1.0
    budget["curve"] = "Budget limit = 1"
    budget["panel"] = "em"
    em = pd.concat([em, budget], ignore_index=True, sort=False)

    margin = grouped[grouped["is_critical"].astype(bool)].groupby(["method", "t", "regime"], as_index=False)["margin"].mean()
    margin = margin[margin["method"].isin(["Proposed", "w/o crit.-urg."])].rename(columns={"margin": "value"})
    margin["curve"] = margin["method"]
    margin["panel"] = "margin"

    final_csv = pd.concat(
        [
            allocation[["panel", "method", "curve", "t", "regime", "value"]],
            em[["panel", "method", "curve", "t", "regime", "value"]],
            margin[["panel", "method", "curve", "t", "regime", "value"]],
        ],
        ignore_index=True,
    )
    return {"allocation": allocation, "em": em, "margin": margin, "csv": final_csv}


def plot_fig4(condition_raw: pd.DataFrame, cfg: SimConfig, final_dir: Path) -> None:
    apply_style()
    frames = build_fig4_mechanism_frames(condition_raw, cfg)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)
    bands = [(0, 300, "rest"), (300, 600, "surface_sweat"), (600, 900, "surface_moderate_loose"), (900, 1200, "recovery_rest")]
    for ax in axes:
        for start, end, label in bands:
            ax.axvspan(start, end, color="#EFEFEF" if "rest" in label else "#F6E7D8", alpha=0.60, linewidth=0)
            ax.text((start + end) / 2, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.5, color="#444444")
        for boundary in [300, 600, 900]:
            ax.axvline(boundary, color="#777777", linestyle="--", linewidth=0.5)

    allocation = frames["allocation"]
    for curve, color in [("Critical implant nodes", "#1F4E79"), ("Surface nodes", "#E67E22")]:
        sub = allocation[allocation["curve"].eq(curve)]
        axes[0].plot(sub["t"], sub["value"].rolling(18, min_periods=1).mean(), color=color, label=curve)

    em = frames["em"]
    proposed_em = em[em["method"].eq("Proposed")]
    axes[1].plot(
        proposed_em["t"],
        proposed_em["value"].rolling(18, min_periods=1).mean(),
        color=METHOD_COLORS["Proposed"],
        linestyle=METHOD_LINESTYLES["Proposed"],
        label="Proposed raw EM utilization",
    )
    axes[1].axhline(1.0, color="#333333", linestyle=":", linewidth=0.8, label="Budget limit = 1")

    margin = frames["margin"]
    for method in ["Proposed", "w/o crit.-urg."]:
        sub = margin[margin["method"].eq(method)]
        axes[2].plot(sub["t"], sub["value"].rolling(18, min_periods=1).mean(), color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], label=method)
    axes[2].axhline(0.0, color="#333333", linestyle=":", linewidth=0.8)
    axes[0].set_title(FIG4_TITLES["allocation"], fontsize=9, pad=5)
    axes[1].set_title(FIG4_TITLES["em"], fontsize=9, pad=5)
    axes[2].set_title(FIG4_TITLES["margin"], fontsize=9, pad=5)
    axes[0].set_ylabel("Mean energy-support fraction $u_i^E$")
    axes[1].set_ylabel("Raw EM utilization")
    axes[2].set_ylabel("Critical implant energy margin")
    axes[2].set_xlabel("Slot")
    for ax in axes:
        ax.legend(fontsize=6.2, loc="upper right")
    _save(fig, final_dir, "fig4_condition_switching_response", final_csv=frames["csv"])


def plot_fig5(stress_raw: pd.DataFrame, cfg: SimConfig, final_dir: Path) -> None:
    apply_style()
    affected = stress_raw[stress_raw["node"].isin(cfg.stress_nodes)]
    csv = affected.groupby(["method", "t"], as_index=False).agg(e_next=("e_next", "mean"), xi=("xi", "mean"), severity=("xi", lambda s: float((s / cfg.E_min_critical).mean())))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    for ax in axes:
        ax.axvspan(cfg.stress_start, cfg.stress_end, color="#888888", alpha=0.18, linewidth=0)
        ax.axvline(cfg.stress_start, color="#777777", linestyle="--", linewidth=0.5)
        ax.axvline(cfg.stress_end, color="#777777", linestyle="--", linewidth=0.5)
    for method in STRESS_METHODS:
        sub = csv[csv["method"].eq(method)]
        axes[0].plot(sub["t"], sub["e_next"].rolling(10, min_periods=1).mean(), color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], label=method)
        axes[1].plot(sub["t"], sub["severity"].rolling(4, min_periods=1).mean(), color=METHOD_COLORS[method], linestyle=METHOD_LINESTYLES[method], label=method)
    axes[0].axhline(cfg.E_min_critical, color="#333333", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Stressed implant energy")
    axes[1].set_ylabel("Shortage severity")
    axes[1].set_xlabel("Slot")
    axes[0].legend(fontsize=6.0, ncol=2)
    _save(fig, final_dir, "fig5_implant_stress_response", final_csv=csv)


def plot_fig6(metrics: dict[str, dict[str, float]], final_dir: Path) -> None:
    apply_style()
    frame = pd.DataFrame([{"method": method, **metrics[method]} for method in ABLATION_METHODS])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for ax, (metric, title) in zip(axes.ravel(), FIG6_PANELS):
        _bar(ax, frame, metric, title, ABLATION_METHODS)
    _save(fig, final_dir, "fig6_ablation_hfss", final_csv=frame)


def plot_fig7(phase: pd.DataFrame, final_dir: Path) -> None:
    apply_style()
    pivot = phase.pivot(index="P_H_multiplier", columns="B_EM_multiplier", values="stress_shortage_severity").sort_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.20, float(phase["stress_shortage_severity"].max())))
    ax.set_xticks(range(len(pivot.columns)), [f"{v:.2g}" for v in pivot.columns], rotation=25)
    ax.set_yticks(range(len(pivot.index)), [f"{v:.2g}" for v in pivot.index])
    ax.set_xlabel("$B_{EM}$ multiplier")
    ax.set_ylabel("$P_H$ multiplier")
    if {"is_final_workpoint"}.issubset(phase.columns):
        marked = phase[phase["is_final_workpoint"].astype(bool)]
        if not marked.empty:
            row = marked.iloc[0]
            x = list(pivot.columns).index(row["B_EM_multiplier"])
            y = list(pivot.index).index(row["P_H_multiplier"])
            ax.scatter([x], [y], marker="*", s=90, color="#B03A2E", edgecolor="white", linewidth=0.6)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Stress shortage severity")
    _save(fig, final_dir, "fig7_phase_diagram_implant_shortage", final_csv=phase)


def _save_redesign(fig: plt.Figure, out_dir: Path, stem: str, *, tight_rect: tuple[float, float, float, float] | None = None) -> None:
    fig.tight_layout(pad=0.22, rect=tight_rect)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=600)
    plt.close(fig)


def _panel_title(ax: plt.Axes, label: str, title: str, *, pad: float = 3.0) -> None:
    ax.set_title(f"{label}. {title}", fontsize=PANEL_TITLE_SIZE, fontweight="bold", loc="center", pad=pad)


def _scheme_bar(ax: plt.Axes, frame: pd.DataFrame, metric: str, methods: list[str]) -> None:
    vals, ci95 = bar_values_and_ci95(frame, metric, methods)
    xpos = np.arange(len(methods))
    ax.bar(
        xpos,
        vals,
        yerr=ci95,
        width=0.62,
        capsize=2.2 if ci95 is not None else 0,
        error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": "#222222"},
        color=[SCHEME_A_METHOD_COLORS[m] for m in methods],
        edgecolor="#333333",
        linewidth=0.35,
    )
    zero_positions = [x for x, value in zip(xpos, vals) if abs(value) < 1e-12]
    if zero_positions:
        ax.plot(zero_positions, [0.0] * len(zero_positions), linestyle="None", marker="_", markersize=7, color="#222222", markeredgewidth=0.8)
    ax.set_xticks(xpos, [METHOD_SHORT_LABELS.get(method, method) for method in methods], rotation=0, ha="center")
    max_top = max([abs(v) for v in vals] + ([abs(v) for v in ci95] if ci95 is not None else [0.0]))
    ax.set_ylim(bottom=0.0, top=max_top * 1.22 if max_top > 0 else 1.0)
    ax.margins(x=0.08)


def _plot_redesigned_bar_grid(
    frame: pd.DataFrame,
    out_dir: Path,
    stem: str,
    methods: list[str],
    panels: list[tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.05))
    for index, (ax, (metric, title)) in enumerate(zip(axes.ravel(), panels)):
        _scheme_bar(ax, frame, metric, methods)
        _panel_title(ax, chr(ord("a") + index), title)
    _save_redesign(fig, out_dir, stem)


def _smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1, center=False).mean()


def _regime_label(label: object) -> str:
    text = str(label)
    return {
        "rest": "rest",
        "surface_sweat": "sweat",
        "surface_moderate_loose": "moderate loose",
        "recovery_rest": "recovery rest",
    }.get(text, text.replace("_", " "))


def _shade_regimes(ax: plt.Axes, frame: pd.DataFrame, *, show_labels: bool = False) -> None:
    if not {"t", "regime"}.issubset(frame.columns):
        return
    timeline = frame[["t", "regime"]].drop_duplicates().sort_values("t")
    if timeline.empty:
        return
    rows = timeline.to_dict(orient="records")
    max_t = float(timeline["t"].max())
    starts: list[tuple[float, float, object]] = []
    current_start = float(rows[0]["t"])
    current_label = rows[0]["regime"]
    for prev, row in zip(rows, rows[1:]):
        if row["regime"] != current_label:
            starts.append((current_start, float(row["t"]), current_label))
            current_start = float(row["t"])
            current_label = row["regime"]
    step = float(np.median(np.diff(sorted(timeline["t"].unique())))) if len(timeline["t"].unique()) > 1 else 1.0
    starts.append((current_start, max_t + step, current_label))
    for start, end, label in starts:
        warm = "sweat" in str(label) or "loose" in str(label)
        ax.axvspan(start, end, color="#F4E6D7" if warm else "#EFEFEF", alpha=0.48, linewidth=0)
        if show_labels:
            ax.text(
                (start + end) / 2,
                REGIME_LABEL_Y,
                _regime_label(label),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=BODY_TEXT_SIZE,
                color=REGIME_LABEL_COLOR,
            )
    for _, end, _ in starts[:-1]:
        ax.axvline(end, color="#777777", linestyle="--", linewidth=0.45)


def _plot_redesigned_fig4(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.55), sharex=True)
    for index, ax in enumerate(axes):
        _shade_regimes(ax, frame, show_labels=index == 0)

    allocation = frame[frame["panel"].eq("allocation")]
    for curve, color, style in [
        ("Critical implant nodes", SCHEME_A_METHOD_COLORS["Proposed"], "-"),
        ("Surface nodes", "#E67E22", "-"),
    ]:
        sub = allocation[allocation["curve"].eq(curve)].sort_values("t")
        axes[0].plot(sub["t"], _smooth(sub["value"], 18), color=color, linestyle=style, linewidth=1.05, label=curve)
    axes[0].set_ylim(-0.005, FIG4_ALLOCATION_YMAX)

    em = frame[frame["panel"].eq("em")]
    proposed_em = em[em["method"].eq("Proposed")].sort_values("t")
    axes[1].plot(proposed_em["t"], _smooth(proposed_em["value"], 18), color=SCHEME_A_METHOD_COLORS["Proposed"], linewidth=1.05, label="Proposed")
    axes[1].axhline(1.0, color="#333333", linestyle=":", linewidth=0.75, label="Budget")

    margin = frame[frame["panel"].eq("margin")]
    for method in ["Proposed", "w/o crit.-urg."]:
        sub = margin[margin["method"].eq(method)].sort_values("t")
        axes[2].plot(sub["t"], _smooth(sub["value"], 18), color=SCHEME_A_METHOD_COLORS[method], linestyle=SCHEME_A_LINESTYLES[method], linewidth=1.05, label=METHOD_SHORT_LABELS[method])
    axes[2].axhline(0.0, color="#333333", linestyle=":", linewidth=0.75)

    _panel_title(axes[0], "a", "Energy-support allocation")
    _panel_title(axes[1], "b", "EM-budget usage")
    _panel_title(axes[2], "c", "Critical implant margin")
    axes[0].set_ylabel("$u_i^E$")
    axes[1].set_ylabel("EM util.")
    axes[2].set_ylabel("Margin")
    axes[2].set_xlabel("Slot")
    axes[0].plot(
        [FIG4A_LEGEND_LINE_X0, FIG4A_LEGEND_LINE_X1],
        [FIG4A_LEGEND_CRITICAL_Y, FIG4A_LEGEND_CRITICAL_Y],
        color=SCHEME_A_METHOD_COLORS["Proposed"],
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG4A_LEGEND_TEXT_X, FIG4A_LEGEND_CRITICAL_Y, "Critical implant nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    axes[0].plot(
        [FIG4A_LEGEND_LINE_X0, FIG4A_LEGEND_LINE_X1],
        [FIG4A_LEGEND_SURFACE_Y, FIG4A_LEGEND_SURFACE_Y],
        color="#E67E22",
        linewidth=0.95,
        solid_capstyle="butt",
    )
    axes[0].text(FIG4A_LEGEND_TEXT_X, FIG4A_LEGEND_SURFACE_Y, "Surface nodes", ha="left", va="center", fontsize=BODY_TEXT_SIZE, color="#111111")
    axes[1].legend(fontsize=BODY_TEXT_SIZE, loc="upper right", frameon=False)
    axes[2].legend(fontsize=BODY_TEXT_SIZE, loc="lower right", frameon=False)
    _save_redesign(fig, out_dir, "fig4_condition_switching_response")


def _stress_window_from_frame(frame: pd.DataFrame) -> tuple[float, float] | None:
    if "severity" not in frame.columns:
        return None
    by_t = frame.groupby("t")["severity"].max()
    active = by_t[by_t > 1e-12]
    if active.empty:
        return None
    values = sorted(active.index.astype(float))
    step = float(np.median(np.diff(sorted(by_t.index.astype(float))))) if len(by_t.index) > 1 else 1.0
    return values[0], values[-1] + step


def _plot_redesigned_fig5(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)
    stress_window = _stress_window_from_frame(frame)
    for index, ax in enumerate(axes):
        if stress_window is not None:
            start, end = stress_window
            ax.axvspan(start, end, color="#D8D8D8", alpha=0.58, linewidth=0)
            ax.axvline(start, color="#777777", linestyle="--", linewidth=0.45)
            ax.axvline(end, color="#777777", linestyle="--", linewidth=0.45)

    for method in STRESS_METHODS:
        sub = frame[frame["method"].eq(method)].sort_values("t")
        if sub.empty:
            continue
        axes[0].plot(sub["t"], _smooth(sub["e_next"], 10), color=SCHEME_A_METHOD_COLORS[method], linestyle=SCHEME_A_LINESTYLES[method], linewidth=1.05, label=METHOD_SHORT_LABELS.get(method, method))
        axes[1].plot(sub["t"], _smooth(sub["severity"], 4), color=SCHEME_A_METHOD_COLORS[method], linestyle=SCHEME_A_LINESTYLES[method], linewidth=1.05, label=METHOD_SHORT_LABELS.get(method, method))
    axes[0].axhline(0.35, color="#333333", linestyle=":", linewidth=0.75)
    axes[1].set_ylim(-0.05, FIG5_SHORTAGE_YMAX)
    if stress_window is not None:
        start, end = stress_window
        axes[1].text((start + end) / 2, FIG5_STRESS_WINDOW_LABEL_Y, "stress window", ha="center", va="center", fontsize=BODY_TEXT_SIZE, color=REGIME_LABEL_COLOR)
    _panel_title(axes[0], "a", "Stressed implant energy")
    _panel_title(axes[1], "b", "Normalized shortage")
    axes[0].set_ylabel("Energy")
    axes[1].set_ylabel("Shortage")
    axes[1].set_xlabel("Slot")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=BODY_TEXT_SIZE, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0, frameon=False)
    _save_redesign(fig, out_dir, "fig5_implant_stress_response", tight_rect=(0.0, 0.0, 1.0, 0.88))


def _plot_redesigned_fig7(phase: pd.DataFrame, out_dir: Path) -> None:
    pivot = phase.pivot(index="P_H_multiplier", columns="B_EM_multiplier", values="stress_shortage_severity").sort_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    vmax = max(0.20, float(phase["stress_shortage_severity"].max()))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)), [f"{v:.2g}" for v in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), [f"{v:.2g}" for v in pivot.index])
    ax.set_xlabel("$B_{EM}$ multiplier")
    ax.set_ylabel("$P_H$ multiplier")
    if "is_final_workpoint" in phase.columns:
        marked = phase[phase["is_final_workpoint"].astype(bool)]
        if not marked.empty:
            row = marked.iloc[0]
            x = list(pivot.columns).index(row["B_EM_multiplier"])
            y = list(pivot.index).index(row["P_H_multiplier"])
            ax.scatter([x], [y], marker="*", s=62, color=SCHEME_A_METHOD_COLORS["ADT-MAC"], edgecolor="white", linewidth=0.55, zorder=3)
    cbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    cbar.set_label("Shortage severity")
    _save_redesign(fig, out_dir, "fig7_phase_diagram_implant_shortage")


def _unique_redesign_dir(output_root: Path, timestamp: str) -> Path:
    base = output_root / f"final_ieee_scheme_a_{timestamp}"
    if not base.exists():
        return base
    for index in range(2, 100):
        candidate = output_root / f"final_ieee_scheme_a_{timestamp}_{index:02d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to create a unique redesign directory under {output_root}")


def _write_redesign_manifest(out_dir: Path, source_dir: Path, style_path: Path, timestamp: str) -> None:
    palette_lines = [f"- {method}: `{color}`, linestyle `{SCHEME_A_LINESTYLES.get(method)}`" for method, color in SCHEME_A_METHOD_COLORS.items()]
    lines = [
        "# redesign_manifest",
        "",
        f"- source directory: `{source_dir}`",
        f"- created: `{timestamp}`",
        "- palette: Scheme A / Role Hierarchy from `color_design.html`",
        f"- style file: `{style_path.name}`",
        "- original `outputs/final` files were not modified by this redesign mode",
        "",
        "## Palette",
        *palette_lines,
    ]
    (out_dir / "redesign_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def redesign_final_figures(source_dir: Path, *, output_root: Path | None = None, timestamp: str | None = None, style_path: Path | None = None) -> Path:
    source_dir = Path(source_dir)
    output_root = Path(output_root) if output_root is not None else source_dir.parent
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if style_path is None:
        package_path = Path(__file__).resolve()
        candidates = [
            package_path.parents[2] / "ieee_trans.mplstyle",
            package_path.parents[1] / "ieee_trans.mplstyle",
        ]
        style_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    missing = [source_dir / f"{stem}.csv" for stem in REDESIGN_FIGURE_STEMS if not (source_dir / f"{stem}.csv").exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing final CSV source(s): {missing_text}")

    out_dir = _unique_redesign_dir(output_root, timestamp)
    out_dir.mkdir(parents=True, exist_ok=False)
    for stem in REDESIGN_FIGURE_STEMS:
        shutil.copy2(source_dir / f"{stem}.csv", out_dir / f"{stem}.csv")

    apply_ieee_scheme_a_style(style_path)
    fig3 = pd.read_csv(source_dir / "fig3_main_comparison.csv")
    _plot_redesigned_bar_grid(
        fig3,
        out_dir,
        "fig3_main_comparison",
        MAIN_METHODS,
        [
            ("stress_shortage_severity", "Implant shortage"),
            ("surface_served_ratio", "Surface service"),
            ("p95_backlog", "p95 backlog"),
            ("recovery_time", "Recovery time"),
        ],
    )

    fig4 = pd.read_csv(source_dir / "fig4_condition_switching_response.csv")
    _plot_redesigned_fig4(fig4, out_dir)

    fig5 = pd.read_csv(source_dir / "fig5_implant_stress_response.csv")
    _plot_redesigned_fig5(fig5, out_dir)

    fig6 = pd.read_csv(source_dir / "fig6_ablation_hfss.csv")
    _plot_redesigned_bar_grid(
        fig6,
        out_dir,
        "fig6_ablation_hfss",
        ABLATION_METHODS,
        [
            ("stress_shortage_severity", "Implant shortage"),
            ("implant_energy_p05", "Energy p05"),
            ("em_violation_rate", "EM violation"),
            ("served_workload_ratio", "Served workload"),
        ],
    )

    fig7 = pd.read_csv(source_dir / "fig7_phase_diagram_implant_shortage.csv")
    _plot_redesigned_fig7(fig7, out_dir)
    _write_redesign_manifest(out_dir, source_dir, style_path, timestamp)
    return out_dir
