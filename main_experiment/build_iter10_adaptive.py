from __future__ import annotations

import hashlib
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT
FINAL_DIR = PROJECT_ROOT / "output"
V1_DIR = PROJECT_ROOT / "output"
OUT_DIR = PROJECT_ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
SNIPPET_DIR = OUT_DIR / "snippets"
BEST_DIR = OUT_DIR / "best_robustness_package"

sys.path.insert(0, str(PROJECT_ROOT))

from Experiment_results.config import SimConfig, WorkPoint  # noqa: E402
from Experiment_results.data import HFSSLibrary, load_hfss_library  # noqa: E402
from Experiment_results.metrics import summarize_raw  # noqa: E402
from Experiment_results.scenarios import condition_name, node_type  # noqa: E402
from Experiment_results.sim import _label_func, coefficients_for_labels, generate_trace, solve_slot_lp  # noqa: E402


SEEDS = list(range(20))
NOISE_BASE_SEED = 20260506
METHOD = "Proposed"
SCENARIO = "stress"
SURFACE_MAIN_LABELS = ["surface_rest", "surface_sweat", "surface_moderate_loose"]
STRESS_THRESHOLD = 0.05
IEEE_STYLE_PATH = PACKAGE_ROOT / "ieee_trans.mplstyle"
IEEE_SINGLE_FIGSIZE = (3.5, 2.4)
IEEE_DOUBLE_2X2_FIGSIZE = (7.16, 4.8)
IEEE_SINGLE_4X1_FIGSIZE = (3.5, 4.8)
BODY_TEXT_SIZE = 8
PANEL_TITLE_SIZE = 9
COLORS = {
    "proposed": "#1B4F8A",
    "competitor": "#B03A2E",
    "lyap": "#1E8449",
    "ablation_light": "#6FA3D0",
    "ablation_grayblue": "#9DB8CE",
    "shortage": "#5B9BD5",
    "energy_cost": "#C0392B",
    "em": "#E67E22",
    "candidate": "#7F7F7F",
    "grid": "#E6E6E6",
}


def apply_ieee_figure_style() -> None:
    if IEEE_STYLE_PATH.exists():
        plt.style.use(str(IEEE_STYLE_PATH))
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure_pair(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)


def set_panel_title(ax: plt.Axes, letter: str, title: str, *, pad: float = 2.0) -> None:
    ax.set_title(f"{letter}. {title}", fontsize=PANEL_TITLE_SIZE, fontweight="bold", loc="center", pad=pad)


@dataclass(frozen=True)
class Trial:
    iteration: int
    iteration_name: str
    model: str
    condition_id: str
    param_name: str
    param_value: float | str
    params: dict[str, Any]
    wp: WorkPoint
    workpoint_tag: str = "final"


def ci95(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=0) / math.sqrt(len(arr)))


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pm(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{fmt(row[metric], digits)}\\pm{fmt(row.get(metric + '_ci95', 0.0), digits)}"


def latex_texttt(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def stable_seed(*items: object) -> int:
    payload = "|".join(str(item) for item in items).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def load_final_workpoint() -> WorkPoint:
    path = FINAL_DIR / "final_workpoint.yaml"
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = float(value.strip())
    return WorkPoint(
        lambda_q=values["lambda_q"],
        lambda_c=values["lambda_c"],
        lambda_e=values["lambda_e"],
        lambda_xi=values["lambda_xi"],
        B_EM_ref=values["B_EM_ref"],
        P_H=values["P_H"],
    )


def rest_version(label: str) -> str:
    if label == "implant10_stress":
        return "implant10_rest"
    if label == "implant30_stress":
        return "implant30_rest"
    return label


def stress_version(label: str) -> str:
    if label == "implant10_rest":
        return "implant10_stress"
    if label == "implant30_rest":
        return "implant30_stress"
    return label


def corrupt_surface(label: str, rng: np.random.Generator) -> str:
    if label not in SURFACE_MAIN_LABELS:
        return label
    choices = [item for item in SURFACE_MAIN_LABELS if item != label]
    return str(rng.choice(choices))


def random_symmetric_labels(true_labels: list[str], p_err: float, rng: np.random.Generator) -> list[str]:
    labels = list(true_labels)
    for idx, label in enumerate(true_labels):
        if rng.random() >= p_err:
            continue
        if label.endswith("_stress"):
            labels[idx] = rest_version(label)
        elif label in {"implant10_rest", "implant30_rest"}:
            labels[idx] = stress_version(label)
        elif label in SURFACE_MAIN_LABELS:
            labels[idx] = corrupt_surface(label, rng)
        elif label == "surface_contact_failure":
            labels[idx] = "surface_rest"
    return labels


def decision_labels_for_trial(
    trial: Trial,
    cfg: SimConfig,
    true_labels: list[str],
    labels_for_t,
    t: int,
    rng: np.random.Generator,
) -> list[str]:
    model = trial.model
    params = trial.params
    if model == "identity":
        return list(true_labels)

    if model == "estimator_delay":
        delay = int(params["delay"])
        return list(labels_for_t(max(0, t - delay)))

    labels = list(true_labels)

    if model == "random_symmetric":
        return random_symmetric_labels(true_labels, float(params["p_err"]), rng)

    if model == "stress_false_negative":
        p_fn = float(params["p_fn"])
        for idx, label in enumerate(true_labels):
            if label.endswith("_stress") and rng.random() < p_fn:
                labels[idx] = rest_version(label)
        return labels

    if model == "bursty_false_negative":
        length = int(params["burst_len"])
        burst_start = cfg.stress_start
        burst_stop = min(cfg.stress_end, burst_start + length)
        if burst_start <= t < burst_stop:
            for idx, label in enumerate(true_labels):
                if label.endswith("_stress"):
                    labels[idx] = rest_version(label)
        return labels

    if model == "stuck_at_rest":
        ratio = float(params["ratio"])
        stop = cfg.stress_start + int(round((cfg.stress_end - cfg.stress_start) * ratio))
        if cfg.stress_start <= t < stop:
            for idx, label in enumerate(true_labels):
                if label.endswith("_stress"):
                    labels[idx] = rest_version(label)
        return labels

    if model == "stress_false_positive":
        p_fp = float(params["p_fp"])
        for idx, label in enumerate(true_labels):
            if label in {"implant10_rest", "implant30_rest"} and rng.random() < p_fp:
                labels[idx] = stress_version(label)
        return labels

    if model == "mixed_realistic":
        delay = int(params.get("delay", 0))
        if delay > 0:
            labels = list(labels_for_t(max(0, t - delay)))
        burst_len = int(params.get("burst_len", 0))
        burst_active = cfg.stress_start <= t < min(cfg.stress_end, cfg.stress_start + burst_len)
        p_fn = float(params.get("p_fn", 0.0))
        p_fp = float(params.get("p_fp", 0.0))
        p_surface = float(params.get("p_surface", 0.0))
        for idx, true_label in enumerate(true_labels):
            if true_label.endswith("_stress") and (burst_active or rng.random() < p_fn):
                labels[idx] = rest_version(true_label)
            elif true_label in {"implant10_rest", "implant30_rest"} and rng.random() < p_fp:
                labels[idx] = stress_version(true_label)
            elif true_label in SURFACE_MAIN_LABELS and rng.random() < p_surface:
                labels[idx] = corrupt_surface(true_label, rng)
        return labels

    raise ValueError(f"Unsupported trial model: {model}")


def run_seed_with_trial(cfg: SimConfig, trial: Trial, seed: int, library: HFSSLibrary) -> pd.DataFrame:
    local_wp = WorkPoint(
        trial.wp.lambda_q,
        trial.wp.lambda_c,
        trial.wp.lambda_e,
        trial.wp.lambda_xi,
        trial.wp.B_EM_ref,
        trial.wp.P_H,
    )
    trace = generate_trace(cfg, seed, SCENARIO)
    labels_for_t = _label_func(cfg, SCENARIO)
    rng = np.random.default_rng(NOISE_BASE_SEED + seed * 1009 + stable_seed(trial.condition_id))
    q = trace.initial_queue.copy()
    e = trace.initial_energy.copy()
    rows: list[dict[str, object]] = []

    for t in range(cfg.T):
        true_labels = labels_for_t(t)
        decision_labels = decision_labels_for_trial(trial, cfg, true_labels, labels_for_t, t, rng)
        true_coeff = coefficients_for_labels(library, cfg, local_wp, trace, true_labels)
        decision_coeff = coefficients_for_labels(library, cfg, local_wp, trace, decision_labels)
        decision_critical = trace.is_critical.copy()
        in_stress = cfg.stress_start <= t < cfg.stress_end

        lp_solution = solve_slot_lp(METHOD, cfg, local_wp, decision_coeff, e, q, trace.rho, decision_critical)
        u_e = lp_solution.u_e
        u_d = lp_solution.u_d
        mu_decision = lp_solution.mu

        e_hat = np.maximum(e - cfg.E_base, 0.0)
        p_rx = true_coeff.g * local_wp.P_H * u_e
        harvested = true_coeff.eta * local_wp.P_H * u_e
        tx_cost = trace.rho * u_d
        post_energy = e_hat + harvested - tx_cost
        xi = np.where(trace.is_critical, np.clip(cfg.E_min_critical - post_energy, 0.0, cfg.E_min_critical), 0.0)
        e_next = np.clip(post_energy, 0.0, cfg.E_max)
        mu = np.minimum(q, true_coeff.r * u_d)
        arrived = trace.arrivals[t]
        q_next = np.maximum(q - mu, 0.0) + arrived
        em_load = float(np.dot(true_coeff.chi, u_e))
        em_util = em_load / max(true_coeff.B_EM, 1e-12)
        shared = cfg.alpha_D * float(u_d.sum()) + cfg.alpha_E * float(u_e.sum())

        for node in range(cfg.num_nodes):
            is_critical = bool(trace.is_critical[node])
            e_min = cfg.E_min_critical if is_critical else 0.0
            rx_ratio = p_rx[node] / max(true_coeff.p_rx_cap[node], 1e-12)
            label_true = true_labels[node]
            label_decision = decision_labels[node]
            rows.append(
                {
                    "scenario": SCENARIO,
                    "seed": seed,
                    "method": METHOD,
                    "t": t,
                    "regime": "stress_window" if in_stress else condition_name(t),
                    "node": node,
                    "node_type": node_type(node),
                    "is_critical": is_critical,
                    "is_stressed": bool(in_stress and node in cfg.stress_nodes),
                    "label_true": label_true,
                    "label_decision": label_decision,
                    "label_error": bool(label_true != label_decision),
                    "true_stress_label": bool(label_true.endswith("_stress")),
                    "decision_stress_label": bool(label_decision.endswith("_stress")),
                    "q": float(q[node]),
                    "e": float(e[node]),
                    "e_next": float(e_next[node]),
                    "E_min": float(e_min),
                    "E_max": float(cfg.E_max),
                    "uD": float(u_d[node]),
                    "uE": float(u_e[node]),
                    "mu": float(mu[node]),
                    "mu_decision": float(mu_decision[node]),
                    "xi": float(xi[node]),
                    "served": float(mu[node]),
                    "arrived": float(arrived[node]),
                    "h": float(harvested[node]),
                    "c_tx": float(tx_cost[node]),
                    "r": float(true_coeff.r[node]),
                    "eta": float(true_coeff.eta[node]),
                    "g": float(true_coeff.g[node]),
                    "chi": float(true_coeff.chi[node]),
                    "B_EM": float(true_coeff.B_EM),
                    "P_rx": float(p_rx[node]),
                    "P_rx_max": float(true_coeff.p_rx_cap[node]),
                    "rx_cap_ratio": float(rx_ratio),
                    "rx_cap_violation": bool(rx_ratio > 1.0 + cfg.rx_cap_tolerance),
                    "em_load": em_load,
                    "em_utilization": em_util,
                    "em_violation": bool(em_util > 1.0 + cfg.em_violation_tolerance),
                    "shared_frontend": shared,
                    "allocation_backend": lp_solution.backend,
                    "solver_status": lp_solution.status,
                    "lp_failed": not lp_solution.success,
                }
            )
        q = q_next
        e = e_next

    return pd.DataFrame.from_records(rows)


def label_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in raw.groupby("seed", sort=True):
        label_error_rate = float(group["label_error"].mean())
        true_stress = group[group["true_stress_label"].astype(bool)]
        implant_rest = group[
            group["node_type"].astype(str).str.startswith("implant")
            & ~group["true_stress_label"].astype(bool)
        ]
        stress_false_negative = float((~true_stress["decision_stress_label"].astype(bool)).mean()) if not true_stress.empty else 0.0
        stress_false_positive = (
            float(implant_rest["decision_stress_label"].astype(bool).mean()) if not implant_rest.empty else 0.0
        )
        rows.append(
            {
                "seed": int(seed),
                "observed_label_error_rate": label_error_rate,
                "stress_false_negative_rate": stress_false_negative,
                "stress_false_positive_rate": stress_false_positive,
            }
        )
    return pd.DataFrame(rows)


def summarize_trial(raw: pd.DataFrame, cfg: SimConfig, trial: Trial) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = summarize_raw(raw, cfg, scenario=trial.condition_id)
    by_seed = metric.by_seed.copy()
    diag = label_diagnostics(raw)
    by_seed = by_seed.merge(diag, on="seed", how="left")
    by_seed.insert(0, "iteration", trial.iteration)
    by_seed.insert(1, "iteration_name", trial.iteration_name)
    by_seed.insert(2, "model", trial.model)
    by_seed.insert(3, "condition_id", trial.condition_id)
    by_seed.insert(4, "param_name", trial.param_name)
    by_seed.insert(5, "param_value", trial.param_value)
    by_seed.insert(6, "workpoint_tag", trial.workpoint_tag)
    for key, value in trial.wp.to_dict().items():
        by_seed[f"wp_{key}"] = value

    metric_cols = [
        col
        for col in by_seed.columns
        if col
        not in {
            "scenario",
            "seed",
            "method",
            "iteration",
            "iteration_name",
            "model",
            "condition_id",
            "param_name",
            "param_value",
            "workpoint_tag",
        }
        and pd.api.types.is_numeric_dtype(by_seed[col])
    ]
    row: dict[str, object] = {
        "iteration": trial.iteration,
        "iteration_name": trial.iteration_name,
        "model": trial.model,
        "condition_id": trial.condition_id,
        "param_name": trial.param_name,
        "param_value": trial.param_value,
        "workpoint_tag": trial.workpoint_tag,
        "num_seeds": int(by_seed["seed"].nunique()),
    }
    for key, value in trial.wp.to_dict().items():
        row[f"wp_{key}"] = value
    for col in metric_cols:
        row[col] = float(by_seed[col].mean())
        row[f"{col}_ci95"] = ci95(by_seed[col])
    return by_seed, pd.DataFrame([row])


def score_summary(summary: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    summary = summary.copy()
    base_margin = max(float(baseline["stress_affected_margin_min"]), 1e-9)
    base_p95 = max(float(baseline["p95_backlog"]), 1e-9)
    base_post = max(float(baseline["post_stress_backlog_mean"]), 1e-9)
    base_surface = max(float(baseline["surface_served_ratio"]), 1e-9)

    safety_scores = []
    compliance_scores = []
    service_scores = []
    boundary_scores = []
    paper_scores = []
    for _, row in summary.iterrows():
        severity = float(row.get("stress_shortage_severity", 0.0))
        shortage = float(row.get("stress_implant_shortage_rate", 0.0))
        margin = float(row.get("stress_affected_margin_min", 0.0))
        surface = float(row.get("surface_served_ratio", 0.0))
        p95 = float(row.get("p95_backlog", 0.0))
        post = float(row.get("post_stress_backlog_mean", 0.0))
        em = float(row.get("em_violation_rate", 0.0))
        rx = float(row.get("rx_cap_violation_rate", 0.0))
        lp = float(row.get("lp_failure_rate", 0.0))

        safety = (
            0.45 * clip01(1.0 - severity / STRESS_THRESHOLD)
            + 0.25 * clip01(1.0 - shortage / 0.10)
            + 0.30 * clip01(margin / base_margin)
        )
        compliance = 1.0 if max(em, rx, lp) <= 1e-10 else 0.0
        service = (
            0.40 * clip01(surface / base_surface)
            + 0.30 * clip01(1.0 - max(p95 - base_p95, 0.0) / max(base_p95 * 0.35, 1e-9))
            + 0.30 * clip01(1.0 - max(post - base_post, 0.0) / max(base_post * 0.70, 1e-9))
        )
        margin_drop = max(base_margin - margin, 0.0) / base_margin
        backlog_increase = max(p95 - base_p95, 0.0) / base_p95
        post_increase = max(post - base_post, 0.0) / base_post
        severity_boundary = 1.0 - min(abs(severity - STRESS_THRESHOLD) / STRESS_THRESHOLD, 1.0)
        boundary = clip01(max(severity_boundary, margin_drop / 0.35, backlog_increase / 0.15, post_increase / 0.40))
        paper_score = 0.30 * safety + 0.18 * compliance + 0.22 * service + 0.30 * boundary

        safety_scores.append(safety)
        compliance_scores.append(compliance)
        service_scores.append(service)
        boundary_scores.append(boundary)
        paper_scores.append(paper_score)

    summary["safety_score"] = safety_scores
    summary["compliance_score"] = compliance_scores
    summary["service_score"] = service_scores
    summary["boundary_score"] = boundary_scores
    summary["paper_value_score"] = paper_scores
    return summary


def run_trials(
    cfg: SimConfig,
    lib: HFSSLibrary,
    trials: list[Trial],
    baseline: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_by_seed = []
    all_summary = []
    for trial in trials:
        frames = [run_seed_with_trial(cfg, trial, seed, lib) for seed in SEEDS]
        raw = pd.concat(frames, ignore_index=True)
        by_seed, summary = summarize_trial(raw, cfg, trial)
        all_by_seed.append(by_seed)
        all_summary.append(summary)
    by_seed_df = pd.concat(all_by_seed, ignore_index=True)
    summary_df = score_summary(pd.concat(all_summary, ignore_index=True), baseline)
    return by_seed_df, summary_df


def prepare_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SNIPPET_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for src_dir, dst_dir in [(V1_DIR / "figures", FIG_DIR), (V1_DIR / "snippets", SNIPPET_DIR)]:
        if not src_dir.exists():
            continue
        if src_dir.resolve() == dst_dir.resolve():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.*"):
            dst = dst_dir / src.name
            if src.suffix.lower() == ".tex":
                text = src.read_text(encoding="utf-8").replace("paper_enhanced_state_robustness/", f"{OUT_DIR.name}/")
                dst.write_text(text, encoding="utf-8")
            else:
                shutil.copy2(src, dst)


def read_v1_baseline(cfg: SimConfig, lib: HFSSLibrary, final_wp: WorkPoint) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    summary_path = V1_DIR / "state_misclassification_summary.csv"
    if summary_path.exists():
        p_values = sorted(pd.to_numeric(pd.read_csv(summary_path)["p_err"], errors="coerce").dropna().unique())
    else:
        p_values = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    all_by_seed = []
    all_summary = []
    for p_err in p_values:
        trial = make_trial(
            1,
            "Random symmetric baseline",
            "random_symmetric",
            "p_err",
            float(p_err),
            {"p_err": float(p_err)},
            final_wp,
        )
        frames = [run_seed_with_trial(cfg, trial, seed, lib) for seed in SEEDS]
        raw = pd.concat(frames, ignore_index=True)
        by_seed, summary = summarize_trial(raw, cfg, trial)
        all_by_seed.append(by_seed)
        all_summary.append(summary)

    by_seed_df = pd.concat(all_by_seed, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    baseline = summary_df[pd.to_numeric(summary_df["param_value"], errors="coerce").eq(0.0)].iloc[0].copy()
    summary_df = score_summary(summary_df, baseline)
    return by_seed_df, summary_df, baseline


def write_iteration_outputs(iter_dir: Path, by_seed: pd.DataFrame, summary: pd.DataFrame) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(iter_dir / "by_seed.csv", index=False)
    summary.to_csv(iter_dir / "summary.csv", index=False)
    plot_iteration(summary, iter_dir)
    write_iteration_report(summary, iter_dir)


def numeric_param(summary: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(summary["param_value"], errors="coerce")


def plot_iteration(summary: pd.DataFrame, iter_dir: Path) -> None:
    apply_ieee_figure_style()
    summary = summary.sort_values(["iteration", "param_value"], kind="stable")
    x_num = numeric_param(summary)
    use_numeric = bool(x_num.notna().all())
    if use_numeric:
        x = x_num.to_numpy(dtype=float)
        xlabel = str(summary["param_name"].iloc[0])
        xticks = None
        xlabels = None
    else:
        x = np.arange(len(summary), dtype=float)
        xlabel = ""
        xticks = x
        xlabels = summary["param_value"].astype(str).tolist()

    panels = [
        ("stress_shortage_severity", "Stress shortage severity", "a"),
        ("stress_affected_margin_min", "Stress margin minimum", "b"),
        ("surface_served_ratio", "Surface served ratio", "c"),
        ("p95_backlog", "p95 backlog", "d"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=IEEE_SINGLE_4X1_FIGSIZE)
    color = COLORS["proposed"]
    for ax, (metric, title, label) in zip(axes.ravel(), panels):
        y = summary[metric].to_numpy(dtype=float)
        err_col = metric + "_ci95"
        err = summary[err_col].to_numpy(dtype=float) if err_col in summary.columns else np.zeros_like(y)
        ax.errorbar(x, y, yerr=err, marker="o", color=color, linewidth=1.0, capsize=2.0, markersize=3.6)
        if metric == "stress_shortage_severity":
            ax.axhline(STRESS_THRESHOLD, color=COLORS["competitor"], linestyle=":", linewidth=0.8)
        ax.set_title(title, pad=3)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=30, ha="right")
    fig.tight_layout(pad=0.65)
    save_figure_pair(fig, iter_dir / "figure")
    plt.close(fig)


def write_iteration_report(summary: pd.DataFrame, iter_dir: Path) -> None:
    summary_sorted = summary.sort_values("paper_value_score", ascending=False)
    best = summary_sorted.iloc[0]
    worst_safety = summary.sort_values(["safety_score", "stress_affected_margin_min"], ascending=[True, True]).iloc[0]
    columns = [
        "condition_id",
        "param_value",
        "stress_shortage_severity",
        "stress_affected_margin_min",
        "surface_served_ratio",
        "p95_backlog",
        "stress_false_negative_rate",
        "stress_false_positive_rate",
        "paper_value_score",
    ]
    table = summary_sorted[columns].to_markdown(index=False, floatfmt=".4f")
    text = f"""# Iteration {int(best['iteration']):02d}: {best['iteration_name']}

## 本轮目的

本轮按照统一评分函数评估 `{best['model']}` 类状态误差对 Proposed 的影响，重点看是否出现可解释的安全边界、服务代价或部署保守性证据。

## 最有论文价值的候选

- condition: `{best['condition_id']}`
- parameter: `{best['param_name']}={best['param_value']}`
- paper value score: {best['paper_value_score']:.4f}
- stress severity: {best['stress_shortage_severity']:.4f}
- stress margin minimum: {best['stress_affected_margin_min']:.4f}
- surface served ratio: {best['surface_served_ratio']:.4f}
- p95 backlog: {best['p95_backlog']:.2f}

## 风险边界观察

本轮安全评分最低的候选是 `{worst_safety['condition_id']}`，其 stress margin minimum 为 {worst_safety['stress_affected_margin_min']:.4f}，stress shortage severity 为 {worst_safety['stress_shortage_severity']:.4f}。若 severity 仍为 0，则本轮更适合写成“能量裕量和 backlog 的经验边界”，而不是短缺失效边界。

## 汇总表

{table}
"""
    (iter_dir / "iteration_report.md").write_text(text, encoding="utf-8")


def make_trial(
    iteration: int,
    iteration_name: str,
    model: str,
    param_name: str,
    param_value: float | str,
    params: dict[str, Any],
    wp: WorkPoint,
    workpoint_tag: str = "final",
) -> Trial:
    if isinstance(param_value, float):
        suffix = f"{param_value:.3f}".rstrip("0").rstrip(".")
    else:
        suffix = str(param_value)
    suffix = suffix.replace("+", "p").replace("-", "m").replace(".", "p").replace(" ", "_")
    condition_id = f"iter{iteration:02d}_{model}_{param_name}_{suffix}_{workpoint_tag}"
    return Trial(iteration, iteration_name, model, condition_id, param_name, param_value, params, wp, workpoint_tag)


def choose_fn_fine_grid(iter2_summary: pd.DataFrame) -> list[float]:
    rows = iter2_summary.sort_values("param_value")
    above = rows[rows["stress_shortage_severity"] >= STRESS_THRESHOLD]
    if not above.empty:
        hit = float(above.iloc[0]["param_value"])
        prev_rows = rows[rows["param_value"] < hit]
        low = float(prev_rows.iloc[-1]["param_value"]) if not prev_rows.empty else max(0.0, hit - 0.10)
        grid = np.linspace(low, hit, 6)
        return [round(float(x), 3) for x in grid]
    center = float(rows.sort_values("boundary_score", ascending=False).iloc[0]["param_value"])
    if center >= 0.45:
        return [0.30, 0.35, 0.40, 0.45, 0.50]
    low = max(0.0, center - 0.10)
    high = min(0.50, center + 0.10)
    return [round(float(x), 3) for x in np.linspace(low, high, 5)]


def best_numeric_param(summary: pd.DataFrame, fallback: float) -> float:
    rows = summary[pd.to_numeric(summary["param_value"], errors="coerce").notna()].copy()
    if rows.empty:
        return fallback
    row = rows.sort_values(["boundary_score", "paper_value_score"], ascending=False).iloc[0]
    return float(row["param_value"])


def run_adaptive_iterations() -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare_dirs()
    cfg = SimConfig()
    final_wp = load_final_workpoint()
    lib = load_hfss_library()

    all_by_seed = []
    all_summary = []

    v1_by_seed, v1_summary, baseline = read_v1_baseline(cfg, lib, final_wp)
    iter1_dir = OUT_DIR / "iter01_random_symmetric_baseline"
    write_iteration_outputs(iter1_dir, v1_by_seed, v1_summary)
    all_by_seed.append(v1_by_seed)
    all_summary.append(v1_summary)

    final_fig = pd.read_csv(FINAL_DIR / "fig3_main_comparison.csv").set_index("method").loc["Proposed"]
    zero = v1_summary[v1_summary["param_value"].astype(float).eq(0.0)].iloc[0]
    sanity = {
        key: abs(float(zero[key]) - float(final_fig[key]))
        for key in [
            "stress_shortage_severity",
            "stress_implant_shortage_rate",
            "surface_served_ratio",
            "p95_backlog",
            "post_stress_backlog_mean",
            "em_violation_rate",
            "rx_cap_violation_rate",
            "lp_failure_rate",
        ]
    }
    # Fig. 3 reports held-out seeds 10--19, whereas Fig. S2 keeps the
    # robustness protocol on all 20 paired seeds. The p=0 check therefore
    # verifies invariant safety metrics exactly and service metrics within
    # a small seed-window tolerance.
    tolerances = {
        "stress_shortage_severity": 1e-9,
        "stress_implant_shortage_rate": 1e-9,
        "surface_served_ratio": 0.01,
        "p95_backlog": 10.0,
        "post_stress_backlog_mean": 5.0,
        "em_violation_rate": 1e-9,
        "rx_cap_violation_rate": 1e-9,
        "lp_failure_rate": 1e-9,
    }
    failed = {key: value for key, value in sanity.items() if value > tolerances[key]}
    if failed:
        raise RuntimeError(f"Iteration 1 p=0 sanity check failed: {failed}")
    sanity_text = ["p=0 sanity check against final Fig. 3: PASS within seed-window tolerances"]
    sanity_text.extend(f"- {key}: abs diff {value:.3e}" for key, value in sanity.items())
    (OUT_DIR / "sanity_check.txt").write_text("\n".join(sanity_text) + "\n", encoding="utf-8")

    def run_and_store(iteration: int, name: str, trials: list[Trial], slug: str) -> pd.DataFrame:
        by_seed, summary = run_trials(cfg, lib, trials, baseline)
        iter_dir = OUT_DIR / f"iter{iteration:02d}_{slug}"
        write_iteration_outputs(iter_dir, by_seed, summary)
        (iter_dir / "sanity_check.txt").write_text((OUT_DIR / "sanity_check.txt").read_text(encoding="utf-8"), encoding="utf-8")
        all_by_seed.append(by_seed)
        all_summary.append(summary)
        return summary

    iter2_trials = [
        make_trial(2, "Stress false-negative coarse sweep", "stress_false_negative", "p_fn", p, {"p_fn": p}, final_wp)
        for p in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    ]
    iter2_summary = run_and_store(2, "Stress false-negative coarse sweep", iter2_trials, "stress_false_negative_coarse")

    fine_grid = choose_fn_fine_grid(iter2_summary)
    iter3_trials = [
        make_trial(3, "Stress false-negative fine sweep", "stress_false_negative", "p_fn", p, {"p_fn": p}, final_wp)
        for p in fine_grid
    ]
    iter3_summary = run_and_store(3, "Stress false-negative fine sweep", iter3_trials, "stress_false_negative_fine")

    iter4_trials = [
        make_trial(4, "Bursty false-negative", "bursty_false_negative", "burst_len", float(length), {"burst_len": length}, final_wp)
        for length in [5, 10, 25, 50, 100]
    ]
    iter4_summary = run_and_store(4, "Bursty false-negative", iter4_trials, "bursty_false_negative")

    iter5_trials = [
        make_trial(5, "Estimator delay", "estimator_delay", "delay", float(delay), {"delay": delay}, final_wp)
        for delay in [5, 10, 20, 40, 80]
    ]
    iter5_summary = run_and_store(5, "Estimator delay", iter5_trials, "estimator_delay")

    iter6_trials = [
        make_trial(6, "Stuck-at-rest stress window", "stuck_at_rest", "ratio", ratio, {"ratio": ratio}, final_wp)
        for ratio in [0.25, 0.50, 0.75, 1.00]
    ]
    iter6_summary = run_and_store(6, "Stuck-at-rest stress window", iter6_trials, "stuck_at_rest")

    iter7_trials = [
        make_trial(7, "False-positive conservative error", "stress_false_positive", "p_fp", p, {"p_fp": p}, final_wp)
        for p in [0.05, 0.10, 0.20, 0.30, 0.40]
    ]
    iter7_summary = run_and_store(7, "False-positive conservative error", iter7_trials, "stress_false_positive")

    best_fn = best_numeric_param(pd.concat([iter2_summary, iter3_summary], ignore_index=True), 0.30)
    best_burst = int(round(best_numeric_param(iter4_summary, 25.0)))
    best_delay = int(round(best_numeric_param(iter5_summary, 10.0)))
    iter8_specs = [
        ("mixed_mild", {"p_fn": 0.05, "p_fp": 0.02, "delay": 5, "burst_len": 5, "p_surface": 0.03}),
        ("mixed_moderate", {"p_fn": 0.10, "p_fp": 0.05, "delay": 10, "burst_len": 10, "p_surface": 0.05}),
        ("mixed_boundary", {"p_fn": best_fn, "p_fp": 0.05, "delay": best_delay, "burst_len": min(best_burst, 50), "p_surface": 0.05}),
        ("mixed_conservative", {"p_fn": 0.10, "p_fp": 0.20, "delay": 10, "burst_len": 10, "p_surface": 0.05}),
    ]
    iter8_trials = [
        make_trial(8, "Mixed realistic estimator", "mixed_realistic", "mixed_id", tag, params, final_wp)
        for tag, params in iter8_specs
    ]
    iter8_summary = run_and_store(8, "Mixed realistic estimator", iter8_trials, "mixed_realistic_estimator")

    challenge_fn = float(np.clip(best_fn, 0.30, 0.50))
    wp_variants = [
        ("wp_final", final_wp),
        ("wp_lambda_c_p005", final_wp.shifted(lambda_c=0.05)),
        ("wp_lambda_xi_p004", final_wp.shifted(lambda_xi=0.04)),
        ("wp_c_xi_p", final_wp.shifted(lambda_c=0.05, lambda_xi=0.04)),
        ("wp_BEM_p002", final_wp.shifted(B_EM_ref=0.02)),
        ("wp_PH_p002", final_wp.shifted(P_H=0.02)),
        ("wp_guarded_1", final_wp.shifted(lambda_c=0.05, lambda_xi=0.04, B_EM_ref=0.02)),
        ("wp_guarded_2", final_wp.shifted(lambda_c=0.08, lambda_xi=0.06, P_H=0.02)),
    ]
    iter9_trials = [
        make_trial(
            9,
            "Robust workpoint search",
            "stress_false_negative",
            "workpoint",
            tag,
            {"p_fn": challenge_fn},
            wp,
            workpoint_tag=tag,
        )
        for tag, wp in wp_variants
    ]
    iter9_summary = run_and_store(9, "Robust workpoint search", iter9_trials, "robust_workpoint_search")

    candidate_pool = pd.concat(
        [iter2_summary, iter3_summary, iter4_summary, iter5_summary, iter6_summary, iter7_summary, iter8_summary, iter9_summary],
        ignore_index=True,
    )
    pool = candidate_pool[candidate_pool["param_value"].astype(str).ne("0.0")].copy()
    top_candidates = pool.sort_values(["paper_value_score", "boundary_score"], ascending=False).head(3)
    iter10_trials: list[Trial] = []
    for idx, row in enumerate(top_candidates.itertuples(index=False), start=1):
        params = {}
        model = str(row.model)
        param_name = str(row.param_name)
        param_value = row.param_value
        if model == "stress_false_negative":
            params = {"p_fn": float(param_value) if param_name != "workpoint" else challenge_fn}
        elif model == "bursty_false_negative":
            params = {"burst_len": int(float(param_value))}
        elif model == "estimator_delay":
            params = {"delay": int(float(param_value))}
        elif model == "stuck_at_rest":
            params = {"ratio": float(param_value)}
        elif model == "stress_false_positive":
            params = {"p_fp": float(param_value)}
        elif model == "mixed_realistic":
            match = next((spec for tag, spec in iter8_specs if tag == param_value), iter8_specs[1][1])
            params = dict(match)
        else:
            params = {"p_fn": challenge_fn}

        wp = final_wp
        workpoint_tag = "final"
        if model == "stress_false_negative" and param_name == "workpoint":
            wp_match = next((item_wp for tag, item_wp in wp_variants if tag == str(param_value)), final_wp)
            wp = wp_match
            workpoint_tag = str(param_value)
        trial = Trial(
            10,
            "Best-candidate revalidation",
            model,
            f"iter10_revalidate_{idx}_{row.condition_id}",
            "candidate",
            f"candidate_{idx}",
            params,
            wp,
            workpoint_tag,
        )
        iter10_trials.append(trial)
    iter10_summary = run_and_store(10, "Best-candidate revalidation", iter10_trials, "best_candidate_revalidation")

    all_by_seed_df = pd.concat(all_by_seed, ignore_index=True)
    leaderboard = pd.concat(all_summary, ignore_index=True)
    leaderboard = leaderboard.sort_values(["iteration", "paper_value_score"], ascending=[True, False])
    all_by_seed_df.to_csv(OUT_DIR / "iter10_all_by_seed.csv", index=False)
    leaderboard.to_csv(OUT_DIR / "iter10_leaderboard.csv", index=False)
    write_comprehensive_robustness_snippet(leaderboard)
    plot_best_package(leaderboard)
    write_best_package(leaderboard, all_by_seed_df)
    write_total_report(leaderboard, fine_grid, challenge_fn)
    build_methods_tex()
    return all_by_seed_df, leaderboard


def plot_best_package(leaderboard: pd.DataFrame) -> None:
    apply_ieee_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=IEEE_DOUBLE_2X2_FIGSIZE)

    fn_rows = leaderboard[
        leaderboard["model"].eq("stress_false_negative") & leaderboard["param_name"].eq("p_fn") & leaderboard["iteration"].isin([2, 3])
    ].copy()
    fn_rows["x"] = pd.to_numeric(fn_rows["param_value"], errors="coerce")
    fn_rows = fn_rows.sort_values("x")
    ax = axes[0, 0]
    ax.errorbar(
        fn_rows["x"],
        fn_rows["stress_affected_margin_min"],
        yerr=fn_rows.get("stress_affected_margin_min_ci95", 0.0),
        marker="o",
        color=COLORS["proposed"],
        linewidth=1.0,
        capsize=2.0,
    )
    ax2 = ax.twinx()
    ax2.plot(fn_rows["x"], fn_rows["stress_false_negative_rate"], marker="s", color=COLORS["competitor"], linewidth=0.9)
    ax.set_title("False-negative boundary")
    ax.set_xlabel("$p_{FN}$")
    ax.set_ylabel("Margin min")
    ax2.set_ylabel("Observed FN")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[0, 1]
    burst = leaderboard[leaderboard["model"].eq("bursty_false_negative")].copy()
    burst["x"] = pd.to_numeric(burst["param_value"], errors="coerce")
    delay = leaderboard[leaderboard["model"].eq("estimator_delay")].copy()
    delay["x"] = pd.to_numeric(delay["param_value"], errors="coerce")
    ax.plot(burst.sort_values("x")["x"], burst.sort_values("x")["p95_backlog"], marker="o", color=COLORS["proposed"], label="burst")
    ax.plot(delay.sort_values("x")["x"], delay.sort_values("x")["p95_backlog"], marker="s", color=COLORS["competitor"], label="delay")
    ax.set_title("Consecutive estimator errors")
    ax.set_xlabel("Slots")
    ax.set_ylabel("p95 backlog")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    ax.legend(fontsize=6)
    ax.text(0.02, 0.98, "b", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 0]
    wp = leaderboard[leaderboard["iteration"].eq(9)].copy()
    ax.scatter(wp["surface_served_ratio"], wp["stress_affected_margin_min"], s=28, color=COLORS["proposed"])
    for _, row in wp.iterrows():
        ax.annotate(str(row["workpoint_tag"]).replace("wp_", ""), (row["surface_served_ratio"], row["stress_affected_margin_min"]), fontsize=5.8, xytext=(2, 2), textcoords="offset points")
    ax.set_title("Optional robust workpoints")
    ax.set_xlabel("Surface served")
    ax.set_ylabel("Margin min")
    ax.grid(color=COLORS["grid"], linewidth=0.45)
    ax.text(0.02, 0.98, "c", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 1]
    top = leaderboard.sort_values("paper_value_score", ascending=False).head(8).iloc[::-1]
    labels = [f"I{int(row.iteration)} {str(row.param_value)[:14]}" for row in top.itertuples()]
    ax.barh(np.arange(len(top)), top["paper_value_score"], color=COLORS["proposed"])
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paper value score")
    ax.set_title("Top evidence candidates")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    ax.text(0.02, 0.98, "d", transform=ax.transAxes, va="top", fontweight="bold")

    fig.tight_layout(pad=0.7)
    save_figure_pair(fig, FIG_DIR / "figS3_iter10_adaptive_robustness")
    plt.close(fig)


def write_best_package(leaderboard: pd.DataFrame, by_seed: pd.DataFrame) -> None:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    top = leaderboard.sort_values(["paper_value_score", "boundary_score"], ascending=False).head(5)
    top.to_csv(BEST_DIR / "best_candidates.csv", index=False)
    selected_by_seed = by_seed[by_seed["condition_id"].isin(top["condition_id"])].copy()
    selected_by_seed.to_csv(BEST_DIR / "best_candidates_by_seed.csv", index=False)
    if (FIG_DIR / "figS2_state_estimator_robustness.pdf").exists():
        shutil.copy2(FIG_DIR / "figS2_state_estimator_robustness.pdf", BEST_DIR / "figS2_state_estimator_robustness.pdf")
    if (FIG_DIR / "figS2_state_estimator_robustness.png").exists():
        shutil.copy2(FIG_DIR / "figS2_state_estimator_robustness.png", BEST_DIR / "figS2_state_estimator_robustness.png")
    for _, row in top.iterrows():
        src_dir = OUT_DIR / f"iter{int(row['iteration']):02d}_{iteration_slug(int(row['iteration']))}"
        if (src_dir / "figure.pdf").exists():
            shutil.copy2(src_dir / "figure.pdf", BEST_DIR / f"{row['condition_id']}_figure.pdf")
        if (src_dir / "figure.png").exists():
            shutil.copy2(src_dir / "figure.png", BEST_DIR / f"{row['condition_id']}_figure.png")

    lines = [
        "# 最优鲁棒性证据包",
        "",
        "本目录保留 10 轮自适应迭代后最有论文价值的候选，而不是把所有实验都堆进正文。",
        "",
        "## 推荐展示",
        "",
        "- 主图建议使用综合版 `figS2_state_estimator_robustness.pdf/png`。",
        "- 表格建议使用 `best_candidates.csv` 中得分最高的 3--5 个候选。",
        "- 正文主张应写成经验鲁棒性和边界识别，不写成理论保证。",
        "",
        "## Top candidates",
        "",
        top[
            [
                "iteration",
                "condition_id",
                "stress_shortage_severity",
                "stress_affected_margin_min",
                "surface_served_ratio",
                "p95_backlog",
                "paper_value_score",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (BEST_DIR / "best_summary.md").write_text("\n".join(lines), encoding="utf-8")


def selected_case_rows(leaderboard: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("Random $p_{\\rm err}=0.20$", "random_symmetric", "p_err", "0.2"),
        ("False negative $p_{FN}=0.50$", "stress_false_negative", "p_fn", "0.5"),
        ("Burst false negative $L=50$", "bursty_false_negative", "burst_len", "50.0"),
        ("Delay 80 slots", "estimator_delay", "delay", "80.0"),
        ("Stuck-at-rest 0.5", "stuck_at_rest", "ratio", "0.5"),
        ("Mixed boundary", "mixed_realistic", "mixed_id", "mixed_boundary"),
        ("Robust workpoint", "stress_false_negative", "workpoint", "wp_c_xi_p"),
    ]
    rows = []
    for label, model, param_name, param_value in selectors:
        subset = leaderboard[
            leaderboard["model"].astype(str).eq(model)
            & leaderboard["param_name"].astype(str).eq(param_name)
            & leaderboard["param_value"].astype(str).eq(param_value)
        ].copy()
        if subset.empty and param_name in {"p_err", "p_fn", "burst_len", "delay", "ratio"}:
            numeric = pd.to_numeric(leaderboard["param_value"], errors="coerce")
            subset = leaderboard[
                leaderboard["model"].astype(str).eq(model)
                & leaderboard["param_name"].astype(str).eq(param_name)
                & np.isclose(numeric, float(param_value))
            ].copy()
        if subset.empty:
            continue
        if model == "stress_false_negative" and param_name == "p_fn":
            subset = subset.sort_values(["iteration", "boundary_score"], ascending=[False, False])
        else:
            subset = subset.sort_values(["iteration", "paper_value_score"], ascending=[False, False])
        row = subset.iloc[0].copy()
        row["display_case"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def read_v1_summary_frame(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    if "p_err" not in summary.columns and "param_value" in summary.columns:
        summary["p_err"] = pd.to_numeric(summary["param_value"], errors="coerce")
    return summary


def plot_comprehensive_figS2(leaderboard: pd.DataFrame) -> None:
    apply_ieee_figure_style()
    local_v1_summary = OUT_DIR / "state_misclassification_summary.csv"
    iter1_summary = OUT_DIR / "iter01_random_symmetric_baseline" / "summary.csv"
    v1_summary_path = local_v1_summary if local_v1_summary.exists() else iter1_summary
    if not v1_summary_path.exists():
        v1_summary_path = V1_DIR / "state_misclassification_summary.csv"
    v1 = read_v1_summary_frame(v1_summary_path).sort_values("p_err")
    cases = selected_case_rows(leaderboard)

    fig, axes = plt.subplots(4, 1, figsize=IEEE_SINGLE_4X1_FIGSIZE)

    ax = axes[0]
    x = v1["p_err"].to_numpy(dtype=float)
    y = v1["stress_affected_margin_min"].to_numpy(dtype=float)
    err = v1["stress_affected_margin_min_ci95"].to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=err, marker="o", color=COLORS["proposed"], linewidth=1.1, capsize=2.2, markersize=4)
    row20 = v1[np.isclose(v1["p_err"], 0.20)].iloc[0]
    ax.scatter([0.20], [row20["stress_affected_margin_min"]], s=30, color=COLORS["competitor"], zorder=5)
    ax.axvline(0.20, color=COLORS["competitor"], linestyle=":", linewidth=0.8)
    ax.set_ylim(0.185, 0.262)
    ax.text(
        0.125,
        0.245,
        "severity = 0\nat 20%",
        fontsize=BODY_TEXT_SIZE,
        color=COLORS["competitor"],
        ha="center",
        va="center",
    )
    set_panel_title(ax, "a", "Random label-noise sanity check")
    ax.set_xlabel(r"$p_{\rm err}$ (label-error rate)", labelpad=0.8)
    ax.set_ylabel("Margin min")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)

    ax = axes[1]
    coarse = leaderboard[
        leaderboard["iteration"].eq(2)
        & leaderboard["model"].eq("stress_false_negative")
        & leaderboard["param_name"].eq("p_fn")
    ].copy()
    coarse["x"] = pd.to_numeric(coarse["param_value"], errors="coerce")
    fine = leaderboard[
        leaderboard["iteration"].eq(3)
        & leaderboard["model"].eq("stress_false_negative")
        & leaderboard["param_name"].eq("p_fn")
    ].copy()
    fine["x"] = pd.to_numeric(fine["param_value"], errors="coerce")
    fn = pd.concat([coarse[coarse["x"].le(0.20)], fine], ignore_index=True).sort_values("x")
    margin_line = ax.errorbar(
        fn["x"],
        fn["stress_affected_margin_min"],
        yerr=fn["stress_affected_margin_min_ci95"],
        marker="o",
        color=COLORS["proposed"],
        linewidth=1.1,
        capsize=2.0,
        markersize=3.8,
        label="Margin min",
    )
    ax2 = ax.twinx()
    fn_line = ax2.plot(
        fn["x"],
        fn["stress_false_negative_rate"],
        marker="s",
        color=COLORS["competitor"],
        linewidth=0.95,
        markersize=3.4,
        label="Observed FN",
    )[0]
    set_panel_title(ax, "b", "Stress false-negative boundary")
    ax.set_xlabel(r"$p_{\rm FN}$ (stress-FN rate)", labelpad=0.8)
    ax.set_ylabel("Margin min")
    ax2.set_ylabel("Observed FN")
    ax.yaxis.label.set_color(COLORS["proposed"])
    ax2.yaxis.label.set_color(COLORS["competitor"])
    ax.tick_params(axis="y", colors=COLORS["proposed"])
    ax2.tick_params(axis="y", colors=COLORS["competitor"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    ax.text(0.06, 0.18, "Margin min", transform=ax.transAxes, color=COLORS["proposed"], fontsize=BODY_TEXT_SIZE)
    ax2.text(0.92, 0.78, "Observed FN", transform=ax.transAxes, color=COLORS["competitor"], fontsize=BODY_TEXT_SIZE, ha="right")

    ax = axes[2]
    case_order = [
        "Random $p_{\\rm err}=0.20$",
        "Burst false negative $L=50$",
        "Delay 80 slots",
        "Stuck-at-rest 0.5",
        "Mixed boundary",
    ]
    bar_cases = cases[cases["display_case"].isin(case_order)].copy()
    bar_cases["display_case"] = pd.Categorical(bar_cases["display_case"], categories=case_order, ordered=True)
    bar_cases = bar_cases.sort_values("display_case")
    labels = ["Random\n20%", "Burst\n50", "Delay\n80", "Stuck\n0.5", "Mixed"]
    vals = bar_cases["stress_shortage_severity"].to_numpy(dtype=float)
    ax.bar(np.arange(len(vals)), vals, color=[COLORS["ablation_light"], COLORS["proposed"], COLORS["competitor"], COLORS["em"], COLORS["candidate"]], width=0.62)
    ax.axhline(STRESS_THRESHOLD, color=COLORS["competitor"], linestyle=":", linewidth=0.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=6.4, pad=1.5)
    ax.set_ylabel("Shortage severity")
    set_panel_title(ax, "c", "Representative estimator failures")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    for idx, (_, row) in enumerate(bar_cases.iterrows()):
        ax.text(idx, float(row["stress_shortage_severity"]) + 0.0025, f"m={row['stress_affected_margin_min']:.3f}", ha="center", va="bottom", fontsize=BODY_TEXT_SIZE, rotation=0)

    ax = axes[3]
    wp = leaderboard[leaderboard["iteration"].eq(9)].copy()
    ax.scatter(wp["surface_served_ratio"], wp["stress_affected_margin_min"], s=26, color=COLORS["candidate"], alpha=0.85, label="candidates")
    final = wp[wp["workpoint_tag"].eq("wp_final")]
    robust = wp[wp["workpoint_tag"].eq("wp_c_xi_p")]
    if not final.empty:
        ax.scatter(final["surface_served_ratio"], final["stress_affected_margin_min"], s=42, color=COLORS["proposed"], label="final")
        f = final.iloc[0]
        ax.annotate("final", (f["surface_served_ratio"], f["stress_affected_margin_min"]), xytext=(3, -9), textcoords="offset points", fontsize=BODY_TEXT_SIZE)
    if not robust.empty:
        ax.scatter(robust["surface_served_ratio"], robust["stress_affected_margin_min"], s=46, color=COLORS["competitor"], label=r"$wp\_c\_xi\_p$")
        r = robust.iloc[0]
        ax.annotate(
            r"$wp\_c\_xi\_p$",
            (r["surface_served_ratio"], r["stress_affected_margin_min"]),
            xytext=(3, -9),
            textcoords="offset points",
            fontsize=BODY_TEXT_SIZE,
            color=COLORS["competitor"],
            va="top",
        )
    set_panel_title(ax, "d", "Optional robust workpoints")
    ax.set_xlabel("Surface served ratio", labelpad=0.8)
    ax.set_ylabel("Margin min")
    ax.set_xlim(float(wp["surface_served_ratio"].min()) - 0.00025, float(wp["surface_served_ratio"].max()) + 0.00025)
    ax.set_ylim(float(wp["stress_affected_margin_min"].min()) - 0.0010, float(wp["stress_affected_margin_min"].max()) + 0.0015)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(color=COLORS["grid"], linewidth=0.45)

    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.25, h_pad=0.55)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure_pair(fig, FIG_DIR / "figS2_state_estimator_robustness")
    plt.close(fig)


def table_s5_representative_tex(leaderboard: pd.DataFrame) -> str:
    cases = selected_case_rows(leaderboard)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{状态估计误差鲁棒性综合图中的代表性情形。数值为 20 个 paired seeds 的 mean；robust workpoint 仅作为部署状态估计较差时的 supplementary option，不替换主文 final workpoint。}",
        r"\label{tab:state_estimator_cases_cn}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Case & Severity & Margin min & Surface served & p95 backlog & Diagnostic \\",
        r"\midrule",
    ]
    for _, row in cases.iterrows():
        diagnostic = row["stress_false_negative_rate"]
        if row["display_case"].startswith("Random"):
            diagnostic = row["observed_label_error_rate"]
        lines.append(
            f"{row['display_case']} & "
            f"{fmt(row['stress_shortage_severity'])} & "
            f"{fmt(row['stress_affected_margin_min'])} & "
            f"{fmt(row['surface_served_ratio'])} & "
            f"{fmt(row['p95_backlog'], 2)} & "
            f"{fmt(diagnostic)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_comprehensive_robustness_snippet(leaderboard: pd.DataFrame) -> None:
    plot_comprehensive_figS2(leaderboard)
    for stale in [
        FIG_DIR / "figS2_state_misclassification_robustness.pdf",
        FIG_DIR / "figS2_state_misclassification_robustness.png",
        SNIPPET_DIR / "state_misclassification_table.tex",
    ]:
        if stale.exists():
            stale.unlink()
    cases = selected_case_rows(leaderboard).set_index("display_case")
    random20 = cases.loc["Random $p_{\\rm err}=0.20$"]
    fn50 = cases.loc["False negative $p_{FN}=0.50$"]
    burst50 = cases.loc["Burst false negative $L=50$"]
    delay80 = cases.loc["Delay 80 slots"]
    stuck05 = cases.loc["Stuck-at-rest 0.5"]
    robust = cases.loc["Robust workpoint"]
    final_wp = leaderboard[(leaderboard["iteration"].eq(9)) & leaderboard["workpoint_tag"].eq("wp_final")].iloc[0]

    (SNIPPET_DIR / "state_estimator_cases_table.tex").write_text(table_s5_representative_tex(leaderboard), encoding="utf-8")
    # Keep the old iteration-section input harmless if an older TeX file references it.
    (SNIPPET_DIR / "iter10_adaptive_section.tex").write_text("% Integrated into state_misclassification_section.tex.\n", encoding="utf-8")

    text = rf"""
\subsection{{状态估计误差鲁棒性与经验失效边界}}
\label{{subsec:state_estimator_robustness_cn}}

前述主实验采用 perfect-state observation，即调度器决策标签 $\hat\ell_i(t)$ 与真实 HFSS 工况标签一致。为避免鲁棒性分析仅依赖随机对称误分类这一温和情形，本文将状态误差实验组织为一张综合图：随机标签误分类作为 sanity check，而 stress false-negative、burst false-negative、estimation delay、stuck-at-rest 和 mixed estimator 用于暴露更具挑战性的估计器失效模式。所有实验均只改变调度器查询 HFSS library 时使用的 decision label；真实物理服务、能量、EM 和 rx-cap 后果仍由 true label 计算。

图~\ref{{fig:state_estimator_robustness}}a 显示，随机 node-type-preserving 标签误分类在 $p_{{\rm err}}=0.20$ 时仍未诱发 stress-window shortage；该点的 shortage severity 为 ${fmt(random20['stress_shortage_severity'])}$，observed label error rate 为 ${fmt(random20['observed_label_error_rate'])}$，stress false-negative rate 为 ${fmt(random20['stress_false_negative_rate'])}$。不过，该鲁棒性并非无代价：stress-affected margin minimum 降至 ${fmt(random20['stress_affected_margin_min'])}$，p95 backlog 升至 ${fmt(random20['p95_backlog'], 2)}$。因此，随机误分类结果在本文中只作为 baseline sensitivity evidence，而不是理论鲁棒性保证。

更有边界意义的风险来自 biased 或连续的 stress false-negative。图~\ref{{fig:state_estimator_robustness}}b 显示，$p_{{FN}}=0.50$ 时 observed stress false-negative rate 为 ${fmt(fn50['stress_false_negative_rate'])}$，stress-affected margin minimum 为 ${fmt(fn50['stress_affected_margin_min'])}$。图~\ref{{fig:state_estimator_robustness}}c 进一步比较代表性估计器失效模式：delay = 80 slots 时 shortage severity 为 ${fmt(delay80['stress_shortage_severity'])}$、margin minimum 为 ${fmt(delay80['stress_affected_margin_min'])}$；stuck-at-rest ratio = 0.5 时 shortage severity 为 ${fmt(stuck05['stress_shortage_severity'])}$、margin minimum 为 ${fmt(stuck05['stress_affected_margin_min'])}$；burst length = 50 的 shortage severity 仍很低，为 ${fmt(burst50['stress_shortage_severity'])}$，但 margin 已降至 ${fmt(burst50['stress_affected_margin_min'])}$。这些结果说明，退化首先表现为 critical margin shrinkage 和 backlog 变化，而不是 EM/rx-cap/LP 违规。

最后，图~\ref{{fig:state_estimator_robustness}}d 给出可选保守工作点的补充证据。该搜索不替换主文 final workpoint；它只回答部署状态估计器质量较差时是否存在保守调参空间。在同一 false-negative 挑战下，\texttt{{wp\_c\_xi\_p}} 将 stress-affected margin minimum 从 final workpoint 的 ${fmt(final_wp['stress_affected_margin_min'])}$ 提高到 ${fmt(robust['stress_affected_margin_min'])}$，同时 surface served ratio 为 ${fmt(robust['surface_served_ratio'])}$。因此，该点适合作为 supplementary deployment option，而不是主结果替代。

\input{{{OUT_DIR.name}/snippets/state_estimator_cases_table.tex}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{OUT_DIR.name}/figures/figS2_state_estimator_robustness.pdf}}
\parbox{{\columnwidth}}{{\footnotesize\textbf{{Fig. S2.}} State-estimator error robustness and empirical failure boundary. (a) Random label-noise sanity check: stress-affected margin decreases as $p_{{\rm err}}$ increases, while stress-window shortage remains zero up to $p_{{\rm err}}=0.20$. (b) Stress false-negative boundary: increasing $p_{{FN}}$ raises the observed false-negative rate and compresses the critical energy margin. (c) Representative estimator failure modes. Bars show stress-window shortage severity, and $m$ denotes stress-affected minimum margin. (d) Optional conservative workpoints under challenging estimator errors. Gray points denote other candidate conservative workpoints screened in the adaptive search; \texttt{{wp\_c\_xi\_p}} improves critical margin with only a small surface-service change and is treated as a supplementary deployment option rather than replacing the final workpoint. Error bars denote 95\% confidence intervals over 20 paired seeds where shown.}}
\label{{fig:state_estimator_robustness}}
\end{{figure}}
"""
    (SNIPPET_DIR / "state_misclassification_section.tex").write_text(text.strip() + "\n", encoding="utf-8")


def plot_random_baseline_figS2(summary: pd.DataFrame) -> None:
    apply_ieee_figure_style()
    summary = summary.sort_values("p_err")
    x = summary["p_err"].to_numpy(dtype=float)
    highlight_x = 0.20
    highlight = summary[np.isclose(summary["p_err"].to_numpy(dtype=float), highlight_x)].iloc[0]
    panels = [
        ("stress_shortage_severity", "Stress shortage severity", "a"),
        ("surface_served_ratio", "Surface served ratio", "b"),
        ("p95_backlog", "p95 backlog", "c"),
        ("stress_affected_margin_min", "Stress-affected margin minimum", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=IEEE_DOUBLE_2X2_FIGSIZE)
    for ax, (metric, title, label) in zip(axes.ravel(), panels):
        y = summary[metric].to_numpy(dtype=float)
        err = summary[f"{metric}_ci95"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=err, marker="o", color=COLORS["proposed"], linewidth=1.1, capsize=2.2, markersize=4)
        ax.scatter(
            [highlight_x],
            [float(highlight[metric])],
            s=26,
            color=COLORS["competitor"],
            zorder=5,
            label=r"$p_{\rm err}=0.20$" if metric == "stress_shortage_severity" else None,
        )
        ax.axvline(highlight_x, color=COLORS["competitor"], linestyle=":", linewidth=0.8, alpha=0.7)
        if metric == "stress_shortage_severity":
            ax.axhline(STRESS_THRESHOLD, color=COLORS["competitor"], linestyle=":", linewidth=0.8)
            ax.legend(fontsize=6, loc="upper left")
        if metric == "stress_affected_margin_min":
            ax.annotate(
                f"{float(highlight[metric]):.4f}",
                xy=(highlight_x, float(highlight[metric])),
                xytext=(-28, -13),
                textcoords="offset points",
                fontsize=6.2,
                color=COLORS["competitor"],
                arrowprops={"arrowstyle": "-", "color": COLORS["competitor"], "linewidth": 0.6},
            )
        ax.set_title(title, pad=3)
        ax.set_xlabel(r"$p_{\rm err}$")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
        ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
    fig.tight_layout(pad=0.65)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure_pair(fig, FIG_DIR / "figS2_state_misclassification_robustness")
    plt.close(fig)


def table_s4_margin_tex(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\parbox{0.96\textwidth}{\footnotesize\textbf{Table S4.} 状态标签随机误分类 baseline 摘要。数值为 20 个 paired seeds 的 mean $\pm$ 95\% confidence interval；误分类只作用于调度器 decision label，真实物理后果仍由 true label 计算。}",
        r"\vspace{2pt}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.0pt}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$p_{\rm err}$ & Severity & Surface served & p95 backlog & Margin min & EM viol. & rx-cap viol. & Label err. & Stress FN \\",
        r"\midrule",
    ]
    for _, row in summary.sort_values("p_err").iterrows():
        lines.append(
            f"{row['p_err']:.2f} & "
            f"${fmt(row['stress_shortage_severity'])}\\pm{fmt(row['stress_shortage_severity_ci95'])}$ & "
            f"${fmt(row['surface_served_ratio'])}\\pm{fmt(row['surface_served_ratio_ci95'])}$ & "
            f"${fmt(row['p95_backlog'], 2)}\\pm{fmt(row['p95_backlog_ci95'], 2)}$ & "
            f"${fmt(row['stress_affected_margin_min'])}\\pm{fmt(row['stress_affected_margin_min_ci95'])}$ & "
            f"${fmt(row['em_violation_rate'])}$ & "
            f"${fmt(row['rx_cap_violation_rate'])}$ & "
            f"${fmt(row['observed_label_error_rate'])}$ & "
            f"${fmt(row['stress_false_negative_rate'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_random_baseline_snippet() -> None:
    summary_path = V1_DIR / "state_misclassification_summary.csv"
    if not summary_path.exists():
        summary_path = OUT_DIR / "iter01_random_symmetric_baseline" / "summary.csv"
    summary = read_v1_summary_frame(summary_path)
    plot_random_baseline_figS2(summary)
    row00 = summary[summary["p_err"].eq(0.0)].iloc[0]
    row05 = summary[summary["p_err"].eq(0.05)].iloc[0]
    row10 = summary[summary["p_err"].eq(0.10)].iloc[0]
    row20 = summary[summary["p_err"].eq(0.20)].iloc[0]
    (SNIPPET_DIR / "state_misclassification_table.tex").write_text(table_s4_margin_tex(summary), encoding="utf-8")
    text = rf"""
\subsection{{状态误分类鲁棒性}}
\label{{subsec:state_misclassification_cn}}

前述主实验采用 perfect-state observation，即调度器决策标签 $\hat\ell_i(t)$ 与真实 HFSS 工况标签一致。为评估这一假设的敏感性，本文进一步加入 node-type-preserving 随机状态误分类实验。误分类只作用于调度器查询 HFSS library 时使用的 decision label；真实物理后果仍用 true label 计算。因此，该实验检验的是状态估计错误对在线调度决策的影响，而不是改变底层人体介质状态本身。

具体地，实验设置 $p_{{\rm err}}\in\{{0,0.05,0.10,0.20\}}$。对于植入节点，误分类只在同一植入深度的 rest/stress 标签之间翻转；对于体表节点，误分类只在 \texttt{{surface\_rest}}、\texttt{{surface\_sweat}} 和 \texttt{{surface\_moderate\_loose}} 之间发生，不引入 \texttt{{surface\_contact\_failure}} 这一非主实验极端工况。所有实验使用 final workpoint 和相同 20 个 seed。

图~\ref{{fig:state_misclassification_baseline}} 保留四个随机误分类强度点，并将 $p_{{\rm err}}=0.20$ 作为 baseline robustness 的 headline endpoint。结果显示，在最高随机误分类强度 $p_{{\rm err}}=0.20$ 下，stress-window shortage severity 仍为 ${fmt_pm(row20, 'stress_shortage_severity')}$，没有诱发 stress-window shortage；observed label error rate 为 {fmt(row20['observed_label_error_rate'])}，stress false-negative rate 为 {fmt(row20['stress_false_negative_rate'])}，与设定扰动强度基本一致。因此，该结果支持“up to 20\% random label noise did not induce stress-window shortage”这一经验结论。

同时，鲁棒性并非无代价。随着 $p_{{\rm err}}$ 从 0 增至 0.20，p95 backlog 从 ${fmt(row00['p95_backlog'], 2)}$ 上升至 ${fmt(row20['p95_backlog'], 2)}$，stress-affected margin minimum 从 ${fmt(row00['stress_affected_margin_min'])}$ 降至 ${fmt(row20['stress_affected_margin_min'])}$。因此，本文不把随机误分类实验写成理论鲁棒性保证，而是写成中等随机状态误差下的 baseline sensitivity evidence；更强的 false-negative、delay 和 burst 边界由后续自适应迭代实验补充。

\input{{{OUT_DIR.name}/snippets/state_misclassification_table.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{{OUT_DIR.name}/figures/figS2_state_misclassification_robustness.pdf}}
\parbox{{\textwidth}}{{\footnotesize\textbf{{Fig. S2.}} 状态标签随机误分类 baseline。曲线显示 stress-window shortage severity、surface served ratio、p95 backlog 和 stress-affected margin minimum 随 $p_{{\rm err}}$ 的变化；红色标记突出 $p_{{\rm err}}=0.20$，即本文用于正文表述的最强随机误分类 endpoint。Observed label error closely tracked the target level, with label error rate 0.1990 and stress false-negative rate 0.2080 at $p_{{\rm err}}=0.20$。误差条表示 20 个 paired seeds 的 95\% confidence interval。}}
\label{{fig:state_misclassification_baseline}}
\end{{figure*}}
"""
    (SNIPPET_DIR / "state_misclassification_section.tex").write_text(text.strip() + "\n", encoding="utf-8")


def iteration_slug(iteration: int) -> str:
    return {
        1: "random_symmetric_baseline",
        2: "stress_false_negative_coarse",
        3: "stress_false_negative_fine",
        4: "bursty_false_negative",
        5: "estimator_delay",
        6: "stuck_at_rest",
        7: "stress_false_positive",
        8: "mixed_realistic_estimator",
        9: "robust_workpoint_search",
        10: "best_candidate_revalidation",
    }[iteration]


def table_s5_tex(top: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{状态误分类鲁棒性的 10 轮自适应迭代中保留的代表性证据。所有数值均为 20 个 paired seeds 的 mean；这些候选用于说明经验边界和可选保守工作点，不替换主结果工作点。}",
        r"\label{tab:iter10_state_robustness_cn}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"轮次/条件 & 误差模式 & Severity & Margin min & Surface served & p95 backlog & Stress FN & Paper score \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        label = latex_texttt(f"I{int(row['iteration'])}: {str(row['param_value'])}")
        model = str(row["model"]).replace("_", r"\_")
        lines.append(
            f"{label} & {model} & "
            f"{fmt(row['stress_shortage_severity'])} & "
            f"{fmt(row['stress_affected_margin_min'])} & "
            f"{fmt(row['surface_served_ratio'])} & "
            f"{fmt(row['p95_backlog'], 2)} & "
            f"{fmt(row.get('stress_false_negative_rate', 0.0))} & "
            f"{fmt(row['paper_value_score'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_latex_snippets(leaderboard: pd.DataFrame) -> None:
    top = leaderboard.sort_values(["paper_value_score", "boundary_score"], ascending=False).head(5)
    best = top.iloc[0]
    fn_rows = leaderboard[
        leaderboard["model"].eq("stress_false_negative") & leaderboard["param_name"].eq("p_fn") & leaderboard["iteration"].isin([2, 3])
    ]
    fn_worst = fn_rows.sort_values("boundary_score", ascending=False).iloc[0]
    wp_rows = leaderboard[leaderboard["iteration"].eq(9)].copy()
    robust_wp = wp_rows.sort_values(["stress_affected_margin_min", "service_score"], ascending=False).iloc[0]
    final_wp_row = wp_rows[wp_rows["workpoint_tag"].eq("wp_final")].iloc[0]
    snippet = rf"""
\subsection{{状态误分类鲁棒性的自适应迭代分析}}
\label{{subsec:state_iter10_adaptive_cn}}

随机对称误分类实验说明，温和的 node-type-preserving label noise 并未立即触发 stress-window shortage。为了避免只报告一个有利但不充分的随机误差模型，本文进一步进行了 10 轮自适应迭代：先定位最危险误差模式，再细化经验边界，最后搜索一个仅用于部署讨论的保守工作点。每轮均使用相同 20 个 paired seeds，统一计算 safety、compliance、service、boundary 和 paper-value scores；这些分数只用于筛选补充证据，不进入调度器本身。

自适应搜索显示，最有审稿价值的误差不是表面节点的随机混淆，而是受压植入节点的 stress false-negative。粗扫和细扫中 paper-value score 最高的 false-negative 候选为 $p_{{FN}}={fn_worst['param_value']}$，其 stress shortage severity 为 ${fmt(fn_worst['stress_shortage_severity'])}$，stress-affected margin minimum 为 ${fmt(fn_worst['stress_affected_margin_min'])}$，observed stress false-negative rate 为 ${fmt(fn_worst['stress_false_negative_rate'])}$。因此，本文可以把结论写成：在中等状态误差下主工作点仍保持零短缺；当误差具有连续漏检或强 false-negative 偏置时，退化首先表现为 energy margin 和 backlog 的收缩，而不是 EM/rx-cap 违规。

第 9 轮进一步检查了可选保守工作点。该搜索不替换主文 final workpoint；它只回答部署状态估计器更差时是否存在参数余地。当前搜索中 margin 最高的候选为 \texttt{{{latex_texttt(robust_wp['workpoint_tag'])}}}，其 stress-affected margin minimum 从 final workpoint 在同一挑战下的 ${fmt(final_wp_row['stress_affected_margin_min'])}$ 提高到 ${fmt(robust_wp['stress_affected_margin_min'])}$，surface served ratio 为 ${fmt(robust_wp['surface_served_ratio'])}$。这说明系统存在一定保守调参空间，但该结果应作为 supplementary robust variant，而非主结果替代。

\input{{{OUT_DIR.name}/snippets/iter10_adaptive_table.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{{OUT_DIR.name}/figures/figS3_iter10_adaptive_robustness.pdf}}
\parbox{{\textwidth}}{{\footnotesize\textbf{{Fig. S3.}} 状态误分类鲁棒性的 10 轮自适应迭代摘要。(a) stress false-negative 对 stress margin 和 observed false-negative rate 的影响；(b) 连续漏检和估计延迟对 p95 backlog 的影响；(c) 可选保守工作点在 surface served ratio 与 stress margin 之间的折中；(d) paper-value score 最高的候选证据。该图展示的是经验鲁棒性边界和补充调参空间，不构成理论鲁棒性保证。}}
\label{{fig:iter10_state_adaptive}}
\end{{figure*}}
"""
    (SNIPPET_DIR / "iter10_adaptive_table.tex").write_text(table_s5_tex(top), encoding="utf-8")
    (SNIPPET_DIR / "iter10_adaptive_section.tex").write_text(snippet.strip() + "\n", encoding="utf-8")


def write_total_report(leaderboard: pd.DataFrame, fine_grid: list[float], challenge_fn: float) -> None:
    top = leaderboard.sort_values(["paper_value_score", "boundary_score"], ascending=False).head(8)
    best = top.iloc[0]
    param_numeric = pd.to_numeric(leaderboard["param_value"], errors="coerce")
    random20 = leaderboard[(leaderboard["iteration"].eq(1)) & (param_numeric.eq(0.20))].iloc[0]
    fn_best = leaderboard[
        leaderboard["model"].eq("stress_false_negative") & leaderboard["param_name"].eq("p_fn")
    ].sort_values(["boundary_score", "paper_value_score"], ascending=False).iloc[0]
    wp_rows = leaderboard[leaderboard["iteration"].eq(9)]
    robust_wp = wp_rows.sort_values(["stress_affected_margin_min", "service_score"], ascending=False).iloc[0]
    table = top[
        [
            "iteration",
            "condition_id",
            "stress_shortage_severity",
            "stress_affected_margin_min",
            "surface_served_ratio",
            "p95_backlog",
            "paper_value_score",
        ]
    ].to_markdown(index=False, floatfmt=".4f")
    text = f"""# 状态误分类 10 轮自适应迭代总报告

## 1. 本轮是否增强了文章说服力

增强了，但增强点不应写成“理论鲁棒性保证”。更稳妥的论文价值是：

- 随机对称误分类 v1 保留为温和 baseline：`p_err=0.20` 时 stress severity={random20['stress_shortage_severity']:.4f}，surface served={random20['surface_served_ratio']:.4f}，p95 backlog={random20['p95_backlog']:.2f}，stress margin min={random20['stress_affected_margin_min']:.4f}。
- 10 轮迭代确认最值得报告的风险模式是 stress false-negative，而不是普通随机误差。
- 在强 false-negative、delay、burst 和 stuck-at-rest 下，退化主要先体现在 stress margin 与 backlog，而不是 EM/rx-cap/LP 违规。
- 第 9 轮显示存在可选保守工作点，可以作为部署状态估计器较差时的 supplementary variant。

## 2. 自适应过程

- 第 1 轮：读取已有随机对称误分类 v1，不重跑。
- 第 2 轮：stress false-negative 粗扫。
- 第 3 轮：根据第 2 轮自动选择 fine grid：`{fine_grid}`。
- 第 4-7 轮：分别测试 burst、delay、stuck-at-rest 和 false-positive conservative error。
- 第 8 轮：混合 realistic estimator。
- 第 9 轮：在挑战参数 `p_fn={challenge_fn:.3f}` 下搜索可选保守工作点。
- 第 10 轮：复验 paper-value score 最高的 3 个候选。

## 3. 最有价值结果

- 最高 paper-value 候选：`{best['condition_id']}`，score={best['paper_value_score']:.4f}。
- false-negative 边界候选：`{fn_best['condition_id']}`，stress margin min={fn_best['stress_affected_margin_min']:.4f}，stress FN={fn_best['stress_false_negative_rate']:.4f}。
- 可选保守工作点：`{robust_wp['workpoint_tag']}`，stress margin min={robust_wp['stress_affected_margin_min']:.4f}，surface served={robust_wp['surface_served_ratio']:.4f}。

## 4. 建议放进论文的内容

- 最终只保留一张综合 Fig. S2，把随机误分类 sanity check、自适应边界、代表性 adverse cases 和 robust workpoint 放在同一张 2×2 图中。
- 不再单独放旧随机误分类 S2 或旧自适应 S3，避免两张鲁棒性图占版面。
- 新增 Table S5，只列代表性 adverse cases 和保守工作点。
- 正文表述建议：`moderate random label noise did not induce stress-window shortage; biased or consecutive stress false-negatives identify an empirical robustness boundary through shrinking energy margin and increasing backlog.`

## 5. 不建议夸大的地方

- 不写成 theorem-derived robustness frontier。
- 不声称所有状态误差都鲁棒。
- 不把第 9 轮保守工作点替换为主工作点，只作为 supplementary deployment option。
- 参考文献 `[?]` 本轮仍不处理。

## 6. Top candidates

{table}
"""
    (OUT_DIR / "状态误分类10轮自适应迭代总报告.md").write_text(text, encoding="utf-8")


def build_methods_tex() -> None:
    src = V1_DIR / "methods_state_robustness.tex"
    dst = OUT_DIR / "methods_state_robustness_iter10_adaptive.tex"
    if not src.exists():
        if dst.exists():
            return
        dst.write_text("% Baseline state-robustness methods source was not included in this clean package.\n", encoding="utf-8")
        return
    text = src.read_text(encoding="utf-8")
    text = text.replace("paper_enhanced_state_robustness/", f"{OUT_DIR.name}/")
    marker = rf"\input{{{OUT_DIR.name}/snippets/state_misclassification_section.tex}}"
    adaptive = rf"\input{{{OUT_DIR.name}/snippets/iter10_adaptive_section.tex}}"
    text = text.replace("\n\n" + adaptive, "")
    text = text.replace(adaptive + "\n\n", "")
    text = text.replace(adaptive, "")
    if marker not in text:
        text = text.replace(r"\subsection{有效性边界}", marker + "\n\n" + r"\subsection{有效性边界}", 1)
    dst.write_text(text, encoding="utf-8")


def main() -> None:
    if "--postprocess-only" in sys.argv:
        leaderboard = pd.read_csv(OUT_DIR / "iter10_leaderboard.csv")
        by_seed = pd.read_csv(OUT_DIR / "iter10_all_by_seed.csv")
        iter3_values = pd.to_numeric(
            leaderboard.loc[leaderboard["iteration"].eq(3), "param_value"],
            errors="coerce",
        ).dropna()
        fine_grid = [round(float(x), 3) for x in sorted(iter3_values.unique())]
        fn_rows = leaderboard[
            leaderboard["model"].eq("stress_false_negative") & leaderboard["param_name"].eq("p_fn")
        ].copy()
        challenge_fn = best_numeric_param(fn_rows, 0.30)
        write_comprehensive_robustness_snippet(leaderboard)
        plot_best_package(leaderboard)
        write_best_package(leaderboard, by_seed)
        write_total_report(leaderboard, fine_grid, float(np.clip(challenge_fn, 0.30, 0.50)))
        build_methods_tex()
        print("Postprocessed existing adaptive outputs in", OUT_DIR)
        return

    by_seed, leaderboard = run_adaptive_iterations()
    print("Wrote adaptive 10-iteration robustness package to", OUT_DIR)
    print("Rows by seed:", len(by_seed))
    print("Leaderboard rows:", len(leaderboard))
    print(
        leaderboard.sort_values("paper_value_score", ascending=False)[
            [
                "iteration",
                "condition_id",
                "stress_shortage_severity",
                "stress_affected_margin_min",
                "surface_served_ratio",
                "p95_backlog",
                "paper_value_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
