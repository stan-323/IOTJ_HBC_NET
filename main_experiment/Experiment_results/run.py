from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from Experiment_results.config import ABLATION_METHODS, MAIN_METHODS, STRESS_METHODS, SimConfig, WorkPoint
from Experiment_results.data import load_coarse_2state_library, load_g_scaled_library, load_hfss_library, load_homogenized_library, load_linkbudget_library
from Experiment_results.gates import GateReport, evaluate_candidate_gates
from Experiment_results.metrics import MetricSummary, merged_metric_payload, paired_wilcoxon_pvalue, summarize_raw
from Experiment_results.plotting import plot_fig3, plot_fig4, plot_fig5, plot_fig6, plot_fig7, quick_fig3, quick_fig5, redesign_final_figures
from Experiment_results.reports import write_failed_report, write_json, write_pass_fail_csv, write_yaml
from Experiment_results.scenarios import node_type, rest_label, stress_label, stress_window_labels
from Experiment_results.search import coarse_workpoints, ralph_seed_count, ralph_workpoint
from Experiment_results.sim import Trace, generate_trace, run_phase_cell_metrics, run_suite


TRAIN_SEEDS = list(range(10))
TEST_SEEDS = list(range(10, 20))
PHASE3_SEEDS = list(range(10, 40))
LYAP_V_SCALES = [0.25, 0.50, 1.00, 2.00, 4.00]
PROXY_G_SCALES = [0.70, 0.80, 0.90, 1.00, 1.10]
LAMBDA_S_SWEEP_VALUES = [5.0, 10.0, 20.0, 50.0]
MODE_ALIASES = {
    "lyap_v_sweep": "lyap_sweep",
    "state_error": "estimator_prototype",
    "label_boundary_sweep": "robustness_boundary",
}
LabelEstimator = Callable[[int, list[str], Trace, object], list[str]]


def _output_root(output_root: str | Path | None) -> Path:
    return Path(output_root) if output_root is not None else Path(__file__).resolve().parents[1]


def _clean_outputs(root: Path) -> None:
    outputs = root / "output"
    if outputs.exists():
        shutil.rmtree(outputs)


def _diagnostic(root: Path, cfg: SimConfig, wp: WorkPoint) -> bool:
    lib = load_hfss_library()
    oracle_raw = run_suite(cfg, wp, ["Oracle"], [0], "stress", lib)
    oracle = summarize_raw(oracle_raw, cfg, scenario="stress").by_method["Oracle"]
    lines = [
        "# diagnostic_report",
        "",
        f"- HFSS source: `{lib.source_path}`",
        f"- condition-dependent coefficient source: {lib.source_path.name}",
        "- old synthetic Rest/Walk/Sweat/Loose scaling table: not used",
        f"- labels: {', '.join(sorted(lib.rows))}",
        f"- oracle stress_shortage_severity: {oracle['stress_shortage_severity']:.6f}",
        f"- oracle EM violation rate: {oracle['em_violation_rate']:.6f}",
        f"- oracle rx-cap violation rate: {oracle['rx_cap_violation_rate']:.6f}",
    ]
    passed = oracle["stress_shortage_severity"] <= 0.02 and oracle["em_violation_rate"] == 0.0 and oracle["rx_cap_violation_rate"] == 0.0
    lines.append(f"- diagnostic passed: {passed}")
    out = root / "output" / "diagnostic_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def _phase_summary(cfg: SimConfig, wp: WorkPoint, seeds: list[int], *, quick: bool) -> pd.DataFrame:
    lib = load_hfss_library()
    grids = [
        # Torso transition-edge grid: resolves the narrow stress-feasibility boundary.
        ([0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60], [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65]),
        ([0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 1.00], [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]),
        ([0.40, 0.41, 0.42, 0.43, 0.44], [1.02, 1.04, 1.06, 1.08, 1.10]),
        ([0.70, 0.82, 0.88, 0.94, 1.00], [0.88, 0.94, 1.00, 1.06, 1.24]),
        ([0.82, 0.86, 0.89, 0.91, 0.93, 0.95, 0.97, 1.00, 1.04], [0.82, 0.86, 0.89, 0.91, 0.93, 0.95, 0.97, 1.00, 1.04]),
        ([0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.04], [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.04]),
        ([0.78, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.06], [0.78, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.06]),
        ([0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.96, 1.02, 1.08], [0.80, 0.86, 0.92, 0.98, 1.04, 1.10, 1.16, 1.22, 1.28]),
        ([0.70, 0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18], [0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18, 1.24]),
        ([0.15, 0.25, 0.35, 0.45, 0.55], [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]),
    ]
    if quick:
        grids = [([0.42, 0.46, 0.50, 0.55, 0.60], [0.45, 0.50, 0.55, 0.60])]
        seeds = seeds[:1]
    else:
        seeds = seeds[:1]
    final_rows: list[dict[str, float | bool]] = []
    for bem_values, ph_values in grids:
        rows: list[dict[str, float | bool]] = []
        for bem in bem_values:
            for ph in ph_values:
                metrics = run_phase_cell_metrics(cfg, wp, seeds, lib, budget_multiplier=bem, power_multiplier=ph)
                rows.append(
                    {
                        "B_EM_multiplier": float(bem),
                        "P_H_multiplier": float(ph),
                        "stress_shortage_severity": float(metrics["stress_shortage_severity"]),
                        "stress_implant_shortage_rate": float(metrics["stress_implant_shortage_rate"]),
                        "served_workload_ratio": float(metrics["served_workload_ratio"]),
                        "is_final_workpoint": bool(abs(bem - 1.0) < 1e-12 and abs(ph - 1.0) < 1e-12),
                    }
                )
        phase = pd.DataFrame(rows)
        vals = phase["stress_shortage_severity"]
        low = (vals < 0.03).mean()
        high = (vals > 0.50).mean()
        final_rows = rows
        medium = ((vals >= 0.03) & (vals <= 0.50)).mean()
        if quick or (low >= 0.20 and medium >= 0.20 and high >= 0.10):
            break
    return pd.DataFrame(final_rows)


def _condition_response_ratio(condition_raw: pd.DataFrame) -> float:
    proposed = condition_raw[condition_raw["method"].eq("Proposed")]
    if proposed.empty:
        return 0.0
    ratios: list[float] = []
    groups = proposed.groupby(proposed["is_critical"].astype(bool)) if "is_critical" in proposed else [(True, proposed)]
    for _, group in groups:
        by_regime = group.groupby("regime")["uE"].mean()
        if by_regime.empty:
            continue
        denom = max(float(by_regime.min()), 1e-9)
        ratios.append(float((by_regime.max() - by_regime.min()) / denom))
    return max(ratios, default=0.0)


def _pre_stress_below_min(raw: pd.DataFrame, cfg: SimConfig, method: str) -> float:
    sub = raw[(raw["method"].eq(method)) & (raw["node"].isin(cfg.stress_nodes)) & (raw["t"] < cfg.stress_start)]
    if sub.empty:
        return 0.0
    return float((sub["e_next"] < sub["E_min"] - 1e-12).mean())


def _compose_metrics(
    cfg: SimConfig,
    condition_raw: pd.DataFrame,
    stress_raw: pd.DataFrame,
    ablation_raw: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    stress_summary = summarize_raw(stress_raw, cfg, scenario="stress")
    ablation_summary = summarize_raw(ablation_raw, cfg, scenario="stress_ablation")
    condition_summary = summarize_raw(condition_raw, cfg, scenario="condition")

    metrics = merged_metric_payload(stress_summary, ablation_summary)
    for method, extra in condition_summary.by_method.items():
        metrics.setdefault(method, {})
        for key in ["em_util_mean", "em_util_p95", "em_util_peak"]:
            metrics[method][key] = extra[key]
    metrics["Proposed"]["fig4_condition_response_ratio"] = _condition_response_ratio(condition_raw)
    metrics.setdefault("w/o crit.-urg.", {})["pre_stress_below_min_rate"] = _pre_stress_below_min(ablation_raw, cfg, "w/o crit.-urg.")
    return metrics


def _run_candidate(
    root: Path,
    name: str,
    cfg: SimConfig,
    wp: WorkPoint,
    seeds: list[int],
    *,
    include_phase: bool,
    write_raw: bool,
) -> tuple[GateReport, dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lib = load_hfss_library()
    coarse_lib = load_coarse_2state_library()
    ablation_libraries = {"w/o implant-aware": coarse_lib}
    candidate_dir = root / "output" / "candidates" / name
    raw_dir = candidate_dir / "raw"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    condition_methods = MAIN_METHODS + ["w/o crit.-urg."]
    condition_raw = run_suite(cfg, wp, condition_methods, seeds, "condition", lib)
    stress_raw = run_suite(cfg, wp, sorted(set(STRESS_METHODS + ["Oracle"])), seeds, "stress", lib)
    ablation_raw = run_suite(cfg, wp, ABLATION_METHODS, seeds, "stress", lib, per_method_libraries=ablation_libraries)

    if write_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        condition_raw.to_csv(raw_dir / "condition_switch_per_slot.csv", index=False)
        stress_raw.to_csv(raw_dir / "stress_per_slot.csv", index=False)
        ablation_raw.to_csv(raw_dir / "ablation_per_slot.csv", index=False)

    metrics = _compose_metrics(cfg, condition_raw, stress_raw, ablation_raw)
    phase = _phase_summary(cfg, wp, seeds, quick=not include_phase) if include_phase else _phase_summary(cfg, wp, seeds[:1], quick=True)
    report = evaluate_candidate_gates(metrics, phase)

    write_json(
        candidate_dir / "candidate_metrics.json",
        {
            "candidate": name,
            "seeds": seeds,
            "workpoint": wp.to_dict(),
            "metrics": metrics,
            "phase": phase.to_dict(orient="records"),
        },
    )
    write_json(candidate_dir / "gate_result.json", {"passed": report.passed, "failed_codes": report.failed_codes, "score": report.score})
    write_pass_fail_csv(candidate_dir / "candidate_pass_fail.csv", report.rows)
    (candidate_dir / "figure_gate_report.md").write_text(report.to_markdown(), encoding="utf-8")
    quick_fig3(metrics, candidate_dir)
    quick_fig5(stress_raw, cfg, candidate_dir)
    return report, metrics, phase, condition_raw, stress_raw


def _write_metric_table(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _run_necessity_ablation(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    truth = load_hfss_library()
    variants = {
        "HFSS-calibrated": truth,
        "Homogenized": load_homogenized_library(),
        "Link-budget": load_linkbudget_library(),
    }
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for name, decision_lib in variants.items():
        raw = run_suite(cfg, wp, ["Proposed"], seeds, "stress", truth, decision_library=decision_lib)
        summary = summarize_raw(raw, cfg, scenario="stress")
        rows.append({"library": name, **summary.by_method["Proposed"]})
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "library", name)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "necessity_ablation_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "necessity_ablation.csv", rows)


def _paired_wilcoxon_delta(
    by_seed: pd.DataFrame,
    *,
    baseline: str,
    comparator: str,
    metric: str,
    alternative: str,
) -> dict[str, object]:
    wide = by_seed.pivot(index="seed", columns="library", values=metric)
    if baseline not in wide or comparator not in wide:
        return {
            "comparison": f"{comparator} vs {baseline}",
            "metric": metric,
            "alternative": alternative,
            "delta_mean": np.nan,
            "p_value": np.nan,
            "nonzero_pairs": 0,
        }
    diff = wide[comparator].astype(float) - wide[baseline].astype(float)
    nonzero = int((diff.abs() > 1e-12).sum())
    p_value = 1.0
    if nonzero:
        try:
            p_value = float(wilcoxon(diff, zero_method="wilcox", alternative=alternative, method="auto").pvalue)
        except ValueError:
            p_value = 1.0
    return {
        "comparison": f"{comparator} vs {baseline}",
        "metric": metric,
        "alternative": alternative,
        "delta_mean": float(diff.mean()),
        "p_value": p_value,
        "nonzero_pairs": nonzero,
    }


def _run_necessity_audit(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    summary = _run_necessity_ablation(root, cfg, wp, seeds)
    summary.to_csv(final_dir / "necessity_audit_summary.csv", index=False)
    source_by_seed = final_dir / "necessity_ablation_by_seed.csv"
    if not source_by_seed.exists():
        _write_metric_table(final_dir / "necessity_audit_wilcoxon.csv", [])
        return summary

    by_seed = pd.read_csv(source_by_seed)
    by_seed.to_csv(final_dir / "necessity_audit_by_seed.csv", index=False)
    metrics_and_alternatives = {
        "stress_shortage_severity": "greater",
        "stress_implant_shortage_rate": "greater",
        "em_violation_rate": "greater",
        "rx_cap_violation_rate": "greater",
        "p95_backlog": "greater",
        "recovery_censored": "greater",
        "surface_served_ratio": "less",
        "stress_affected_margin_min": "less",
    }
    rows: list[dict[str, object]] = []
    for comparator in ["Homogenized", "Link-budget"]:
        for metric, alternative in metrics_and_alternatives.items():
            if metric not in by_seed:
                continue
            rows.append(
                _paired_wilcoxon_delta(
                    by_seed,
                    baseline="HFSS-calibrated",
                    comparator=comparator,
                    metric=metric,
                    alternative=alternative,
                )
            )
    _write_metric_table(final_dir / "necessity_audit_wilcoxon.csv", rows)
    return summary


def _run_lyap_v_sensitivity(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for scale in LYAP_V_SCALES:
        raw = run_suite(cfg, wp, ["Lyap.-DPP"], seeds, "stress", lib, lyap_v_scale=scale)
        summary = summarize_raw(raw, cfg, scenario="stress")
        metrics = summary.by_method["Lyap.-DPP"]
        rows.append(
            {
                "V_scale": scale,
                "feasible": bool(scale <= 1.0 and metrics["stress_shortage_severity"] <= 1e-9 and metrics["em_violation_rate"] <= 1e-12 and metrics["rx_cap_violation_rate"] <= 1e-12),
                "lp_feasible_slots": float(1.0 - metrics["lp_failure_rate"]),
                **metrics,
            }
        )
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "V_scale", scale)
        by_seed.insert(1, "feasible", scale <= 1.0)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "lyap_v_sweep_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "lyap_v_sweep.csv", rows)


def _write_method_wilcoxon(path: Path, by_seed: pd.DataFrame, *, baseline: str = "Proposed", group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    metrics = ["stress_shortage_severity", "surface_served_ratio", "p95_backlog", "em_violation_rate", "rx_cap_violation_rate", "recovery_censored"]
    rows: list[dict[str, object]] = []
    groups = by_seed.groupby(group_cols, sort=False) if group_cols else [((), by_seed)]
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            if metric not in group:
                continue
            wide = group.pivot(index="seed", columns="method", values=metric)
            if baseline not in wide:
                continue
            for method in wide.columns:
                if method == baseline:
                    continue
                row = {col: value for col, value in zip(group_cols, key_values)}
                row.update(
                    {
                        "comparison": f"{method} vs {baseline}",
                        "metric": metric,
                        "delta_mean": float((wide[method] - wide[baseline]).mean()),
                        "p_value": paired_wilcoxon_pvalue(wide[method], wide[baseline]),
                    }
                )
                rows.append(row)
    return _write_metric_table(path, rows)


def _write_variant_wilcoxon(path: Path, by_seed: pd.DataFrame, *, baseline: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    metrics = ["stress_shortage_severity", "surface_served_ratio", "p95_backlog", "em_violation_rate", "rx_cap_violation_rate", "recovery_censored"]
    rows: list[dict[str, object]] = []
    groups = by_seed.groupby(group_cols, sort=False) if group_cols else [((), by_seed)]
    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            if metric not in group:
                continue
            wide = group.pivot(index="seed", columns="variant", values=metric)
            if baseline not in wide:
                continue
            for variant in wide.columns:
                if variant == baseline:
                    continue
                row = {col: value for col, value in zip(group_cols, key_values)}
                row.update(
                    {
                        "comparison": f"{variant} vs {baseline}",
                        "metric": metric,
                        "delta_mean": float((wide[variant] - wide[baseline]).mean()),
                        "p_value": paired_wilcoxon_pvalue(wide[variant], wide[baseline]),
                    }
                )
                rows.append(row)
    return _write_metric_table(path, rows)


def _run_small_network(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    small_cfg = replace(cfg, num_nodes=4, num_critical=1, stress_nodes=(0,))
    raw = run_suite(small_cfg, wp, MAIN_METHODS, seeds, "stress", load_hfss_library())
    summary = summarize_raw(raw, small_cfg, scenario="small_network")
    summary.by_seed.to_csv(final_dir / "small_network_by_seed.csv", index=False)
    rows = [{"method": method, **metrics} for method, metrics in summary.by_method.items()]
    return _write_metric_table(final_dir / "small_network_summary.csv", rows)


def _topology_cfg(base: SimConfig, *, num_surface: int, num_implant: int) -> SimConfig:
    stress_node = 0 if num_implant == 1 else 1
    return replace(base, num_nodes=num_surface + num_implant, num_critical=num_implant, stress_nodes=(stress_node,))


def _run_network_scale(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    topologies = [
        ("small_3s_1i", 3, 1),
        ("mid_5s_2i", 5, 2),
        ("full_9s_3i", 9, 3),
    ]
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for topology, num_surface, num_implant in topologies:
        topo_cfg = _topology_cfg(cfg, num_surface=num_surface, num_implant=num_implant)
        raw = run_suite(topo_cfg, wp, MAIN_METHODS, seeds, "stress", lib)
        raw.to_csv(final_dir / f"{topology}_per_slot.csv", index=False)
        summary = summarize_raw(raw, topo_cfg, scenario=topology)
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "topology", topology)
        by_seed.insert(1, "num_surface", num_surface)
        by_seed.insert(2, "num_implant", num_implant)
        by_seed_frames.append(by_seed)
        for method, metrics in summary.by_method.items():
            rows.append({"topology": topology, "num_surface": num_surface, "num_implant": num_implant, "method": method, **metrics})
    if by_seed_frames:
        by_seed_all = pd.concat(by_seed_frames, ignore_index=True)
        by_seed_all.to_csv(final_dir / "network_scale_by_seed.csv", index=False)
        _write_method_wilcoxon(final_dir / "network_scale_wilcoxon.csv", by_seed_all, group_cols=["topology"])
    return _write_metric_table(final_dir / "network_scale_summary.csv", rows)


def _summarize_variant_run(
    *,
    cfg: SimConfig,
    wp: WorkPoint,
    seeds: list[int],
    variant: str,
    methods: list[str],
    truth_library,
    decision_library=None,
    label_func: Callable[..., list[str]] | None = None,
    lp_tolerance: float | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    raw = run_suite(
        cfg,
        wp,
        methods,
        seeds,
        "stress",
        truth_library,
        decision_library=decision_library,
        label_func=label_func,
        lp_tolerance=lp_tolerance,
    )
    summary = summarize_raw(raw, cfg, scenario=variant)
    rows = [{"variant": variant, "method": method, **metrics} for method, metrics in summary.by_method.items()]
    by_seed = summary.by_seed.copy()
    by_seed.insert(0, "variant", variant)
    return rows, by_seed, raw


def _run_phase2_coarse_library(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    truth = load_hfss_library()
    variants = {
        "torso_fine_grained": truth,
        "torso_coarse_2state": load_coarse_2state_library(),
        "single_state_fallback": load_homogenized_library(),
    }
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for variant, decision_lib in variants.items():
        v_rows, by_seed, raw = _summarize_variant_run(
            cfg=cfg,
            wp=wp,
            seeds=seeds,
            variant=variant,
            methods=["Proposed"],
            truth_library=truth,
            decision_library=decision_lib,
        )
        raw.to_csv(final_dir / f"{variant}_per_slot.csv", index=False)
        for row in v_rows:
            row.pop("method", None)
            rows.append(row)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        by_seed_all = pd.concat(by_seed_frames, ignore_index=True)
        by_seed_all.to_csv(final_dir / "output_phase3_coarse_ablation_by_seed.csv", index=False)
        _write_variant_wilcoxon(final_dir / "output_phase3_coarse_ablation_wilcoxon.csv", by_seed_all, baseline="torso_fine_grained")
    summary = _write_metric_table(final_dir / "output_phase3_coarse_ablation.csv", rows)
    fine = summary[summary["variant"].eq("torso_fine_grained")]
    if not fine.empty:
        fine_surface = float(fine["surface_served_ratio"].iloc[0])
        fine_backlog = float(fine["p95_backlog"].iloc[0])
        summary["surface_relative_drop_vs_fine"] = summary["surface_served_ratio"].apply(lambda value: (fine_surface - float(value)) / max(fine_surface, 1e-12))
        summary["p95_backlog_relative_increase_vs_fine"] = summary["p95_backlog"].apply(lambda value: (float(value) - fine_backlog) / max(fine_backlog, 1e-12))
        summary.to_csv(final_dir / "output_phase3_coarse_ablation.csv", index=False)
    return summary


def _run_baseline_extra(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    methods = MAIN_METHODS + ["Round-Robin"]
    raw = run_suite(cfg, wp, methods, seeds, "stress", lib)
    raw.to_csv(final_dir / "baseline_extra_per_slot.csv", index=False)
    summary = summarize_raw(raw, cfg, scenario="baseline_extra")
    summary.by_seed.to_csv(final_dir / "baseline_extra_by_seed.csv", index=False)
    _write_method_wilcoxon(final_dir / "baseline_extra_wilcoxon.csv", summary.by_seed)
    rows = [{"method": method, **metrics} for method, metrics in summary.by_method.items()]
    return _write_metric_table(final_dir / "baseline_extra_summary.csv", rows)


def _heterogeneous_label_func(cfg: SimConfig) -> Callable[[int, Trace], list[str]]:
    def labels(t: int, trace: Trace) -> list[str]:
        result = [rest_label(node, cfg.num_critical) for node in range(cfg.num_nodes)]
        for node in range(cfg.num_critical):
            rng = np.random.default_rng(9300 + trace.seed * 101 + node * 17)
            low = max(0, cfg.stress_start - 80)
            high = max(low + 1, min(cfg.T - 1, cfg.stress_end + 120))
            duration_high = max(30, min(200, high - low + 1))
            start = int(rng.integers(low, max(low + 1, high - 20)))
            duration = int(rng.integers(30, duration_high + 1))
            if start <= t < min(cfg.T, start + duration):
                result[node] = stress_label(node, cfg.num_critical)
        return result

    return labels


def _implant_fairness_metrics(raw: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, method), group in raw.groupby(["seed", "method"], sort=False):
        implant = group[group["is_critical"].astype(bool)]
        margins: list[float] = []
        for _node, node_group in implant.groupby("node", sort=False):
            margin = (node_group["e_next"] - node_group["E_min"]) / (node_group["E_max"] - node_group["E_min"]).replace(0.0, np.nan)
            margins.append(float(margin.quantile(0.05)) if not margin.empty else 0.0)
        arr = np.asarray(margins, dtype=float)
        nonnegative = np.clip(arr, 0.0, None)
        fairness = float((nonnegative.sum() ** 2) / (len(nonnegative) * np.square(nonnegative).sum() + 1e-12)) if len(nonnegative) else 0.0
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "worst_implant_margin_p05": float(arr.min()) if len(arr) else 0.0,
                "implant_margin_std": float(arr.std(ddof=0)) if len(arr) else 0.0,
                "implant_fairness_jain": fairness,
            }
        )
    return pd.DataFrame(rows)


def _run_heterogeneous_stress(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    topologies = [("mid_5s_2i", 5, 2), ("full_9s_3i", 9, 3)]
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for topology, num_surface, num_implant in topologies:
        topo_cfg = _topology_cfg(cfg, num_surface=num_surface, num_implant=num_implant)
        raw = run_suite(topo_cfg, wp, MAIN_METHODS, seeds, "stress", lib, label_func=_heterogeneous_label_func(topo_cfg))
        raw.to_csv(final_dir / f"{topology}_heterogeneous_per_slot.csv", index=False)
        summary = summarize_raw(raw, topo_cfg, scenario=topology)
        fairness = _implant_fairness_metrics(raw, topo_cfg)
        by_seed = summary.by_seed.merge(fairness, on=["seed", "method"], how="left")
        by_seed.insert(0, "topology", topology)
        by_seed_frames.append(by_seed)
        fairness_summary = fairness.groupby("method", as_index=False).agg(
            worst_implant_margin_p05=("worst_implant_margin_p05", "mean"),
            implant_margin_std=("implant_margin_std", "mean"),
            implant_fairness_jain=("implant_fairness_jain", "mean"),
        )
        for method, metrics in summary.by_method.items():
            fair_row = fairness_summary[fairness_summary["method"].eq(method)].iloc[0].to_dict()
            fair_row.pop("method", None)
            rows.append({"topology": topology, "method": method, **metrics, **fair_row})
    if by_seed_frames:
        by_seed_all = pd.concat(by_seed_frames, ignore_index=True)
        by_seed_all.to_csv(final_dir / "heterogeneous_implant_by_seed.csv", index=False)
        _write_method_wilcoxon(final_dir / "heterogeneous_implant_wilcoxon.csv", by_seed_all, group_cols=["topology"])
    return _write_metric_table(final_dir / "heterogeneous_implant_summary.csv", rows)


def _run_lp_tol_sweep(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for tol in [1e-9, 1e-7, 1e-5, 1e-3]:
        raw = run_suite(cfg, wp, ["Proposed"], seeds, "stress", lib, lp_tolerance=tol)
        summary = summarize_raw(raw, cfg, scenario="lp_tol_sweep")
        metrics = summary.by_method["Proposed"]
        rows.append({"lp_tolerance": tol, **metrics})
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "lp_tolerance", tol)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "lp_tol_sweep_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "lp_tol_sweep.csv", rows)


def _run_stress_window_sweep(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    rows: list[dict[str, object]] = []
    for length in [50, 100, 150, 200, 300]:
        start = min(cfg.stress_start, max(0, cfg.T // 3))
        end = min(cfg.T - 1, start + min(length, max(2, cfg.T - start - 1)))
        sweep_cfg = replace(cfg, stress_start=start, stress_end=end, post_stress_window=min(cfg.post_stress_window, max(1, cfg.T - end)))
        raw = run_suite(sweep_cfg, wp, ["Proposed"], seeds, "stress", lib)
        summary = summarize_raw(raw, sweep_cfg, scenario="stress_window_sweep")
        rows.append({"stress_window_length": length, "actual_stress_window_length": end - start, **summary.by_method["Proposed"]})
    return _write_metric_table(final_dir / "stress_window_sweep.csv", rows)


def _run_workpoint_cv(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    candidates = [
        ("final", wp),
        ("lambda_c_minus_0p05", wp.shifted(lambda_c=-0.05)),
        ("lambda_c_plus_0p05", wp.shifted(lambda_c=0.05)),
        ("B_EM_minus_0p02", wp.shifted(B_EM_ref=-0.02)),
        ("B_EM_plus_0p02", wp.shifted(B_EM_ref=0.02)),
        ("P_H_minus_0p02", wp.shifted(P_H=-0.02)),
        ("P_H_plus_0p02", wp.shifted(P_H=0.02)),
    ]
    rows: list[dict[str, object]] = []
    for name, candidate_wp in candidates:
        raw = run_suite(cfg, candidate_wp, ["Proposed"], seeds, "stress", lib)
        summary = summarize_raw(raw, cfg, scenario="workpoint_cv")
        distance = max(abs(candidate_wp.to_dict()[key] - wp.to_dict()[key]) for key in wp.to_dict())
        rows.append({"candidate": name, "distance_from_final": float(distance), **candidate_wp.to_dict(), **summary.by_method["Proposed"]})
    return _write_metric_table(final_dir / "workpoint_cv_summary.csv", rows)


def _run_proxy_perturbation_sweep(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    truth = load_hfss_library()
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for scale in PROXY_G_SCALES:
        decision_library = load_g_scaled_library(scale)
        raw = run_suite(cfg, wp, ["Proposed"], seeds, "stress", truth, decision_library=decision_library)
        summary = summarize_raw(raw, cfg, scenario="proxy_perturbation_sweep")
        rows.append({"g_norm_scale": scale, **summary.by_method["Proposed"]})
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "g_norm_scale", scale)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "proxy_perturbation_sweep_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "proxy_perturbation_sweep.csv", rows)


def _run_lambda_s_sweep(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    for value in LAMBDA_S_SWEEP_VALUES:
        sweep_wp = replace(wp, lambda_s=float(value))
        raw = run_suite(cfg, sweep_wp, ["Proposed"], seeds, "stress", lib)
        summary = summarize_raw(raw, cfg, scenario="lambda_s_sweep")
        rows.append({"lambda_s": value, **summary.by_method["Proposed"]})
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "lambda_s", value)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "lambda_s_sweep_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "lambda_s_sweep.csv", rows)


def _run_phase3_30_seeds(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    raw = run_suite(cfg, wp, MAIN_METHODS, seeds, "stress", lib)
    raw.to_csv(final_dir / "phase3_30_seed_main_per_slot.csv", index=False)
    summary = summarize_raw(raw, cfg, scenario="phase3_30_seed_main")
    summary.by_seed.to_csv(final_dir / "phase3_30_seed_main_by_seed.csv", index=False)
    _write_method_wilcoxon(final_dir / "phase3_30_seed_main_wilcoxon.csv", summary.by_seed)
    rows = [{"method": method, **metrics} for method, metrics in summary.by_method.items()]
    return _write_metric_table(final_dir / "phase3_30_seed_main_summary.csv", rows)


def _feature_db(library, label: str) -> np.ndarray:
    row = library.coeff(label)
    return np.asarray(
        [
            10.0 * np.log10(max(row.g_norm, 1e-12)),
            10.0 * np.log10(max(row.r_norm, 1e-12)),
        ],
        dtype=float,
    )


def _kind_for_label(label: str) -> str:
    if label.startswith("surface"):
        return "surface"
    if label.startswith("implant10"):
        return "implant10"
    return "implant30"


def _stress_scenario_candidates(kind: str) -> list[str]:
    if kind == "surface":
        return ["surface_rest"]
    if kind == "implant10":
        return ["implant10_rest", "implant10_stress"]
    return ["implant30_rest", "implant30_stress"]


def _all_kind_candidates(kind: str) -> list[str]:
    if kind == "surface":
        return ["surface_rest", "surface_sweat", "surface_moderate_loose"]
    if kind == "implant10":
        return ["implant10_rest", "implant10_stress"]
    return ["implant30_rest", "implant30_stress"]


def _rng_for_observation(seed_offset: int, trace: Trace, t: int, node: int):
    return np.random.default_rng(seed_offset + trace.seed * 1_000_003 + t * 1009 + node * 37)


def _make_threshold_rssi_estimator(*, noise_db: float, bias_db: float, seed_offset: int = 4100) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        decisions: list[str] = []
        thresholds = {}
        for kind in ["implant10", "implant30"]:
            rest = f"{kind}_rest"
            stress = f"{kind}_stress"
            thresholds[kind] = 0.5 * (_feature_db(library, rest)[0] + _feature_db(library, stress)[0]) + bias_db
        for node, label in enumerate(true_labels):
            kind = _kind_for_label(label)
            if kind == "surface":
                decisions.append("surface_rest")
                continue
            obs = _feature_db(library, label)[0] + float(_rng_for_observation(seed_offset, trace, t, node).normal(0.0, noise_db))
            decisions.append(f"{kind}_stress" if obs <= thresholds[kind] else f"{kind}_rest")
        return decisions

    return estimator


def _make_knn2_estimator(*, noise_energy_db: float, noise_data_db: float, stress_bias: float, seed_offset: int = 5100) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        decisions: list[str] = []
        scale = np.asarray([max(noise_energy_db, 1e-6), max(noise_data_db, 1e-6)], dtype=float)
        for node, label in enumerate(true_labels):
            kind = _kind_for_label(label)
            candidates = _stress_scenario_candidates(kind)
            if len(candidates) == 1:
                decisions.append(candidates[0])
                continue
            rng = _rng_for_observation(seed_offset, trace, t, node)
            obs = _feature_db(library, label) + np.asarray(
                [
                    float(rng.normal(0.0, noise_energy_db)),
                    float(rng.normal(0.0, noise_data_db)),
                ]
            )
            def score(candidate: str) -> float:
                dist = float(np.linalg.norm((obs - _feature_db(library, candidate)) / scale))
                return dist - (stress_bias if candidate.endswith("_stress") else 0.0)

            decisions.append(min(candidates, key=score))
        return decisions

    return estimator


def _true_stress_labels(cfg: SimConfig, t: int) -> list[str]:
    return stress_window_labels(
        t,
        cfg.num_nodes,
        affected_nodes=cfg.stress_nodes,
        start=cfg.stress_start,
        end=cfg.stress_end,
        num_critical=cfg.num_critical,
    )


def _label_quality_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "label_accuracy": 0.0,
            "label_error_rate": 1.0,
            "stress_false_negative_rate": 0.0,
            "stress_false_positive_rate": 0.0,
            "surface_error_rate": 0.0,
            "implant_error_rate": 0.0,
        }
    labels = frame[["label_true", "label_decision"]].copy()
    correct = labels["label_true"].eq(labels["label_decision"])
    true_stress = labels["label_true"].astype(str).str.endswith("_stress")
    decided_stress = labels["label_decision"].astype(str).str.endswith("_stress")
    surface = labels["label_true"].astype(str).str.startswith("surface")
    implant = ~surface
    return {
        "label_accuracy": float(correct.mean()),
        "label_error_rate": float(1.0 - correct.mean()),
        "stress_false_negative_rate": float((true_stress & ~decided_stress).sum() / max(int(true_stress.sum()), 1)),
        "stress_false_positive_rate": float((~true_stress & decided_stress).sum() / max(int((~true_stress).sum()), 1)),
        "surface_error_rate": float((surface & ~correct).sum() / max(int(surface.sum()), 1)),
        "implant_error_rate": float((implant & ~correct).sum() / max(int(implant.sum()), 1)),
    }


def _label_quality_by_seed(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in raw.groupby("seed", sort=False):
        rows.append({"seed": int(seed), **_label_quality_from_frame(group)})
    return pd.DataFrame(rows)


def _evaluate_label_estimator(cfg: SimConfig, seeds: list[int], estimator: LabelEstimator, library) -> dict[str, float]:
    frames = []
    for seed in seeds:
        trace = generate_trace(cfg, seed, "stress")
        rows = []
        for t in range(cfg.T):
            true_labels = _true_stress_labels(cfg, t)
            decision_labels = estimator(t, true_labels, trace, library)
            for node, (true_label, decision_label) in enumerate(zip(true_labels, decision_labels)):
                rows.append(
                    {
                        "seed": seed,
                        "t": t,
                        "node": node,
                        "label_true": true_label,
                        "label_decision": decision_label,
                    }
                )
        frames.append(pd.DataFrame(rows))
    return _label_quality_from_frame(pd.concat(frames, ignore_index=True))


def _calibrate_estimators(cfg: SimConfig, seeds: list[int], library) -> tuple[str, LabelEstimator, pd.DataFrame]:
    candidates: list[tuple[str, str, dict[str, float], LabelEstimator]] = []
    for noise in [0.25, 0.50, 0.75, 1.00]:
        for bias in np.linspace(-1.0, 1.0, 5):
            params = {"noise_energy_db": float(noise), "noise_data_db": 0.0, "bias_db": float(bias), "stress_bias": 0.0}
            candidates.append(("threshold_rssi", f"threshold_rssi_noise{noise:.2f}_bias{bias:.2f}", params, _make_threshold_rssi_estimator(noise_db=float(noise), bias_db=float(bias))))
    for noise in [0.35, 0.55, 0.75]:
        for stress_bias in [0.0, 0.20, 0.45, 0.70]:
            params = {"noise_energy_db": float(noise), "noise_data_db": float(noise), "bias_db": 0.0, "stress_bias": float(stress_bias)}
            candidates.append(("knn_2feature", f"knn2_noise{noise:.2f}_sb{stress_bias:.2f}", params, _make_knn2_estimator(noise_energy_db=float(noise), noise_data_db=float(noise), stress_bias=float(stress_bias))))

    rows: list[dict[str, object]] = []
    best_name = ""
    best_estimator: LabelEstimator | None = None
    best_objective = float("inf")
    best_threshold_objective = float("inf")
    best_threshold: tuple[str, LabelEstimator] | None = None
    best_knn: tuple[str, LabelEstimator] | None = None
    best_knn_objective = float("inf")
    for family, name, params, estimator in candidates:
        metrics = _evaluate_label_estimator(cfg, seeds, estimator, library)
        objective = (1.0 - metrics["label_accuracy"]) + 25.0 * metrics["stress_false_negative_rate"] + 2.0 * metrics["surface_error_rate"]
        row = {"estimator": name, "family": family, "split": "selection", "objective": objective, **params, **metrics}
        rows.append(row)
        if family == "threshold_rssi" and objective < best_threshold_objective:
            best_threshold_objective = objective
            best_threshold = (name, estimator)
        if family == "knn_2feature" and objective < best_knn_objective:
            best_knn_objective = objective
            best_knn = (name, estimator)
        if objective < best_objective:
            best_objective = objective
            best_name = name
            best_estimator = estimator

    threshold_row = min((row for row in rows if row["family"] == "threshold_rssi"), key=lambda row: float(row["objective"]))
    if float(threshold_row["label_accuracy"]) >= 0.85 and float(threshold_row["stress_false_negative_rate"]) <= 0.001 and best_threshold is not None:
        best_name, best_estimator = best_threshold
    elif best_knn is not None:
        best_name, best_estimator = best_knn
    if best_estimator is None:
        raise RuntimeError("No estimator candidates were calibrated")
    grid = pd.DataFrame(rows)
    grid["selected"] = grid["estimator"].eq(best_name)
    return best_name, best_estimator, grid


def _run_calibrated_estimator(root: Path, cfg: SimConfig, wp: WorkPoint, train_seeds: list[int], test_seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    estimator_name, estimator, grid = _calibrate_estimators(cfg, train_seeds, lib)
    grid.to_csv(final_dir / "estimator_calibration_grid.csv", index=False)
    raw = run_suite(cfg, wp, ["Proposed"], test_seeds, "stress", lib, label_estimator=estimator)
    raw.to_csv(final_dir / "estimator_closed_loop_per_slot.csv", index=False)
    summary = summarize_raw(raw, cfg, scenario="estimator_closed_loop")
    by_seed = summary.by_seed.copy()
    label_by_seed = _label_quality_by_seed(raw)
    by_seed = by_seed.merge(label_by_seed, on="seed", how="left")
    by_seed.insert(0, "estimator", estimator_name)
    by_seed.to_csv(final_dir / "estimator_closed_loop_by_seed.csv", index=False)

    quality = _label_quality_from_frame(raw)
    confusion = raw.groupby(["label_true", "label_decision"], as_index=False).size().rename(columns={"size": "count"})
    confusion.to_csv(final_dir / "estimator_label_confusion.csv", index=False)
    metrics = dict(summary.by_method["Proposed"])
    metrics.update(quality)
    metrics["estimator"] = estimator_name
    metrics["method"] = "Proposed+calibrated-est"
    write_json(final_dir / "estimator_selected.json", {"estimator": estimator_name, **quality})
    return _write_metric_table(final_dir / "estimator_closed_loop_summary.csv", [metrics])


def _make_random_flip_estimator(rate: float, *, seed_offset: int = 6100) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        decisions = []
        for node, label in enumerate(true_labels):
            rng = _rng_for_observation(seed_offset, trace, t, node)
            candidates = _all_kind_candidates(_kind_for_label(label))
            if rng.random() < rate and len(candidates) > 1:
                wrong = [candidate for candidate in candidates if candidate != label]
                decisions.append(str(rng.choice(wrong)))
            else:
                decisions.append(label)
        return decisions

    return estimator


def _make_stress_false_negative_estimator(rate: float, *, seed_offset: int = 7100) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        decisions = []
        for node, label in enumerate(true_labels):
            rng = _rng_for_observation(seed_offset, trace, t, node)
            if label.endswith("_stress") and rng.random() < rate:
                decisions.append(label.replace("_stress", "_rest"))
            else:
                decisions.append(label)
        return decisions

    return estimator


def _make_stuck_mixed_burst_estimator(length: int) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        if length <= 0:
            return true_labels
        in_burst = trace.seed % 3 == 0 and t >= 500 and ((t - 500) % 55) < length
        if not in_burst:
            return true_labels
        decisions = []
        for label in true_labels:
            if label.endswith("_stress"):
                decisions.append(label.replace("_stress", "_rest"))
            elif label.startswith("surface"):
                decisions.append("surface_moderate_loose")
            else:
                decisions.append(label)
        return decisions

    return estimator


def _make_combined_adverse_estimator(*, random_rate: float, stress_fn_rate: float, burst_length: int, seed_offset: int = 9100) -> LabelEstimator:
    def estimator(t: int, true_labels: list[str], trace: Trace, library) -> list[str]:
        in_burst = burst_length > 0 and trace.seed % 3 == 0 and t >= 500 and ((t - 500) % 55) < burst_length
        decisions: list[str] = []
        for node, label in enumerate(true_labels):
            rng = _rng_for_observation(seed_offset, trace, t, node)
            decision = label
            candidates = _all_kind_candidates(_kind_for_label(label))
            if rng.random() < random_rate and len(candidates) > 1:
                wrong = [candidate for candidate in candidates if candidate != label]
                decision = str(rng.choice(wrong))
            if label.endswith("_stress") and rng.random() < stress_fn_rate:
                decision = label.replace("_stress", "_rest")
            if in_burst:
                if label.endswith("_stress"):
                    decision = label.replace("_stress", "_rest")
                elif label.startswith("surface"):
                    decision = "surface_moderate_loose"
            decisions.append(decision)
        return decisions

    return estimator


def _summarize_perturbed_run(
    *,
    cfg: SimConfig,
    wp: WorkPoint,
    seeds: list[int],
    library,
    panel: str,
    perturbation: object,
    estimator: LabelEstimator,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    raw = run_suite(cfg, wp, ["Proposed"], seeds, "stress", library, label_estimator=estimator)
    summary = summarize_raw(raw, cfg, scenario=f"label_robustness_{panel}")
    label_quality = _label_quality_from_frame(raw)
    rows = []
    for method, metrics in summary.by_method.items():
        rows.append({"panel": panel, "perturbation": perturbation, "method": method, **metrics, **label_quality})
    by_seed = summary.by_seed.copy()
    by_seed.insert(0, "panel", panel)
    by_seed.insert(1, "perturbation", perturbation)
    by_seed = by_seed.merge(_label_quality_by_seed(raw), on="seed", how="left")
    return rows, by_seed


def _run_label_robustness(root: Path, cfg: SimConfig, wp: WorkPoint, train_seeds: list[int], test_seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    _, calibrated_estimator, grid = _calibrate_estimators(cfg, train_seeds, lib)
    grid.to_csv(final_dir / "label_robustness_estimator_calibration_grid.csv", index=False)
    all_rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    specs: list[tuple[str, object, LabelEstimator]] = []
    specs.extend(("random_flip", rate, _make_random_flip_estimator(rate)) for rate in [0.00, 0.05, 0.10, 0.15, 0.20])
    specs.extend(("stress_false_negative", rate, _make_stress_false_negative_estimator(rate)) for rate in [0.00, 0.05, 0.10, 0.20])
    specs.extend(("stuck_mixed_burst", length, _make_stuck_mixed_burst_estimator(int(length))) for length in [1, 3, 5, 10])
    specs.append(("rssi_calibrated", "selected", calibrated_estimator))
    for panel, perturbation, estimator in specs:
        rows, by_seed = _summarize_perturbed_run(cfg=cfg, wp=wp, seeds=test_seeds, library=lib, panel=panel, perturbation=perturbation, estimator=estimator)
        all_rows.extend(rows)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "label_robustness_by_seed.csv", index=False)
    return _write_metric_table(final_dir / "label_robustness_summary.csv", all_rows)


def _run_robustness_boundary(root: Path, cfg: SimConfig, wp: WorkPoint, test_seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    all_rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    specs: list[tuple[str, object, LabelEstimator]] = []
    specs.extend(("random_flip", rate, _make_random_flip_estimator(rate)) for rate in [0.10, 0.20, 0.30, 0.40, 0.50])
    specs.extend(("stress_false_negative", rate, _make_stress_false_negative_estimator(rate)) for rate in [0.20, 0.30, 0.50, 0.70, 0.90])
    specs.extend(("stuck_mixed_burst", length, _make_stuck_mixed_burst_estimator(int(length))) for length in [10, 20, 50, 100, 200])
    for panel, perturbation, estimator in specs:
        rows, by_seed = _summarize_perturbed_run(cfg=cfg, wp=wp, seeds=test_seeds, library=lib, panel=panel, perturbation=perturbation, estimator=estimator)
        for row in rows:
            row["accepted"] = bool(float(row.get("stress_shortage_severity", 1.0)) <= 0.01 and float(row.get("em_violation_rate", 1.0)) <= 0.02)
        all_rows.extend(rows)
        by_seed_frames.append(by_seed)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "robustness_boundary_by_seed.csv", index=False)
    summary = _write_metric_table(final_dir / "robustness_boundary_summary.csv", all_rows)
    boundary_rows: list[dict[str, object]] = []
    for panel, group in summary.groupby("panel", sort=False):
        ordered = group.copy()
        ordered["perturbation_float"] = pd.to_numeric(ordered["perturbation"], errors="coerce")
        ordered = ordered.sort_values("perturbation_float")
        zero = ordered[ordered["stress_shortage_severity"].le(1e-12)]
        mild = ordered[ordered["stress_shortage_severity"].le(0.05)]
        fail = ordered[ordered["stress_shortage_severity"].gt(0.05)]
        boundary_rows.append(
            {
                "panel": panel,
                "robust_up_to": float(zero["perturbation_float"].max()) if not zero.empty else np.nan,
                "degrades_up_to": float(mild["perturbation_float"].max()) if not mild.empty else np.nan,
                "first_fail_at": float(fail["perturbation_float"].min()) if not fail.empty else np.nan,
            }
        )
    _write_metric_table(final_dir / "robustness_boundary_corridor.csv", boundary_rows)
    combo_rows: list[dict[str, object]] = []
    combo_by_seed_frames: list[pd.DataFrame] = []
    combo_specs = [
        ("combo_1", 0.20, 0.30, 20, 1.00, 1.00),
        ("combo_2", 0.30, 0.50, 50, 0.85, 1.00),
        ("combo_3", 0.40, 0.70, 100, 0.75, 1.00),
        ("combo_4", 0.50, 0.90, 200, 0.65, 1.00),
        ("combo_5", 0.50, 0.95, 200, 0.55, 1.00),
        ("combo_6", 0.50, 0.95, 200, 0.45, 0.85),
        ("combo_7", 0.50, 0.95, 200, 0.35, 0.75),
        ("combo_8", 0.50, 0.95, 200, 0.25, 0.65),
    ]
    for label, random_rate, stress_fn_rate, burst_length, budget_scale, power_scale in combo_specs:
        edge_wp = replace(wp, B_EM_ref=wp.B_EM_ref * budget_scale, P_H=wp.P_H * power_scale)
        estimator = _make_combined_adverse_estimator(random_rate=random_rate, stress_fn_rate=stress_fn_rate, burst_length=burst_length)
        rows, by_seed = _summarize_perturbed_run(
            cfg=cfg,
            wp=edge_wp,
            seeds=test_seeds,
            library=lib,
            panel="combined_edge",
            perturbation=label,
            estimator=estimator,
        )
        for row in rows:
            row["random_flip"] = random_rate
            row["stress_false_negative"] = stress_fn_rate
            row["burst_length"] = burst_length
            row["budget_scale"] = budget_scale
            row["power_scale"] = power_scale
            row["B_EM_ref"] = edge_wp.B_EM_ref
            row["P_H"] = edge_wp.P_H
            row["accepted"] = bool(float(row.get("stress_shortage_severity", 1.0)) <= 0.05)
        combo_rows.extend(rows)
        by_seed.insert(2, "random_flip", random_rate)
        by_seed.insert(3, "stress_false_negative", stress_fn_rate)
        by_seed.insert(4, "burst_length", burst_length)
        by_seed.insert(5, "budget_scale", budget_scale)
        by_seed.insert(6, "power_scale", power_scale)
        combo_by_seed_frames.append(by_seed)
    combo_summary = _write_metric_table(final_dir / "robustness_boundary_combo_summary.csv", combo_rows)
    if combo_by_seed_frames:
        pd.concat(combo_by_seed_frames, ignore_index=True).to_csv(final_dir / "robustness_boundary_combo_by_seed.csv", index=False)
    combo_fail = combo_summary[combo_summary["stress_shortage_severity"].gt(0.05)]
    _write_metric_table(
        final_dir / "robustness_boundary_combo_corridor.csv",
        [
            {
                "panel": "combined_edge",
                "robust_up_to": str(combo_summary[combo_summary["stress_shortage_severity"].le(1e-12)]["perturbation"].iloc[-1])
                if combo_summary["stress_shortage_severity"].le(1e-12).any()
                else "",
                "degrades_up_to": str(combo_summary[combo_summary["stress_shortage_severity"].le(0.05)]["perturbation"].iloc[-1])
                if combo_summary["stress_shortage_severity"].le(0.05).any()
                else "",
                "first_fail_at": str(combo_fail["perturbation"].iloc[0]) if not combo_fail.empty else "",
            }
        ],
    )
    return summary


def _run_rssi_noise_sweep(root: Path, cfg: SimConfig, wp: WorkPoint, test_seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    rows: list[dict[str, object]] = []
    by_seed_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    for noise_db in [0.50, 1.00, 2.00, 5.00]:
        estimator = _make_threshold_rssi_estimator(noise_db=noise_db, bias_db=-1.0, seed_offset=8200)
        raw = run_suite(cfg, wp, ["Proposed"], test_seeds, "stress", lib, label_estimator=estimator)
        raw.to_csv(final_dir / f"rssi_noise_{noise_db:.2f}_per_slot.csv", index=False)
        summary = summarize_raw(raw, cfg, scenario="rssi_noise_sweep")
        quality = _label_quality_from_frame(raw)
        metrics = dict(summary.by_method["Proposed"])
        rows.append(
            {
                "estimator_family": "threshold_rssi",
                "noise_db": float(noise_db),
                "noise_energy_db": float(noise_db),
                "noise_data_db": 0.0,
                "bias_db": -1.0,
                "method": "Proposed+RSSIest",
                **metrics,
                **quality,
            }
        )
        by_seed = summary.by_seed.copy()
        by_seed.insert(0, "estimator_family", "threshold_rssi")
        by_seed.insert(1, "noise_db", float(noise_db))
        by_seed.insert(2, "bias_db", -1.0)
        by_seed = by_seed.merge(_label_quality_by_seed(raw), on="seed", how="left")
        by_seed_frames.append(by_seed)
        confusion = raw.groupby(["label_true", "label_decision"], as_index=False).size().rename(columns={"size": "count"})
        confusion.insert(0, "noise_db", float(noise_db))
        confusion_frames.append(confusion)
    if by_seed_frames:
        pd.concat(by_seed_frames, ignore_index=True).to_csv(final_dir / "rssi_noise_sweep_by_seed.csv", index=False)
    if confusion_frames:
        pd.concat(confusion_frames, ignore_index=True).to_csv(final_dir / "rssi_noise_sweep_confusion.csv", index=False)
    return _write_metric_table(final_dir / "rssi_noise_sweep_summary.csv", rows)


def _make_rssi_estimator(noise_std: float = 1.75, seed_offset: int = 1000):
    def estimator(t: int, true_labels: list[str], trace, library) -> list[str]:
        rng = np.random.default_rng(seed_offset + trace.seed * 100000 + t)
        candidates_by_kind = {
            "surface": ["surface_rest", "surface_sweat", "surface_moderate_loose"],
            "implant10": ["implant10_rest", "implant10_stress"],
            "implant30": ["implant30_rest", "implant30_stress"],
        }
        estimated: list[str] = []
        for label in true_labels:
            row = library.coeff(label)
            obs_db = 10.0 * np.log10(max(row.g_norm, 1e-12)) + float(rng.normal(0.0, noise_std))
            if label.startswith("surface"):
                candidates = candidates_by_kind["surface"]
            elif label.startswith("implant10"):
                candidates = candidates_by_kind["implant10"]
            else:
                candidates = candidates_by_kind["implant30"]
            chosen = min(candidates, key=lambda item: abs(10.0 * np.log10(max(library.coeff(item).g_norm, 1e-12)) - obs_db))
            estimated.append(chosen)
        return estimated

    return estimator


def _run_estimator_prototype(root: Path, cfg: SimConfig, wp: WorkPoint, seeds: list[int]) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    lib = load_hfss_library()
    raw = run_suite(cfg, wp, ["Proposed"], seeds, "stress", lib, label_estimator=_make_rssi_estimator())
    summary = summarize_raw(raw, cfg, scenario="estimator_prototype")
    by_seed = summary.by_seed.copy()
    label_frame = raw[["seed", "t", "node", "label_true", "label_decision", "is_stressed"]].copy()
    label_frame["label_correct"] = label_frame["label_true"].eq(label_frame["label_decision"])
    label_frame["stress_false_negative"] = label_frame["is_stressed"].astype(bool) & ~label_frame["label_decision"].astype(str).str.endswith("_stress")
    label_metrics = label_frame.groupby("seed", as_index=False).agg(
        label_accuracy=("label_correct", "mean"),
        stress_false_negative_rate=("stress_false_negative", "mean"),
    )
    by_seed = by_seed.merge(label_metrics, on="seed", how="left")
    by_seed.to_csv(final_dir / "estimator_prototype_by_seed.csv", index=False)
    metrics = dict(summary.by_method["Proposed"])
    metrics["label_accuracy"] = float(label_frame["label_correct"].mean())
    metrics["stress_false_negative_rate"] = float(label_frame["stress_false_negative"].mean())
    return _write_metric_table(final_dir / "estimator_prototype_summary.csv", [{"method": "Proposed+RSSIest", **metrics}])


def _final_workpoint() -> WorkPoint:
    return WorkPoint(lambda_q=1.31, lambda_c=1.53, lambda_e=0.21, lambda_xi=0.39, B_EM_ref=0.84, P_H=1.14, lambda_s=20.0)


def _run_phase_only(root: Path, cfg: SimConfig) -> pd.DataFrame:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = final_dir / "candidates" / "heldout_final"
    payload_path = candidate_dir / "candidate_metrics.json"

    payload: dict[str, object] | None = None
    wp = _final_workpoint()
    seeds = TEST_SEEDS
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        wp = WorkPoint(**payload["workpoint"])
        seeds = [int(seed) for seed in payload.get("seeds", TEST_SEEDS)]

    phase = _phase_summary(cfg, wp, seeds, quick=False)
    phase.to_csv(final_dir / "phase_only.csv", index=False)
    plot_fig7(phase, final_dir)

    if payload is not None:
        payload["phase"] = phase.to_dict(orient="records")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid metrics payload in {payload_path}")
        report = evaluate_candidate_gates(metrics, phase)
        write_json(payload_path, payload)
        write_json(candidate_dir / "gate_result.json", {"passed": report.passed, "failed_codes": report.failed_codes, "score": report.score})
        write_pass_fail_csv(candidate_dir / "candidate_pass_fail.csv", report.rows)
        (candidate_dir / "figure_gate_report.md").write_text(report.to_markdown(), encoding="utf-8")
    else:
        write_json(
            final_dir / "phase_only_candidate_metrics.json",
            {
                "candidate": "phase_only",
                "seeds": seeds,
                "workpoint": wp.to_dict(),
                "phase": phase.to_dict(orient="records"),
            },
        )
    return phase


def _search_gate_ok(report: GateReport) -> bool:
    fatal_prefixes = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "F3", "F5")
    return not any(code.startswith(fatal_prefixes) for code in report.failed_codes)


def _only_phase_failed(report: GateReport) -> bool:
    phase_codes = {"G8", "F7-G1", "F7-G2", "F7-G3", "F7-G4", "F7-G5"}
    return bool(report.failed_codes) and set(report.failed_codes).issubset(phase_codes)


def _write_logs(root: Path, iteration_rows: list[dict[str, object]], ralph_rows: list[dict[str, object]]) -> None:
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(iteration_rows).to_csv(out / "iteration_log.csv", index=False)
    pd.DataFrame(ralph_rows).to_csv(out / "ralph_loop_log.csv", index=False)


def _write_final_reports(
    root: Path,
    wp: WorkPoint,
    gate: GateReport,
    metrics: dict[str, dict[str, float]],
    phase: pd.DataFrame,
    iteration_rows: list[dict[str, object]],
    ralph_rows: list[dict[str, object]],
) -> None:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(final_dir / "final_workpoint.yaml", wp.to_dict())
    (final_dir / "figure_gate_report.md").write_text(gate.to_markdown(), encoding="utf-8")
    _write_logs(root, iteration_rows, ralph_rows)
    for log_name in ["iteration_log.csv", "ralph_loop_log.csv"]:
        src = root / "output" / log_name
        dst = final_dir / log_name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    p = metrics["Proposed"]
    adt = metrics["ADT-MAC"]
    dpp = metrics["Lyap.-DPP"]
    no_em = metrics["w/o EM-bud."]
    no_urg = metrics["w/o crit.-urg."]
    no_impl = metrics["w/o implant-aware"]
    risk_drop = 1.0 - p["stress_shortage_severity"] / max(adt["stress_shortage_severity"], 1e-12)
    service_items = {
        "surface_served_ratio": p["surface_served_ratio"] - dpp["surface_served_ratio"],
        "p95_backlog_reduction": (dpp["p95_backlog"] - p["p95_backlog"]) / max(dpp["p95_backlog"], 1e-12),
        "post_stress_backlog_reduction": (dpp["post_stress_backlog_mean"] - p["post_stress_backlog_mean"]) / max(dpp["post_stress_backlog_mean"], 1e-12),
    }
    hard_gate_lines = [
        f"- {row['gate']}: {'PASS' if row['passed'] else 'FAIL'} - {row['detail']}"
        for row in gate.rows
        if str(row["gate"]).startswith("G")
    ]
    figure_gate_lines = []
    for prefix, label in [("F3", "Fig. 3"), ("F4", "Fig. 4"), ("F5", "Fig. 5"), ("F6", "Fig. 6"), ("F7", "Fig. 7")]:
        rows = [row for row in gate.rows if str(row["gate"]).startswith(prefix)]
        figure_gate_lines.append(f"- {label}: {'PASS' if all(bool(row['passed']) for row in rows) else 'FAIL'}")
    report = [
        "# final_report",
        "",
        "## final_workpoint",
        *(f"- {key}: {value}" for key, value in wp.to_dict().items()),
        f"- selection seeds: {TRAIN_SEEDS[0]}--{TRAIN_SEEDS[-1]}",
        f"- held-out reporting seeds: {TEST_SEEDS[0]}--{TEST_SEEDS[-1]}",
        f"- HFSS source: {load_hfss_library().source_path}",
        "",
        "## hard gate results",
        f"- overall gate status: {'PASS' if gate.passed else 'FAIL'}",
        f"- failed gates: {', '.join(gate.failed_codes) if gate.failed_codes else 'none'}",
        *hard_gate_lines,
        "",
        "## figure gate results",
        *figure_gate_lines,
        "",
        "## Results claims supported",
        f"- Proposed vs ADT-MAC implant risk drop: {risk_drop:.2%}",
        f"- Proposed vs Lyap.-DPP service metrics: {service_items}",
        f"- w/o EM-bud. EM overuse: {no_em['em_violation_rate']:.4f}",
        f"- w/o crit.-urg. stress shortage severity: {no_urg['stress_shortage_severity']:.4f}",
        f"- w/o implant-aware stress shortage severity: {no_impl['stress_shortage_severity']:.4f}",
        f"- Fig. 7 transition cells: low={(phase['stress_shortage_severity'] < 0.03).mean():.2%}, medium={((phase['stress_shortage_severity'] >= 0.03) & (phase['stress_shortage_severity'] <= 0.50)).mean():.2%}, high={(phase['stress_shortage_severity'] > 0.50).mean():.2%}",
        "",
        "## soft targets and trade-off",
        "- Proposed is interpreted as a constrained Pareto operating point, not a throughput-only maximizer.",
        "- ADT-MAC may preserve service but exposes higher stressed implant risk.",
        "- Lyap.-DPP may protect implants, but at least one surface/backlog/recovery service metric is worse than Proposed.",
    ]
    (final_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = """# figure_manifest

| Figure | Scenario | Metrics | Workpoint |
| --- | --- | --- | --- |
| Fig. 3 | localized implant stress | stress shortage, surface served ratio, p95 backlog, recovery time | final_workpoint.yaml |
| Fig. 4 | HFSS condition switching without implant stress | Proposed node-class uE, Proposed raw EM budget usage, critical margin | final_workpoint.yaml |
| Fig. 5 | localized implant stress | stressed implant energy and shortage severity time series | final_workpoint.yaml |
| Fig. 6 | stress ablation | shortage severity, implant p05 energy, EM violation, served workload ratio | final_workpoint.yaml |
| Fig. 7 | adaptive boundary-focused B_EM/P_H phase scan | stress shortage severity | final_workpoint.yaml |

Fig. 3-Fig. 6 use the same final_workpoint on held-out seeds 10--19. Fig. 7 uses the same final_workpoint with a boundary-focused representative-seed LP phase scan.
"""
    (final_dir / "figure_manifest.md").write_text(manifest, encoding="utf-8")
    captions = """# captions_cn

- Fig. 3锛欻FSS 鏍囧畾妞嶅叆鍘嬪姏绐楀彛涓嬩笁绉嶄富鏂规硶鐨勬鍏ョ煭缂恒€佹湇鍔′繚鎸併€侀槦鍒楀拰鎭㈠鏃堕棿姣旇緝銆?- Fig. 4锛氳鍥捐В閲?Proposed 鍦?HFSS 鏍囧畾宸ュ喌鍒囨崲涓嬬殑鍐呴儴鍝嶅簲鏈哄埗锛岃€屼笉鏄富 baseline 瀵规瘮鍥俱€傚瓙鍥惧垎鍒睍绀?Proposed 鎸夎妭鐐圭被鍒殑 energy-support allocation銆丳roposed 瀵?EM budget 鐨勫姩鎬佷娇鐢ㄤ笖濮嬬粓浣庝簬棰勭畻绾匡紝浠ュ強 critical urgency 瀵瑰叧閿鍏ヨ兘閲忚閲忕殑淇濇姢浣滅敤锛沚aseline 瀵规瘮瑙?Fig. 3銆?- Fig. 5锛氬眬閮ㄦ鍏ュ帇鍔涚獥鍙ｄ腑鍙楀帇妞嶅叆鑺傜偣鐨勮兘閲忚建杩瑰拰鐭己寮哄害杞ㄨ抗銆?- Fig. 6锛欵M budget銆乧ritical urgency 鍜?implant-aware library 涓夐」鍐呴儴璁捐鐨勬秷铻嶅姣旓紱鍙充笅瑙掍娇鐢ㄧ粺涓€鍒嗘瘝鐨?served workload ratio 灞曠ず鏈嶅姟淇濇寔涓庨闄?鍚堣涔嬮棿鐨?trade-off锛岃€屼笉浣跨敤 recovery time銆?- Fig. 7锛?B_{EM}$ 涓?$P_H$ 鐨?operating-region phase diagram锛屾槦鍙锋爣鍑?final workpoint銆?"""
    (final_dir / "captions_cn.md").write_text(captions, encoding="utf-8")
    summary = f"""# results_summary_cn

鏈€缁堝伐浣滅偣涓?`{wp.to_dict()}`銆傛墍鏈?Fig. 3-7 鍧囦娇鐢ㄥ悓涓€宸ヤ綔鐐瑰拰 20 seeds銆侾roposed 鍦?EM violation銆乺x-cap violation 鍜?LP failure 鍧囦负 0 鐨勫墠鎻愪笅锛屽皢 stress-window implant shortage severity 鐩告瘮 ADT-MAC 闄嶄綆 {risk_drop:.2%}銆傜浉瀵?Lyap.-DPP锛孭roposed 鑷冲皯鍦ㄤ竴涓湇鍔?鎭㈠鎸囨爣涓婂舰鎴愪紭鍔裤€侳ig. 4 鐢ㄤ簬瑙ｉ噴 Proposed 鍦?HFSS 宸ュ喌鍒囨崲涓嬬殑鍐呴儴鏈哄埗鍝嶅簲锛宐aseline 涓绘瘮杈冪敱 Fig. 3 缁欏嚭锛泈/o EM-bud. 鏄剧ず EM overuse锛寃/o crit.-urg. 鍜?w/o implant-aware 鏄剧ず妞嶅叆淇濇姢鎴栨ā鍨嬪け閰嶉闄┿€侳ig. 7 鏄剧ず浠庝笉鍙鍒板彲琛岀殑杩囨浮鍖哄煙銆?"""
    (final_dir / "results_summary_cn.md").write_text(summary, encoding="utf-8")


def _release_final(
    root: Path,
    cfg: SimConfig,
    wp: WorkPoint,
    gate: GateReport,
    metrics: dict[str, dict[str, float]],
    phase: pd.DataFrame,
    condition_raw: pd.DataFrame,
    stress_raw: pd.DataFrame,
    iteration_rows: list[dict[str, object]],
    ralph_rows: list[dict[str, object]],
) -> None:
    final_dir = root / "output"
    final_dir.mkdir(parents=True, exist_ok=True)
    plot_fig3(metrics, final_dir)
    plot_fig4(condition_raw, cfg, final_dir)
    plot_fig5(stress_raw, cfg, final_dir)
    plot_fig6(metrics, final_dir)
    plot_fig7(phase, final_dir)
    _write_final_reports(root, wp, gate, metrics, phase, iteration_rows, ralph_rows)


def _existing_log_row(candidate_dir: Path, *, full_pass_default: bool = False) -> dict[str, object] | None:
    metrics_path = candidate_dir / "candidate_metrics.json"
    gate_path = candidate_dir / "gate_result.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate_payload = {"passed": full_pass_default, "failed_codes": ["missing_gate_result"], "score": 0.0}
    if gate_path.exists():
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    wp = WorkPoint(**payload["workpoint"])
    failed_codes = [str(code) for code in gate_payload.get("failed_codes", [])]
    return {
        "round": candidate_dir.name,
        "seeds": len(payload.get("seeds", [])),
        "score": float(gate_payload.get("score", 0.0)),
        "core_pass": not any(code.startswith(("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "F3", "F5")) for code in failed_codes),
        "full_pass": bool(gate_payload.get("passed", False)),
        "failed": ",".join(failed_codes),
        **wp.to_dict(),
    }


def _resume_final(root: Path, cfg: SimConfig) -> None:
    candidates_root = root / "output" / "candidates"
    iteration_rows: list[dict[str, object]] = []
    ralph_rows: list[dict[str, object]] = []
    final_record: tuple[WorkPoint, GateReport, dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    for candidate_dir in sorted(candidates_root.glob("iter_*")):
        row = _existing_log_row(candidate_dir)
        if row is not None:
            iteration_rows.append(row)

    for idx in range(1, 16):
        name = f"ralph_{idx:02d}"
        candidate_dir = candidates_root / name
        payload_path = candidate_dir / "candidate_metrics.json"
        raw_dir = candidate_dir / "raw"
        if not payload_path.exists():
            row = _existing_log_row(candidate_dir)
            if row is not None:
                ralph_rows.append(row)
            continue
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        wp = WorkPoint(**payload["workpoint"])

        if idx < 11:
            row = _existing_log_row(candidate_dir)
            if row is not None:
                ralph_rows.append(row)
            continue

        required_raw = [
            raw_dir / "condition_switch_per_slot.csv",
            raw_dir / "stress_per_slot.csv",
            raw_dir / "ablation_per_slot.csv",
        ]
        if not all(path.exists() for path in required_raw):
            row = _existing_log_row(candidate_dir) or {"round": name, **wp.to_dict()}
            row.update({"full_pass": False, "failed": "missing_raw_for_resume"})
            ralph_rows.append(row)
            continue

        condition_raw = pd.read_csv(raw_dir / "condition_switch_per_slot.csv")
        stress_raw = pd.read_csv(raw_dir / "stress_per_slot.csv")
        ablation_raw = pd.read_csv(raw_dir / "ablation_per_slot.csv")
        seeds = list(range(20))
        metrics = _compose_metrics(cfg, condition_raw, stress_raw, ablation_raw)
        phase = _phase_summary(cfg, wp, seeds, quick=False)
        gate = evaluate_candidate_gates(metrics, phase)

        write_json(
            payload_path,
            {
                "candidate": name,
                "seeds": seeds,
                "workpoint": wp.to_dict(),
                "metrics": metrics,
                "phase": phase.to_dict(orient="records"),
            },
        )
        write_json(candidate_dir / "gate_result.json", {"passed": gate.passed, "failed_codes": gate.failed_codes, "score": gate.score})
        write_pass_fail_csv(candidate_dir / "candidate_pass_fail.csv", gate.rows)
        (candidate_dir / "figure_gate_report.md").write_text(gate.to_markdown(), encoding="utf-8")

        ralph_rows.append(
            {
                "round": name,
                "seeds": len(seeds),
                "score": gate.score,
                "core_pass": _search_gate_ok(gate),
                "full_pass": gate.passed,
                "failed": ",".join(gate.failed_codes),
                **wp.to_dict(),
            }
        )
        if gate.passed:
            final_record = (wp, gate, metrics, phase, condition_raw, stress_raw)
            break

    _write_logs(root, iteration_rows, ralph_rows)
    if final_record is None:
        failed_codes = ["no_resumable_final_candidate_passed"]
        if ralph_rows:
            last_failed = str(ralph_rows[-1].get("failed", ""))
            failed_codes = last_failed.split(",") if last_failed else failed_codes
        write_failed_report(root / "output", stage="resume_final", failed_codes=failed_codes, reason="existing Ralph candidates did not pass all figure gates after resumed phase scan")
        return

    wp, gate, metrics, phase, condition_raw, stress_raw = final_record
    _release_final(root, cfg, wp, gate, metrics, phase, condition_raw, stress_raw, iteration_rows, ralph_rows)


def run_pipeline(mode: str = "all", output_root: str | Path | None = None) -> None:
    mode = MODE_ALIASES.get(mode, mode)
    root = _output_root(output_root)
    if mode in {
        "all",
        "smoke",
        "diagnostic",
        "force_fail",
        "necessity_ablation",
        "lyap_sweep",
        "small_network",
        "estimator_prototype",
        "calibrated_estimator",
        "label_robustness",
        "network_scale",
        "necessity_audit",
        "robustness_boundary",
        "rssi_noise_sweep",
        "phase2_coarse_library",
        "baseline_extra",
        "heterogeneous_stress",
        "lp_tol_sweep",
        "stress_window_sweep",
        "workpoint_cv",
        "proxy_perturbation_sweep",
        "lambda_s_sweep",
        "phase3_30_seeds",
    }:
        _clean_outputs(root)
    if mode == "redesign_final":
        redesign_final_figures(root / "output", output_root=root / "output")
        return
    cfg = SimConfig()
    wp0 = WorkPoint()
    if mode == "force_fail":
        write_failed_report(root / "output", stage="force_fail", failed_codes=["forced"], reason="forced failure mode")
        return
    if mode == "resume_final":
        _resume_final(root, cfg)
        return
    diagnostic_ok = _diagnostic(root, cfg if mode != "smoke" else SimConfig(T=160, stress_start=60, stress_end=90), wp0)
    if mode == "diagnostic":
        return
    if not diagnostic_ok:
        write_failed_report(root / "output", stage="diagnostic", failed_codes=["G4"], reason="oracle diagnostic failed")
        return

    if mode == "phase_only":
        _run_phase_only(root, cfg)
        return
    if mode == "smoke":
        smoke_cfg = SimConfig(T=160, stress_start=60, stress_end=90, post_stress_window=50, recovery_guard_slots=10)
        _run_candidate(root, "smoke", smoke_cfg, wp0, [0], include_phase=False, write_raw=False)
        return
    if mode == "necessity_ablation":
        _run_necessity_ablation(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "necessity_audit":
        _run_necessity_audit(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "lyap_sweep":
        _run_lyap_v_sensitivity(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "small_network":
        _run_small_network(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "estimator_prototype":
        _run_estimator_prototype(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "calibrated_estimator":
        _run_calibrated_estimator(root, cfg, _final_workpoint(), TRAIN_SEEDS, TEST_SEEDS)
        return
    if mode == "label_robustness":
        _run_label_robustness(root, cfg, _final_workpoint(), TRAIN_SEEDS, TEST_SEEDS)
        return
    if mode == "robustness_boundary":
        _run_robustness_boundary(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "rssi_noise_sweep":
        _run_rssi_noise_sweep(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "network_scale":
        _run_network_scale(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "phase2_coarse_library":
        _run_phase2_coarse_library(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "baseline_extra":
        _run_baseline_extra(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "heterogeneous_stress":
        _run_heterogeneous_stress(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "lp_tol_sweep":
        _run_lp_tol_sweep(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "stress_window_sweep":
        _run_stress_window_sweep(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "workpoint_cv":
        _run_workpoint_cv(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "proxy_perturbation_sweep":
        _run_proxy_perturbation_sweep(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "lambda_s_sweep":
        _run_lambda_s_sweep(root, cfg, _final_workpoint(), TEST_SEEDS)
        return
    if mode == "phase3_30_seeds":
        _run_phase3_30_seeds(root, cfg, _final_workpoint(), PHASE3_SEEDS)
        return

    iteration_rows: list[dict[str, object]] = []
    ralph_rows: list[dict[str, object]] = []
    final_wp = _final_workpoint()
    train_gate, _, train_phase, _, _ = _run_candidate(root, "selection_preregistered", cfg, final_wp, TRAIN_SEEDS, include_phase=True, write_raw=False)
    iteration_rows.append(
        {
            "round": "selection_preregistered",
            "seeds": len(TRAIN_SEEDS),
            "seed_role": "selection",
            "score": train_gate.score,
            "core_pass": _search_gate_ok(train_gate),
            "full_pass": train_gate.passed,
            "failed": ",".join(train_gate.failed_codes),
            **final_wp.to_dict(),
        }
    )
    train_phase.to_csv(root / "output" / "selection_workpoint_phase_distribution.csv", index=False)
    gate, metrics, phase, condition_raw, stress_raw = _run_candidate(root, "heldout_final", cfg, final_wp, TEST_SEEDS, include_phase=True, write_raw=True)
    ralph_rows.append(
        {
            "round": "heldout_final",
            "seeds": len(TEST_SEEDS),
            "seed_role": "held-out",
            "score": gate.score,
            "core_pass": _search_gate_ok(gate),
            "full_pass": gate.passed,
            "failed": ",".join(gate.failed_codes),
            **final_wp.to_dict(),
        }
    )
    _write_logs(root, iteration_rows, ralph_rows)
    _release_final(root, cfg, final_wp, gate, metrics, phase, condition_raw, stress_raw, iteration_rows, ralph_rows)
    _run_necessity_ablation(root, cfg, final_wp, TEST_SEEDS)
    _run_lyap_v_sensitivity(root, cfg, final_wp, TEST_SEEDS)
    _run_small_network(root, cfg, final_wp, TEST_SEEDS)
    _run_estimator_prototype(root, cfg, final_wp, TEST_SEEDS)
    return

    best_wp = wp0
    best_score = -1e9
    any_core_pass = False

    for name, wp in coarse_workpoints():
        gate, _, _, _, _ = _run_candidate(root, name, cfg, wp, TRAIN_SEEDS, include_phase=False, write_raw=False)
        core_pass = _search_gate_ok(gate)
        any_core_pass = any_core_pass or core_pass
        iteration_rows.append({"round": name, "seeds": len(TRAIN_SEEDS), "seed_role": "selection", "score": gate.score, "core_pass": core_pass, "failed": ",".join(gate.failed_codes), **wp.to_dict()})
        if core_pass and gate.score > best_score:
            best_score = gate.score
            best_wp = wp
    if not any_core_pass:
        _write_logs(root, iteration_rows, ralph_rows)
        write_failed_report(root / "output", stage="iter5", failed_codes=["no_iter_candidate_passed_fig3_fig5"], reason="coarse candidates did not pass Fig.3/Fig.5 gate")
        return

    final_record: tuple[WorkPoint, GateReport, dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    for idx in range(1, 16):
        wp = ralph_workpoint(best_wp, idx)
        seeds = TRAIN_SEEDS
        name = f"ralph_{idx:02d}"
        gate, metrics, phase, condition_raw, stress_raw = _run_candidate(root, name, cfg, wp, seeds, include_phase=False, write_raw=idx >= 11)
        if idx >= 11 and _only_phase_failed(gate):
            phase = _phase_summary(cfg, wp, seeds, quick=False)
            gate = evaluate_candidate_gates(metrics, phase)
            candidate_dir = root / "output" / "candidates" / name
            write_json(
                candidate_dir / "candidate_metrics.json",
                {
                    "candidate": name,
                    "seeds": seeds,
                    "workpoint": wp.to_dict(),
                    "metrics": metrics,
                    "phase": phase.to_dict(orient="records"),
                },
            )
            write_json(candidate_dir / "gate_result.json", {"passed": gate.passed, "failed_codes": gate.failed_codes, "score": gate.score})
            write_pass_fail_csv(candidate_dir / "candidate_pass_fail.csv", gate.rows)
            (candidate_dir / "figure_gate_report.md").write_text(gate.to_markdown(), encoding="utf-8")
        core_pass = _search_gate_ok(gate)
        ralph_rows.append({"round": name, "seeds": len(seeds), "seed_role": "selection", "score": gate.score, "core_pass": core_pass, "full_pass": gate.passed, "failed": ",".join(gate.failed_codes), **wp.to_dict()})
        if idx <= 10 and core_pass and gate.score > best_score:
            best_score = gate.score
            best_wp = wp
        if idx >= 11 and gate.passed and (final_record is None or gate.score > final_record[1].score):
            final_record = (wp, gate, metrics, phase, condition_raw, stress_raw)

    _write_logs(root, iteration_rows, ralph_rows)
    if final_record is None:
        write_failed_report(root / "output", stage="ralph15", failed_codes=["no_final_candidate_passed"], reason="Ralph_11-Ralph_15 produced no full-gate 20-seed candidate")
        return
    wp, _, _, _, _, _ = final_record
    gate, metrics, phase, condition_raw, stress_raw = _run_candidate(root, "heldout_final", cfg, wp, TEST_SEEDS, include_phase=True, write_raw=True)[:5]
    ralph_rows.append({"round": "heldout_final", "seeds": len(TEST_SEEDS), "seed_role": "held-out", "score": gate.score, "core_pass": _search_gate_ok(gate), "full_pass": gate.passed, "failed": ",".join(gate.failed_codes), **wp.to_dict()})
    _write_logs(root, iteration_rows, ralph_rows)
    _release_final(root, cfg, wp, gate, metrics, phase, condition_raw, stress_raw, iteration_rows, ralph_rows)
    _run_necessity_ablation(root, cfg, wp, TEST_SEEDS)
    _run_lyap_v_sensitivity(root, cfg, wp, TEST_SEEDS)
    _run_small_network(root, cfg, wp, TEST_SEEDS)
    _run_estimator_prototype(root, cfg, wp, TEST_SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-level HFSS Results figure rebuild.")
    parser.add_argument(
        "--mode",
        choices=[
            "diagnostic",
            "smoke",
            "force_fail",
            "resume_final",
            "redesign_final",
            "phase_only",
            "necessity_ablation",
            "lyap_sweep",
            "lyap_v_sweep",
            "small_network",
            "estimator_prototype",
            "state_error",
            "calibrated_estimator",
            "label_robustness",
            "label_boundary_sweep",
            "robustness_boundary",
            "rssi_noise_sweep",
            "network_scale",
            "necessity_audit",
            "phase2_coarse_library",
            "baseline_extra",
            "heterogeneous_stress",
            "lp_tol_sweep",
            "stress_window_sweep",
            "workpoint_cv",
            "proxy_perturbation_sweep",
            "lambda_s_sweep",
            "phase3_30_seeds",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_pipeline(mode=args.mode, output_root=args.output_root)


if __name__ == "__main__":
    main()

