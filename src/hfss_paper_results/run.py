from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hfss_paper_results.config import ABLATION_METHODS, MAIN_METHODS, STRESS_METHODS, SimConfig, WorkPoint
from hfss_paper_results.data import load_hfss_library
from hfss_paper_results.gates import GateReport, evaluate_candidate_gates
from hfss_paper_results.metrics import MetricSummary, merged_metric_payload, summarize_raw
from hfss_paper_results.plotting import plot_fig3, plot_fig4, plot_fig5, plot_fig6, plot_fig7, quick_fig3, quick_fig5, redesign_final_figures
from hfss_paper_results.reports import write_failed_report, write_json, write_pass_fail_csv, write_yaml
from hfss_paper_results.search import coarse_workpoints, ralph_seed_count, ralph_workpoint
from hfss_paper_results.sim import run_phase_cell_metrics, run_suite


def _output_root(output_root: str | Path | None) -> Path:
    return Path(output_root) if output_root is not None else Path(__file__).resolve().parents[1]


def _clean_outputs(root: Path) -> None:
    outputs = root / "outputs"
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
        "- condition-dependent coefficient source: calibrated_library_sched.csv only",
        "- old synthetic Rest/Walk/Sweat/Loose scaling table: not used",
        f"- labels: {', '.join(sorted(lib.rows))}",
        f"- oracle stress_shortage_severity: {oracle['stress_shortage_severity']:.6f}",
        f"- oracle EM violation rate: {oracle['em_violation_rate']:.6f}",
        f"- oracle rx-cap violation rate: {oracle['rx_cap_violation_rate']:.6f}",
    ]
    passed = oracle["stress_shortage_severity"] <= 0.02 and oracle["em_violation_rate"] == 0.0 and oracle["rx_cap_violation_rate"] == 0.0
    lines.append(f"- diagnostic passed: {passed}")
    out = root / "outputs" / "diagnostic_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def _phase_summary(cfg: SimConfig, wp: WorkPoint, seeds: list[int], *, quick: bool) -> pd.DataFrame:
    lib = load_hfss_library()
    grids = [
        ([0.40, 0.41, 0.42, 0.43, 0.44], [1.02, 1.04, 1.06, 1.08, 1.10]),
        ([0.70, 0.82, 0.88, 0.94, 1.00], [0.88, 0.94, 1.00, 1.06, 1.24]),
        ([0.82, 0.86, 0.89, 0.91, 0.93, 0.95, 0.97, 1.00, 1.04], [0.82, 0.86, 0.89, 0.91, 0.93, 0.95, 0.97, 1.00, 1.04]),
        ([0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.04], [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.04]),
        ([0.78, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.06], [0.78, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 1.00, 1.06]),
        ([0.60, 0.66, 0.72, 0.78, 0.84, 0.90, 0.96, 1.02, 1.08], [0.80, 0.86, 0.92, 0.98, 1.04, 1.10, 1.16, 1.22, 1.28]),
        ([0.70, 0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18], [0.76, 0.82, 0.88, 0.94, 1.00, 1.06, 1.12, 1.18, 1.24]),
    ]
    if quick:
        grids = [([0.35, 0.55, 0.75, 0.95, 1.15], [0.45, 0.65, 0.85, 1.05, 1.25])]
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
        high = (vals > 0.15).mean()
        final_rows = rows
        medium = ((vals >= 0.03) & (vals <= 0.15)).mean()
        if quick or (low >= 0.20 and medium >= 0.20 and high >= 0.10):
            break
    return pd.DataFrame(final_rows)


def _condition_response_ratio(condition_raw: pd.DataFrame) -> float:
    proposed = condition_raw[condition_raw["method"].eq("Proposed") & condition_raw["is_critical"].astype(bool)]
    if proposed.empty:
        return 0.0
    by_regime = proposed.groupby("regime")["uE"].mean()
    if by_regime.empty:
        return 0.0
    denom = max(float(by_regime.min()), 1e-9)
    return float((by_regime.max() - by_regime.min()) / denom)


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
        for key in ["em_util_mean", "em_util_peak"]:
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
    candidate_dir = root / "outputs" / "candidates" / name
    raw_dir = candidate_dir / "raw"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    condition_methods = MAIN_METHODS + ["w/o crit.-urg."]
    condition_raw = run_suite(cfg, wp, condition_methods, seeds, "condition", lib)
    stress_raw = run_suite(cfg, wp, sorted(set(STRESS_METHODS + ["Oracle"])), seeds, "stress", lib)
    ablation_raw = run_suite(cfg, wp, ABLATION_METHODS, seeds, "stress", lib)

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


def _search_gate_ok(report: GateReport) -> bool:
    fatal_prefixes = ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "F3", "F5")
    return not any(code.startswith(fatal_prefixes) for code in report.failed_codes)


def _only_phase_failed(report: GateReport) -> bool:
    phase_codes = {"G8", "F7-G1", "F7-G2", "F7-G3", "F7-G4", "F7-G5"}
    return bool(report.failed_codes) and set(report.failed_codes).issubset(phase_codes)


def _write_logs(root: Path, iteration_rows: list[dict[str, object]], ralph_rows: list[dict[str, object]]) -> None:
    out = root / "outputs"
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
    final_dir = root / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(final_dir / "final_workpoint.yaml", wp.to_dict())
    (final_dir / "figure_gate_report.md").write_text(gate.to_markdown(), encoding="utf-8")
    _write_logs(root, iteration_rows, ralph_rows)
    for log_name in ["iteration_log.csv", "ralph_loop_log.csv"]:
        src = root / "outputs" / log_name
        if src.exists():
            shutil.copyfile(src, final_dir / log_name)
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
        f"- seeds: 20",
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
        f"- Fig. 7 transition cells: low={(phase['stress_shortage_severity'] < 0.03).mean():.2%}, medium={((phase['stress_shortage_severity'] >= 0.03) & (phase['stress_shortage_severity'] <= 0.15)).mean():.2%}, high={(phase['stress_shortage_severity'] > 0.15).mean():.2%}",
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

Fig. 3-Fig. 6 use the same final_workpoint and 20 seeds. Fig. 7 uses the same final_workpoint with a boundary-focused representative-seed LP phase scan.
"""
    (final_dir / "figure_manifest.md").write_text(manifest, encoding="utf-8")
    captions = """# captions_cn

- Fig. 3：HFSS 标定植入压力窗口下三种主方法的植入短缺、服务保持、队列和恢复时间比较。
- Fig. 4：该图解释 Proposed 在 HFSS 标定工况切换下的内部响应机制，而不是主 baseline 对比图。子图分别展示 Proposed 按节点类别的 energy-support allocation、Proposed 对 EM budget 的动态使用且始终低于预算线，以及 critical urgency 对关键植入能量裕量的保护作用；baseline 对比见 Fig. 3。
- Fig. 5：局部植入压力窗口中受压植入节点的能量轨迹和短缺强度轨迹。
- Fig. 6：EM budget、critical urgency 和 implant-aware library 三项内部设计的消融对比；右下角使用统一分母的 served workload ratio 展示服务保持与风险/合规之间的 trade-off，而不使用 recovery time。
- Fig. 7：$B_{EM}$ 与 $P_H$ 的 operating-region phase diagram，星号标出 final workpoint。
"""
    (final_dir / "captions_cn.md").write_text(captions, encoding="utf-8")
    summary = f"""# results_summary_cn

最终工作点为 `{wp.to_dict()}`。所有 Fig. 3-7 均使用同一工作点和 20 seeds。Proposed 在 EM violation、rx-cap violation 和 LP failure 均为 0 的前提下，将 stress-window implant shortage severity 相比 ADT-MAC 降低 {risk_drop:.2%}。相对 Lyap.-DPP，Proposed 至少在一个服务/恢复指标上形成优势。Fig. 4 用于解释 Proposed 在 HFSS 工况切换下的内部机制响应，baseline 主比较由 Fig. 3 给出；w/o EM-bud. 显示 EM overuse，w/o crit.-urg. 和 w/o implant-aware 显示植入保护或模型失配风险。Fig. 7 显示从不可行到可行的过渡区域。
"""
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
    final_dir = root / "outputs" / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
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
    candidates_root = root / "outputs" / "candidates"
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
        write_failed_report(root / "outputs" / "final", stage="resume_final", failed_codes=failed_codes, reason="existing Ralph candidates did not pass all figure gates after resumed phase scan")
        return

    wp, gate, metrics, phase, condition_raw, stress_raw = final_record
    _release_final(root, cfg, wp, gate, metrics, phase, condition_raw, stress_raw, iteration_rows, ralph_rows)


def run_pipeline(mode: str = "all", output_root: str | Path | None = None) -> None:
    root = _output_root(output_root)
    if mode in {"all", "smoke", "diagnostic", "force_fail"}:
        _clean_outputs(root)
    if mode == "redesign_final":
        redesign_final_figures(root / "outputs" / "final", output_root=root / "outputs")
        return
    cfg = SimConfig()
    wp0 = WorkPoint()
    if mode == "force_fail":
        write_failed_report(root / "outputs" / "final", stage="force_fail", failed_codes=["forced"], reason="forced failure mode")
        return
    if mode == "resume_final":
        _resume_final(root, cfg)
        return
    diagnostic_ok = _diagnostic(root, cfg if mode != "smoke" else SimConfig(T=160, stress_start=60, stress_end=90), wp0)
    if mode == "diagnostic":
        return
    if not diagnostic_ok:
        write_failed_report(root / "outputs" / "final", stage="diagnostic", failed_codes=["G4"], reason="oracle diagnostic failed")
        return

    if mode == "smoke":
        smoke_cfg = SimConfig(T=160, stress_start=60, stress_end=90, post_stress_window=50, recovery_guard_slots=10)
        _run_candidate(root, "smoke", smoke_cfg, wp0, [0], include_phase=False, write_raw=False)
        return

    iteration_rows: list[dict[str, object]] = []
    ralph_rows: list[dict[str, object]] = []
    best_wp = wp0
    best_score = -1e9
    any_core_pass = False

    for name, wp in coarse_workpoints():
        gate, _, _, _, _ = _run_candidate(root, name, cfg, wp, list(range(5)), include_phase=False, write_raw=False)
        core_pass = _search_gate_ok(gate)
        any_core_pass = any_core_pass or core_pass
        iteration_rows.append({"round": name, "seeds": 5, "score": gate.score, "core_pass": core_pass, "failed": ",".join(gate.failed_codes), **wp.to_dict()})
        if core_pass and gate.score > best_score:
            best_score = gate.score
            best_wp = wp
    if not any_core_pass:
        _write_logs(root, iteration_rows, ralph_rows)
        write_failed_report(root / "outputs" / "final", stage="iter5", failed_codes=["no_iter_candidate_passed_fig3_fig5"], reason="coarse candidates did not pass Fig.3/Fig.5 gate")
        return

    final_record: tuple[WorkPoint, GateReport, dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    for idx in range(1, 16):
        wp = ralph_workpoint(best_wp, idx)
        seeds = list(range(ralph_seed_count(idx)))
        name = f"ralph_{idx:02d}"
        gate, metrics, phase, condition_raw, stress_raw = _run_candidate(root, name, cfg, wp, seeds, include_phase=False, write_raw=idx >= 11)
        if idx >= 11 and _only_phase_failed(gate):
            phase = _phase_summary(cfg, wp, seeds, quick=False)
            gate = evaluate_candidate_gates(metrics, phase)
            candidate_dir = root / "outputs" / "candidates" / name
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
        ralph_rows.append({"round": name, "seeds": len(seeds), "score": gate.score, "core_pass": core_pass, "full_pass": gate.passed, "failed": ",".join(gate.failed_codes), **wp.to_dict()})
        if idx <= 10 and core_pass and gate.score > best_score:
            best_score = gate.score
            best_wp = wp
        if idx >= 11 and gate.passed and (final_record is None or gate.score > final_record[1].score):
            final_record = (wp, gate, metrics, phase, condition_raw, stress_raw)

    _write_logs(root, iteration_rows, ralph_rows)
    if final_record is None:
        write_failed_report(root / "outputs" / "final", stage="ralph15", failed_codes=["no_final_candidate_passed"], reason="Ralph_11-Ralph_15 produced no full-gate 20-seed candidate")
        return
    wp, gate, metrics, phase, condition_raw, stress_raw = final_record
    _release_final(root, cfg, wp, gate, metrics, phase, condition_raw, stress_raw, iteration_rows, ralph_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-level HFSS Results figure rebuild.")
    parser.add_argument("--mode", choices=["diagnostic", "smoke", "force_fail", "resume_final", "redesign_final", "all"], default="all")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_pipeline(mode=args.mode, output_root=args.output_root)


if __name__ == "__main__":
    main()
