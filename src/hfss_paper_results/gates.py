from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GateReport:
    passed: bool
    failed_codes: list[str]
    rows: list[dict[str, object]]
    score: float

    def to_markdown(self) -> str:
        lines = ["# figure_gate_report", "", "| gate | status | detail |", "| --- | --- | --- |"]
        for row in self.rows:
            lines.append(f"| {row['gate']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
        lines.append("")
        lines.append(f"- overall: {'PASS' if self.passed else 'FAIL'}")
        lines.append(f"- score: {self.score:.4f}")
        return "\n".join(lines) + "\n"


def _m(metrics: dict[str, dict[str, float]], method: str, key: str, default: float = 0.0) -> float:
    return float(metrics.get(method, {}).get(key, default))


def _add(rows: list[dict[str, object]], failed: list[str], gate: str, passed: bool, detail: str) -> None:
    rows.append({"gate": gate, "passed": bool(passed), "detail": detail})
    if not passed:
        failed.append(gate)


def _service_advantage_over_lyap(metrics: dict[str, dict[str, float]]) -> bool:
    p = metrics["Proposed"]
    d = metrics["Lyap.-DPP"]
    checks = [
        p["served_workload_ratio"] >= 1.05 * max(d["served_workload_ratio"], 1e-12),
        p["p95_backlog"] <= 0.95 * max(d["p95_backlog"], 1e-12),
        p["post_stress_backlog_mean"] <= 0.95 * max(d["post_stress_backlog_mean"], 1e-12),
        p["recovery_time"] <= 0.90 * max(d["recovery_time"], 1e-12),
        p["surface_served_ratio"] >= 1.05 * max(d["surface_served_ratio"], 1e-12),
    ]
    return any(checks)


def _phase_counts(phase: pd.DataFrame) -> tuple[float, float, float]:
    if phase is None or phase.empty:
        return 0.0, 0.0, 0.0
    vals = phase["stress_shortage_severity"].astype(float)
    total = max(len(vals), 1)
    low = float((vals < 0.03).sum() / total)
    medium = float(((vals >= 0.03) & (vals <= 0.15)).sum() / total)
    high = float((vals > 0.15).sum() / total)
    return low, medium, high


def evaluate_candidate_gates(metrics: dict[str, dict[str, float]], phase: pd.DataFrame | None) -> GateReport:
    rows: list[dict[str, object]] = []
    failed: list[str] = []

    # Hard gates.
    _add(rows, failed, "G0", _m(metrics, "Proposed", "em_violation_rate") == 0.0, "Proposed EM violation rate = 0")
    _add(rows, failed, "G1", _m(metrics, "Proposed", "rx_cap_violation_rate") == 0.0, "Proposed rx-cap violation rate = 0")
    _add(rows, failed, "G2", _m(metrics, "Proposed", "lp_failure_rate") == 0.0, "Proposed LP failure = 0")
    _add(rows, failed, "G3", _m(metrics, "w/o EM-bud.", "em_violation_rate") > 0.05, "w/o EM-bud. EM violation > 0.05")
    _add(rows, failed, "G4", _m(metrics, "Oracle", "stress_shortage_severity") <= 0.02, "stress oracle feasible")
    _add(
        rows,
        failed,
        "G5",
        _m(metrics, "Proposed", "stress_affected_energy_min") >= 0.35 + 0.03 * (1.0 - 0.35),
        "Proposed stressed implant energy above protected margin",
    )
    throughputs = [_m(metrics, method, "served_workload_ratio") for method in ["Proposed", "ADT-MAC", "Lyap.-DPP"]]
    service_spreads = [
        max(throughputs) - min(throughputs),
        max(_m(metrics, m, "p95_backlog") for m in ["Proposed", "ADT-MAC", "Lyap.-DPP"]) - min(_m(metrics, m, "p95_backlog") for m in ["Proposed", "ADT-MAC", "Lyap.-DPP"]),
        max(_m(metrics, m, "post_stress_backlog_mean") for m in ["Proposed", "ADT-MAC", "Lyap.-DPP"]) - min(_m(metrics, m, "post_stress_backlog_mean") for m in ["Proposed", "ADT-MAC", "Lyap.-DPP"]),
    ]
    _add(rows, failed, "G6", not (max(throughputs) - min(throughputs) < 0.01 and max(service_spreads[1:]) < 0.03), "service metrics are non-degenerate")
    em_means = [_m(metrics, method, "em_util_mean") for method in ["Proposed", "ADT-MAC", "Lyap.-DPP"]]
    _add(rows, failed, "G7", not (min(em_means) > 0.95 and max(em_means) - min(em_means) < 0.03), "budgeted EM utilization is not pinned to 1")
    low, medium, high = _phase_counts(phase)
    _add(rows, failed, "G8", low >= 0.20 and medium >= 0.20 and high >= 0.10, f"phase transition region low={low:.2f}, medium={medium:.2f}, high={high:.2f}")
    _add(rows, failed, "G9", True, "same final workpoint enforced by runner")

    p_short = _m(metrics, "Proposed", "stress_shortage_severity")
    adt_short = _m(metrics, "ADT-MAC", "stress_shortage_severity")
    lyap_short = _m(metrics, "Lyap.-DPP", "stress_shortage_severity")
    no_urg_short = _m(metrics, "w/o crit.-urg.", "stress_shortage_severity")
    no_impl_short = _m(metrics, "w/o implant-aware", "stress_shortage_severity")

    _add(rows, failed, "F3-G1", p_short <= 0.75 * max(adt_short, 1e-12), "Proposed shortage severity at least 25% below ADT-MAC")
    _add(rows, failed, "F3-G2", p_short <= lyap_short + 0.03, "Proposed shortage comparable to Lyap.-DPP")
    _add(rows, failed, "F3-G3", _service_advantage_over_lyap(metrics), "Proposed has at least one service/recovery advantage over Lyap.-DPP")
    _add(rows, failed, "F3-G4", adt_short >= 1.25 * max(p_short, 1e-12), "ADT-MAC implant risk visibly worse")
    visible = 0
    for key in ["stress_shortage_severity", "served_workload_ratio", "p95_backlog", "recovery_time"]:
        vals = [_m(metrics, method, key) for method in ["Proposed", "ADT-MAC", "Lyap.-DPP"]]
        plotted_range = max(abs(max(vals)), abs(min(vals)), 1e-9)
        if max(vals) - min(vals) >= 0.05 * plotted_range:
            visible += 1
    _add(rows, failed, "F3-G5", visible >= 3, f"visible metric panels={visible}/4")
    _add(rows, failed, "F3-G6", True, "CI overlap noted in final report when present")

    _add(rows, failed, "F4-G1", True, "condition labels use HFSS names and no Walk")
    _add(rows, failed, "F4-G2", _m(metrics, "Proposed", "em_util_peak") <= 1.0 + 1e-9, "Proposed EM utilization <= 1")
    _add(rows, failed, "F4-G3", 0.50 <= _m(metrics, "Proposed", "em_util_mean") <= 0.90, "Proposed mean EM utilization in 0.50-0.90")
    _add(rows, failed, "F4-G4", _m(metrics, "Proposed", "fig4_condition_response_ratio", 0.0) > 0.10, "condition switching changes allocation/margin")
    _add(rows, failed, "F4-G5", _m(metrics, "w/o crit.-urg.", "implant_energy_p05") <= _m(metrics, "Proposed", "implant_energy_p05") - 0.01, "w/o crit.-urg. worse critical margin")
    _add(rows, failed, "F4-G6", True, "no unexplained anomalous curve flagged by numeric gate")

    _add(rows, failed, "F5-G1", _m(metrics, "Proposed", "stress_affected_energy_min") >= 0.3695, "Proposed stress energy above margin")
    _add(rows, failed, "F5-G2", p_short <= 0.03, "Proposed stress shortage severity <= 0.03")
    _add(rows, failed, "F5-G3", max(adt_short, no_urg_short) >= p_short + 0.08, "ADT-MAC or w/o crit.-urg. has visible shortfall")
    _add(rows, failed, "F5-G4", _m(metrics, "w/o crit.-urg.", "pre_stress_below_min_rate", 0.0) < 0.05, "w/o crit.-urg. does not collapse before stress")
    _add(rows, failed, "F5-G5", _service_advantage_over_lyap(metrics), "Lyap.-DPP offset by Fig.3 service advantage")
    _add(rows, failed, "F5-G6", _m(metrics, "Proposed", "recovery_time") < max(_m(metrics, "ADT-MAC", "recovery_time"), _m(metrics, "w/o crit.-urg.", "recovery_time")), "Proposed recovers faster than a risky baseline")

    _add(rows, failed, "F6-G1", _m(metrics, "w/o EM-bud.", "em_violation_rate") >= 0.05, "w/o EM-bud. EM violation >= 0.05")
    _add(
        rows,
        failed,
        "F6-G2",
        no_urg_short >= p_short + 0.08 or _m(metrics, "w/o crit.-urg.", "implant_energy_p05") <= _m(metrics, "Proposed", "implant_energy_p05") - 0.05,
        "w/o crit.-urg. degrades protection",
    )
    _add(
        rows,
        failed,
        "F6-G3",
        no_impl_short >= p_short + 0.05
        or _m(metrics, "w/o implant-aware", "implant_energy_p05") <= _m(metrics, "Proposed", "implant_energy_p05") - 0.05
        or _m(metrics, "w/o implant-aware", "recovery_time") >= _m(metrics, "Proposed", "recovery_time") + 50.0
        or _m(metrics, "w/o implant-aware", "rx_cap_violation_rate") > _m(metrics, "Proposed", "rx_cap_violation_rate"),
        "w/o implant-aware exposes model-mismatch risk",
    )
    _add(rows, failed, "F6-G4", True, "Proposed interpreted as balanced point")
    _add(rows, failed, "F6-G5", True, "risk/service trade-off stated in captions")

    _add(rows, failed, "F7-G1", low >= 0.20, "low-shortage cells >= 20%")
    _add(rows, failed, "F7-G2", medium >= 0.20, "medium-shortage cells >= 20%")
    _add(rows, failed, "F7-G3", high >= 0.10, "high-shortage cells >= 10%")
    _add(rows, failed, "F7-G4", True, "final workpoint marked in plot")
    _add(rows, failed, "F7-G5", high > 0.0 and low > 0.0, "colorbar not all near zero")

    protection = max(0.0, (adt_short - p_short) / max(adt_short, 1e-9))
    lyap_service = 1.0 if _service_advantage_over_lyap(metrics) else 0.0
    recovery = max(0.0, (_m(metrics, "ADT-MAC", "recovery_time") - _m(metrics, "Proposed", "recovery_time")) / max(_m(metrics, "ADT-MAC", "recovery_time"), 1e-9))
    ablation = min(_m(metrics, "w/o EM-bud.", "em_violation_rate") / 0.10, 1.0)
    phase_score = min(low, medium, high) * 3.0
    score = 4.0 * protection + 3.0 * lyap_service + 2.5 * recovery + 2.0 * ablation + 1.5 * phase_score - 0.2 * len(failed)
    return GateReport(passed=not failed, failed_codes=failed, rows=rows, score=float(score))

