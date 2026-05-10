from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hfss_paper_results.config import SimConfig


@dataclass(frozen=True)
class MetricSummary:
    by_method: dict[str, dict[str, float]]
    by_seed: pd.DataFrame
    summary: pd.DataFrame


def _ci95(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=0) / np.sqrt(len(values)))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (den + 1e-9))


def _recovery_time(affected: pd.DataFrame, cfg: SimConfig) -> tuple[float, float]:
    threshold = cfg.E_min_critical + 0.05 * (cfg.E_max - cfg.E_min_critical)
    after = affected[affected["t"] >= cfg.stress_end].sort_values("t")
    if after.empty:
        return float(cfg.T - cfg.stress_end), 1.0
    values = after[["t", "e_next"]].drop_duplicates("t").to_numpy()
    guard = cfg.recovery_guard_slots
    for idx in range(0, max(len(values) - guard + 1, 0)):
        window = values[idx : idx + guard]
        if np.all(window[:, 1] >= threshold):
            return float(window[0, 0] - cfg.stress_end), 0.0
    return float(max(after["t"].max() - cfg.stress_end + 1, 0)), 1.0


def _post_stress_window(raw: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    stop = min(cfg.T, cfg.stress_end + cfg.post_stress_window)
    return raw[(raw["t"] >= cfg.stress_end) & (raw["t"] < stop)]


def summarize_raw(raw: pd.DataFrame, cfg: SimConfig, *, scenario: str) -> MetricSummary:
    per_seed_rows: list[dict[str, object]] = []
    for (seed, method), group in raw.groupby(["seed", "method"], sort=False):
        critical = group[group["is_critical"].astype(bool)]
        implant_critical = critical[critical["node_type"].astype(str).str.startswith("implant")]
        stressed = group[group.get("is_stressed", False).astype(bool)] if "is_stressed" in group.columns else group.iloc[0:0]
        stressed_critical = stressed[stressed["is_critical"].astype(bool)]
        post = _post_stress_window(group, cfg)
        surface = group[group["node_type"].astype(str).eq("surface")]
        slots = group.groupby("t", sort=False).agg(
            em_utilization=("em_utilization", "first"),
            em_violation=("em_violation", "first"),
            rx_cap_violation=("rx_cap_violation", "any"),
            lp_failed=("lp_failed", "any"),
        )

        full_short = 0.0
        if not implant_critical.empty:
            full_short = float(((implant_critical["e_next"] < implant_critical["E_min"] - 1e-12) | (implant_critical["xi"] > 1e-8)).mean())

        stress_short = 0.0
        stress_severity = 0.0
        affected_energy_min = float("nan")
        recovery = 0.0
        recovery_censored = 0.0
        if not stressed_critical.empty:
            stress_short = float(((stressed_critical["e_next"] < stressed_critical["E_min"] - 1e-12) | (stressed_critical["xi"] > 1e-8)).mean())
            stress_severity = float((stressed_critical["xi"] / stressed_critical["E_min"].replace(0.0, np.nan)).fillna(0.0).mean())
            affected_energy_min = float(stressed_critical["e_next"].min())
            recovery, recovery_censored = _recovery_time(
                group[(group["node"].isin(cfg.stress_nodes)) & (group["is_critical"].astype(bool))],
                cfg,
            )

        margin = (implant_critical["e_next"] - implant_critical["E_min"]) / (
            implant_critical["E_max"] - implant_critical["E_min"]
        ).replace(0.0, np.nan)
        stress_margin = pd.Series(dtype=float)
        if not stressed_critical.empty:
            stress_margin = (stressed_critical["e_next"] - stressed_critical["E_min"]) / (
                stressed_critical["E_max"] - stressed_critical["E_min"]
            ).replace(0.0, np.nan)

        row = {
            "scenario": scenario,
            "seed": int(seed),
            "method": method,
            "full_implant_shortage_rate": full_short,
            "stress_implant_shortage_rate": stress_short,
            "stress_shortage_severity": stress_severity,
            "served_workload_ratio": _safe_ratio(float(group["served"].sum()), float(group["arrived"].sum())),
            "post_stress_served_ratio": _safe_ratio(float(post["served"].sum()), float(post["arrived"].sum())) if not post.empty else 0.0,
            "surface_served_ratio": _safe_ratio(float(surface["served"].sum()), float(surface["arrived"].sum())) if not surface.empty else 0.0,
            "average_backlog": float(group["q"].mean()),
            "p95_backlog": float(group["q"].quantile(0.95)),
            "critical_backlog_mean": float(critical["q"].mean()) if not critical.empty else 0.0,
            "surface_backlog_mean": float(surface["q"].mean()) if not surface.empty else 0.0,
            "post_stress_backlog_mean": float(post["q"].mean()) if not post.empty else 0.0,
            "implant_energy_mean": float(implant_critical["e_next"].mean()) if not implant_critical.empty else 0.0,
            "implant_energy_p05": float(implant_critical["e_next"].quantile(0.05)) if not implant_critical.empty else 0.0,
            "implant_energy_min": float(implant_critical["e_next"].min()) if not implant_critical.empty else 0.0,
            "implant_energy_margin_mean": float(margin.mean()) if not margin.empty else 0.0,
            "implant_energy_margin_p05": float(margin.quantile(0.05)) if not margin.empty else 0.0,
            "stress_affected_energy_min": affected_energy_min if np.isfinite(affected_energy_min) else 0.0,
            "stress_affected_margin_min": float(stress_margin.min()) if not stress_margin.empty else 0.0,
            "recovery_time": recovery,
            "recovery_censored": recovery_censored,
            "em_util_mean": float(slots["em_utilization"].mean()) if not slots.empty else 0.0,
            "em_util_p95": float(slots["em_utilization"].quantile(0.95)) if not slots.empty else 0.0,
            "em_util_peak": float(slots["em_utilization"].max()) if not slots.empty else 0.0,
            "em_violation_rate": float(slots["em_violation"].mean()) if not slots.empty else 0.0,
            "rx_cap_violation_rate": float(slots["rx_cap_violation"].mean()) if not slots.empty else 0.0,
            "rx_cap_peak_ratio": float(group["rx_cap_ratio"].max()) if "rx_cap_ratio" in group.columns else 0.0,
            "lp_failure_rate": float(slots["lp_failed"].mean()) if not slots.empty else 0.0,
        }
        per_seed_rows.append(row)
    by_seed = pd.DataFrame(per_seed_rows)

    metric_cols = [col for col in by_seed.columns if col not in {"scenario", "seed", "method"}]
    rows: list[dict[str, object]] = []
    for method, group in by_seed.groupby("method", sort=False):
        row: dict[str, object] = {"scenario": scenario, "method": method, "num_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            row[col] = float(group[col].mean())
            row[f"{col}_ci95"] = _ci95(group[col])
        rows.append(row)
    summary = pd.DataFrame(rows)
    by_method = {str(row["method"]): {k: float(v) for k, v in row.items() if k not in {"scenario", "method"}} for row in rows}
    return MetricSummary(by_method=by_method, by_seed=by_seed, summary=summary)


def merged_metric_payload(*summaries: MetricSummary) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for summary in summaries:
        for method, metrics in summary.by_method.items():
            payload.setdefault(method, {}).update(metrics)
    return payload

