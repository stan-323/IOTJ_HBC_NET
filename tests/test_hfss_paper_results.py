from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_synthetic_sched_library(path: Path) -> Path:
    rows = [
        ("surface_rest", "surface", 0, "rest", 1.50, 1.00, 0.01, 10.0, True),
        ("surface_moderate_loose", "surface", 0, "loose", 1.40, 0.80, 0.01, 10.0, True),
        ("surface_contact_failure", "surface", 0, "failure", 0.30, 0.10, 0.01, 10.0, False),
        ("surface_sweat", "surface", 0, "sweat", 1.60, 1.20, 0.01, 10.0, True),
        ("implant10_rest", "implant10", 10, "rest", 2.00, 1.00, 0.01, 10.0, True),
        ("implant10_stress", "implant10", 10, "stress", 1.90, 0.80, 0.01, 10.0, True),
        ("implant30_rest", "implant30", 30, "rest", 2.10, 1.00, 0.01, 10.0, True),
        ("implant30_stress", "implant30", 30, "stress", 2.00, 0.80, 0.01, 10.0, True),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "label",
            "type",
            "depth_mm",
            "condition",
            "g_norm",
            "r_norm",
            "chi_norm",
            "p_rx_cap_norm",
            "sched_include_main",
        ],
    )
    frame.to_csv(path, index=False)
    return path


def test_full_phase_summary_prefers_boundary_focused_grid(monkeypatch):
    from hfss_paper_results import run as run_module
    from hfss_paper_results.config import SimConfig, WorkPoint

    calls: list[tuple[float, float]] = []
    seen_seed_sets: list[tuple[int, ...]] = []

    def fake_phase_cell_metrics(cfg, wp, seeds, library, *, budget_multiplier, power_multiplier):
        calls.append((float(budget_multiplier), float(power_multiplier)))
        seen_seed_sets.append(tuple(seeds))
        boundary = (budget_multiplier - 0.42) + 0.5 * (power_multiplier - 1.06)
        if boundary > 0.015:
            severity = 0.0
        elif boundary >= -0.015:
            severity = 0.08
        else:
            severity = 0.30
        return {
            "stress_shortage_severity": severity,
            "stress_implant_shortage_rate": severity,
            "served_workload_ratio": 0.5,
        }

    monkeypatch.setattr(run_module, "load_hfss_library", lambda: object())
    monkeypatch.setattr(run_module, "run_phase_cell_metrics", fake_phase_cell_metrics)

    phase = run_module._phase_summary(SimConfig(), WorkPoint(), [0, 1, 2], quick=False)

    assert len(phase) == 25
    assert sorted(phase["B_EM_multiplier"].unique().round(2).tolist()) == [0.4, 0.41, 0.42, 0.43, 0.44]
    assert sorted(phase["P_H_multiplier"].unique().round(2).tolist()) == [1.02, 1.04, 1.06, 1.08, 1.1]
    assert len(calls) == 25
    assert set(seen_seed_sets) == {(0,)}


def test_hfss_loader_uses_only_final_sched_library(tmp_path):
    from hfss_paper_results.data import load_hfss_library

    sched_path = _write_synthetic_sched_library(tmp_path / "calibrated_library_sched.csv")
    library = load_hfss_library(sched_path)

    assert library.source_path == sched_path
    assert library.source_path.name == "calibrated_library_sched.csv"
    assert set(library.rows) == {
        "surface_rest",
        "surface_moderate_loose",
        "surface_contact_failure",
        "surface_sweat",
        "implant10_rest",
        "implant10_stress",
        "implant30_rest",
        "implant30_stress",
    }
    assert library.rows["surface_contact_failure"].sched_include_main is False
    assert library.rows["implant30_stress"].g_norm < library.rows["implant30_rest"].g_norm


def test_scenarios_use_hfss_labels_and_no_walk():
    from hfss_paper_results.scenarios import condition_switch_labels, stress_window_labels

    labels_0 = condition_switch_labels(0, 12)
    labels_300 = condition_switch_labels(300, 12)
    labels_600 = condition_switch_labels(600, 12)
    labels_900 = condition_switch_labels(900, 12)

    assert labels_0[1] == "implant30_rest"
    assert labels_0[3] == "surface_rest"
    assert labels_300[3] == "surface_sweat"
    assert labels_600[3] == "surface_moderate_loose"
    assert labels_900[1] == "implant30_rest"
    assert labels_900[3] == "surface_rest"
    assert not any("walk" in label.lower() for t in range(1200) for label in condition_switch_labels(t, 12))

    before = stress_window_labels(499, 12, affected_nodes=[1], start=500, end=650)
    inside = stress_window_labels(500, 12, affected_nodes=[1], start=500, end=650)
    after = stress_window_labels(650, 12, affected_nodes=[1], start=500, end=650)
    assert before[1] == "implant30_rest"
    assert inside[1] == "implant30_stress"
    assert inside[0] == "implant10_rest"
    assert inside[2] == "implant30_rest"
    assert inside[3] == "surface_rest"
    assert after[1] == "implant30_rest"


def test_metrics_use_unified_definitions_and_raw_em():
    from hfss_paper_results.config import SimConfig
    from hfss_paper_results.metrics import summarize_raw

    cfg = SimConfig(T=4, stress_start=1, stress_end=3, stress_nodes=(1,))
    rows = []
    for t in range(4):
        for node in range(3):
            rows.append(
                {
                    "scenario": "stress",
                    "seed": 0,
                    "method": "Proposed",
                    "t": t,
                    "node": node,
                    "node_type": "implant30" if node in {1, 2} else "implant10",
                    "is_critical": True,
                    "is_stressed": node == 1 and 1 <= t < 3,
                    "q": 2.0 + t,
                    "e": 0.36,
                    "e_next": 0.34 if node == 1 and 1 <= t < 3 else 0.50,
                    "E_min": 0.35,
                    "E_max": 1.0,
                    "uD": 0.1,
                    "uE": 0.2,
                    "mu": 0.5,
                    "xi": 0.035 if node == 1 and 1 <= t < 3 else 0.0,
                    "served": 0.5,
                    "arrived": 1.0,
                    "em_utilization": 1.25 if t == 2 else 0.75,
                    "em_violation": t == 2,
                    "rx_cap_ratio": 1.10 if t == 2 and node == 1 else 0.60,
                    "rx_cap_violation": t == 2 and node == 1,
                    "lp_failed": False,
                }
            )
    raw = pd.DataFrame(rows)

    metrics = summarize_raw(raw, cfg, scenario="stress")
    row = metrics.by_method["Proposed"]

    assert np.isclose(row["served_workload_ratio"], 0.5)
    assert np.isclose(row["stress_shortage_severity"], 0.1)
    assert np.isclose(row["stress_implant_shortage_rate"], 1.0)
    assert np.isclose(row["em_util_peak"], 1.25)
    assert row["em_violation_rate"] == 0.25
    assert row["rx_cap_violation_rate"] == 0.25
    assert row["rx_cap_peak_ratio"] == 1.10


def test_gate_checker_rejects_degenerate_candidate():
    from hfss_paper_results.gates import evaluate_candidate_gates

    methods = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware", "Oracle"]
    metrics = {
        method: {
            "served_workload_ratio": 0.80,
            "post_stress_served_ratio": 0.80,
            "surface_served_ratio": 0.80,
            "p95_backlog": 10.0,
            "post_stress_backlog_mean": 10.0,
            "surface_backlog_mean": 10.0,
            "stress_shortage_severity": 0.10,
            "stress_implant_shortage_rate": 0.10,
            "stress_affected_energy_min": 0.40,
            "implant_energy_p05": 0.40,
            "implant_energy_margin_p05": 0.10,
            "recovery_time": 100.0,
            "em_violation_rate": 0.0,
            "rx_cap_violation_rate": 0.0,
            "lp_failure_rate": 0.0,
            "em_util_mean": 0.99,
            "em_util_peak": 1.0,
        }
        for method in methods
    }
    metrics["w/o EM-bud."]["em_violation_rate"] = 0.10
    phase = pd.DataFrame(
        {
            "B_EM_multiplier": [0.8, 0.9, 1.0],
            "P_H_multiplier": [0.8, 0.9, 1.0],
            "stress_shortage_severity": [0.0, 0.0, 0.0],
        }
    )

    report = evaluate_candidate_gates(metrics, phase)

    assert not report.passed
    assert "G7" in report.failed_codes
    assert "F7-G2" in report.failed_codes


def test_failed_candidate_does_not_release_final_figures(tmp_path):
    from hfss_paper_results.reports import write_failed_report

    final_dir = tmp_path / "outputs" / "final"
    final_dir.mkdir(parents=True)
    stale = final_dir / "fig3_main_comparison.pdf"
    stale.write_text("stale", encoding="utf-8")

    write_failed_report(final_dir, stage="unit", failed_codes=["F3-G1"], reason="gate failed")

    assert not stale.exists()
    report = final_dir / "failed_report.md"
    assert report.exists()
    assert "F3-G1" in report.read_text(encoding="utf-8")


def test_gate_pass_allows_manifest_payload(tmp_path):
    from hfss_paper_results.reports import write_json

    payload = {"workpoint_id": "ralph_11", "all_figures_from_same_workpoint": True}
    out = tmp_path / "candidate_metrics.json"
    write_json(out, payload)

    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_smoke_runner_writes_candidate_artifacts(tmp_path, monkeypatch):
    from hfss_paper_results.run import run_pipeline

    sched_path = _write_synthetic_sched_library(tmp_path / "calibrated_library_sched.csv")
    monkeypatch.setenv("HFSS_SCHED_LIBRARY", str(sched_path))

    run_pipeline(mode="smoke", output_root=tmp_path)

    candidate_dir = tmp_path / "outputs" / "candidates" / "smoke"
    assert (candidate_dir / "candidate_metrics.json").exists()
    assert (candidate_dir / "candidate_pass_fail.csv").exists()
    assert (candidate_dir / "figure_gate_report.md").exists()
    assert (candidate_dir / "gate_result.json").exists()
    assert (candidate_dir / "quick_fig3.png").exists()
    assert (candidate_dir / "quick_fig5.png").exists()


def test_forced_fail_runner_writes_failed_report_only(tmp_path):
    from hfss_paper_results.run import run_pipeline

    run_pipeline(mode="force_fail", output_root=tmp_path)

    final_dir = tmp_path / "outputs" / "final"
    assert (final_dir / "failed_report.md").exists()
    assert not (final_dir / "fig3_main_comparison.pdf").exists()


def test_fig4_mechanism_frame_uses_proposed_only_for_em_budget_panel():
    from hfss_paper_results.config import SimConfig
    from hfss_paper_results.plotting import FIG4_TITLES, build_fig4_mechanism_frames

    cfg = SimConfig(T=2)
    rows = []
    for method in ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o crit.-urg."]:
        for t in range(2):
            for node in range(4):
                is_critical = node < 2
                rows.append(
                    {
                        "method": method,
                        "t": t,
                        "node": node,
                        "node_type": "implant30" if is_critical else "surface",
                        "is_critical": is_critical,
                        "regime": "rest",
                        "uE": 0.2 if is_critical else 0.1,
                        "em_utilization": 0.6 if method == "Proposed" else 0.9,
                        "e_next": 0.5,
                        "E_min": 0.35 if is_critical else 0.0,
                        "E_max": 1.0,
                    }
                )
    frames = build_fig4_mechanism_frames(pd.DataFrame(rows), cfg)

    assert set(frames["em"]["method"]) == {"Proposed", "Budget limit = 1"}
    assert not {"ADT-MAC", "Lyap.-DPP"}.intersection(set(frames["em"]["method"]))
    assert FIG4_TITLES == {
        "allocation": "Proposed energy-support allocation by node class",
        "em": "Proposed EM-budget usage",
        "margin": "Critical implant energy margin",
    }


def test_fig6_uses_served_workload_ratio_instead_of_recovery_time():
    from hfss_paper_results.plotting import FIG6_PANELS

    panel_metrics = [metric for metric, _ in FIG6_PANELS]

    assert "served_workload_ratio" in panel_metrics
    assert "recovery_time" not in panel_metrics


def test_fig6_panels_have_mean_and_ci95_error_bars():
    from hfss_paper_results.config import ABLATION_METHODS
    from hfss_paper_results.plotting import FIG6_PANELS, bar_values_and_ci95

    frame = pd.DataFrame(
        [
            {
                "method": method,
                **{
                    metric: float(index + 1)
                    for index, (metric, _) in enumerate(FIG6_PANELS)
                },
                **{
                    f"{metric}_ci95": 0.1 * float(index + 1)
                    for index, (metric, _) in enumerate(FIG6_PANELS)
                },
            }
            for method in ABLATION_METHODS
        ]
    )

    for index, (metric, _) in enumerate(FIG6_PANELS):
        values, ci95 = bar_values_and_ci95(frame, metric, ABLATION_METHODS)
        assert values == [float(index + 1)] * len(ABLATION_METHODS)
        assert ci95 == [0.1 * float(index + 1)] * len(ABLATION_METHODS)


def _write_final_redesign_sources(final_dir: Path) -> None:
    final_dir.mkdir(parents=True)
    main_methods = ["Proposed", "ADT-MAC", "Lyap.-DPP"]
    fig3_metrics = {
        "stress_shortage_severity": [0.02, 0.85, 0.03],
        "surface_served_ratio": [0.84, 0.85, 0.52],
        "p95_backlog": [174.0, 178.0, 192.0],
        "recovery_time": [0.0, 58.0, 0.0],
    }
    pd.DataFrame(
        [
            {
                "method": method,
                **{metric: values[index] for metric, values in fig3_metrics.items()},
                **{f"{metric}_ci95": 0.01 for metric in fig3_metrics},
            }
            for index, method in enumerate(main_methods)
        ]
    ).to_csv(final_dir / "fig3_main_comparison.csv", index=False)

    fig4_rows = []
    for t, regime in [(0, "rest"), (1, "surface_sweat"), (2, "recovery_rest")]:
        fig4_rows.extend(
            [
                {"panel": "allocation", "method": "Proposed", "curve": "Critical implant nodes", "t": t, "regime": regime, "value": 0.12 + 0.02 * t},
                {"panel": "allocation", "method": "Proposed", "curve": "Surface nodes", "t": t, "regime": regime, "value": 0.04 - 0.01 * min(t, 1)},
                {"panel": "em", "method": "Proposed", "curve": "Proposed raw EM utilization", "t": t, "regime": regime, "value": 0.60 + 0.05 * t},
                {"panel": "em", "method": "Budget limit = 1", "curve": "Budget limit = 1", "t": t, "regime": regime, "value": 1.0},
                {"panel": "margin", "method": "Proposed", "curve": "Proposed", "t": t, "regime": regime, "value": 0.35 - 0.05 * t},
                {"panel": "margin", "method": "w/o crit.-urg.", "curve": "w/o crit.-urg.", "t": t, "regime": regime, "value": 0.25 - 0.12 * t},
            ]
        )
    pd.DataFrame(fig4_rows).to_csv(final_dir / "fig4_condition_switching_response.csv", index=False)

    stress_methods = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o crit.-urg."]
    pd.DataFrame(
        [
            {
                "method": method,
                "t": t,
                "e_next": 0.60 - (0.18 if method in {"ADT-MAC", "w/o crit.-urg."} and t == 1 else 0.02 * t),
                "xi": 0.35 if method in {"ADT-MAC", "w/o crit.-urg."} and t == 1 else 0.0,
                "severity": 1.0 if method in {"ADT-MAC", "w/o crit.-urg."} and t == 1 else 0.0,
            }
            for method in stress_methods
            for t in range(3)
        ]
    ).to_csv(final_dir / "fig5_implant_stress_response.csv", index=False)

    ablation_methods = ["Proposed", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware"]
    fig6_metrics = {
        "stress_shortage_severity": [0.02, 0.0, 0.82, 0.68],
        "implant_energy_p05": [0.54, 0.55, 0.44, 0.13],
        "em_violation_rate": [0.0, 1.0, 0.0, 0.0],
        "served_workload_ratio": [0.59, 0.60, 0.61, 0.40],
    }
    pd.DataFrame(
        [
            {
                "method": method,
                **{metric: values[index] for metric, values in fig6_metrics.items()},
                **{f"{metric}_ci95": 0.01 for metric in fig6_metrics},
            }
            for index, method in enumerate(ablation_methods)
        ]
    ).to_csv(final_dir / "fig6_ablation_hfss.csv", index=False)

    pd.DataFrame(
        [
            {"B_EM_multiplier": 0.9, "P_H_multiplier": 0.9, "stress_shortage_severity": 0.8, "is_final_workpoint": False},
            {"B_EM_multiplier": 1.0, "P_H_multiplier": 0.9, "stress_shortage_severity": 0.2, "is_final_workpoint": False},
            {"B_EM_multiplier": 0.9, "P_H_multiplier": 1.0, "stress_shortage_severity": 0.5, "is_final_workpoint": False},
            {"B_EM_multiplier": 1.0, "P_H_multiplier": 1.0, "stress_shortage_severity": 0.01, "is_final_workpoint": True},
        ]
    ).to_csv(final_dir / "fig7_phase_diagram_implant_shortage.csv", index=False)


def test_scheme_a_palette_matches_color_design():
    from hfss_paper_results.plotting import SCHEME_A_LINESTYLES, SCHEME_A_METHOD_COLORS

    assert SCHEME_A_METHOD_COLORS == {
        "Proposed": "#1B4F8A",
        "ADT-MAC": "#B03A2E",
        "Lyap.-DPP": "#1E8449",
        "w/o EM-bud.": "#6FA3D0",
        "w/o crit.-urg.": "#9DB8CE",
        "w/o implant-aware": "#7F7F7F",
        "Oracle": "#333333",
    }
    assert SCHEME_A_LINESTYLES["Proposed"] == "-"
    assert SCHEME_A_LINESTYLES["ADT-MAC"] == "--"
    assert SCHEME_A_LINESTYLES["Lyap.-DPP"] == "-."
    assert SCHEME_A_LINESTYLES["w/o EM-bud."] == ":"
    assert SCHEME_A_LINESTYLES["w/o crit.-urg."] == (0, (5, 2))


def test_redesign_final_creates_new_directory_and_preserves_originals(tmp_path):
    from hfss_paper_results.run import run_pipeline

    final_dir = tmp_path / "outputs" / "final"
    _write_final_redesign_sources(final_dir)
    original_pdf = final_dir / "fig3_main_comparison.pdf"
    original_pdf.write_text("original-pdf", encoding="utf-8")

    run_pipeline(mode="redesign_final", output_root=tmp_path)

    redesign_dirs = sorted((tmp_path / "outputs").glob("final_ieee_scheme_a_*"))
    assert len(redesign_dirs) == 1
    redesign_dir = redesign_dirs[0]
    assert original_pdf.read_text(encoding="utf-8") == "original-pdf"

    stems = [
        "fig3_main_comparison",
        "fig4_condition_switching_response",
        "fig5_implant_stress_response",
        "fig6_ablation_hfss",
        "fig7_phase_diagram_implant_shortage",
    ]
    for stem in stems:
        assert (redesign_dir / f"{stem}.pdf").exists()
        assert (redesign_dir / f"{stem}.png").exists()
        assert (redesign_dir / f"{stem}.csv").exists()

    manifest = redesign_dir / "redesign_manifest.md"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "Scheme A" in text
    assert "ieee_trans.mplstyle" in text


def test_redesign_style_matches_final_manuscript_typography():
    from hfss_paper_results import plotting

    source = Path(plotting.__file__).read_text(encoding="utf-8")
    assert plotting.BODY_TEXT_SIZE == 8
    assert plotting.PANEL_TITLE_SIZE == 9
    assert plotting.FIG4_ALLOCATION_YMAX >= 0.23
    assert plotting.FIG4A_LEGEND_TEXT_X <= 420
    assert plotting.FIG5_SHORTAGE_YMAX >= 1.20
    assert '"legend.frameon": False' in source

    fig4_body = source.split("def _plot_redesigned_fig4", 1)[1].split("\ndef _stress_window_from_frame", 1)[0]
    assert "FIG4A_LEGEND_LINE_X0" in fig4_body
    assert "Critical implant nodes" in fig4_body
    assert "axes[0].legend" not in fig4_body
    assert "loc=\"best\"" not in fig4_body

    fig5_body = source.split("def _plot_redesigned_fig5", 1)[1].split("\ndef _plot_redesigned_fig7", 1)[0]
    assert "FIG5_SHORTAGE_YMAX" in fig5_body
    assert "stress window" in fig5_body
    assert "loc=\"upper center\"" in fig5_body
