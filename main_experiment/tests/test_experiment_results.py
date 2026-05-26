from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCHEDULER_LABELS = [
    "surface_rest",
    "surface_moderate_loose",
    "surface_contact_failure",
    "surface_sweat",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
]


def _write_scheduler_library(path: Path, labels: list[str] | None = None) -> None:
    coefficients = {
        "surface_rest": 1.0,
        "surface_moderate_loose": 0.55,
        "surface_contact_failure": 0.02,
        "surface_sweat": 1.25,
        "implant10_rest": 0.70,
        "implant10_stress": 0.35,
        "implant30_rest": 0.45,
        "implant30_stress": 0.20,
    }
    rows = []
    for label in labels or SCHEDULER_LABELS:
        depth_mm = 0.0 if label.startswith("surface") else float(label.removeprefix("implant").split("_", 1)[0])
        rows.append(
            {
                "label": label,
                "type": "surface" if label.startswith("surface") else label.split("_", 1)[0],
                "depth_mm": depth_mm,
                "condition": label.split("_", 1)[1],
                "sched_include_main": label != "surface_contact_failure",
                "g_norm": coefficients[label],
                "r_norm": coefficients[label],
                "chi_norm": 1.0,
                "p_rx_cap_norm": min(coefficients[label], 1.0),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_full_phase_summary_prefers_boundary_focused_grid(monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig, WorkPoint

    calls: list[tuple[float, float]] = []
    seen_seed_sets: list[tuple[int, ...]] = []

    def fake_phase_cell_metrics(cfg, wp, seeds, library, *, budget_multiplier, power_multiplier):
        calls.append((float(budget_multiplier), float(power_multiplier)))
        seen_seed_sets.append(tuple(seeds))
        boundary = (budget_multiplier - 0.50) + 0.8 * (power_multiplier - 0.55)
        if boundary > 0.04:
            severity = 0.0
        elif boundary >= -0.04:
            severity = 0.08
        else:
            severity = 0.65
        return {
            "stress_shortage_severity": severity,
            "stress_implant_shortage_rate": severity,
            "served_workload_ratio": 0.5,
        }

    monkeypatch.setattr(run_module, "load_hfss_library", lambda: object())
    monkeypatch.setattr(run_module, "run_phase_cell_metrics", fake_phase_cell_metrics)

    phase = run_module._phase_summary(SimConfig(), WorkPoint(), [0, 1, 2], quick=False)

    assert len(phase) == 63
    assert sorted(phase["B_EM_multiplier"].unique().round(2).tolist()) == [0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.55, 0.58, 0.6]
    assert sorted(phase["P_H_multiplier"].unique().round(2).tolist()) == [0.45, 0.5, 0.55, 0.58, 0.6, 0.62, 0.65]
    assert len(calls) == 63
    assert set(seen_seed_sets) == {(0,)}


def test_phase_summary_falls_back_to_low_power_grid_when_library_is_too_feasible(monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig, WorkPoint

    calls: list[tuple[float, float]] = []

    def fake_phase_cell_metrics(cfg, wp, seeds, library, *, budget_multiplier, power_multiplier):
        calls.append((float(budget_multiplier), float(power_multiplier)))
        if power_multiplier <= 0.25:
            severity = 0.60
        elif power_multiplier <= 0.55:
            severity = 0.12
        else:
            severity = 0.0
        return {
            "stress_shortage_severity": severity,
            "stress_implant_shortage_rate": severity,
            "served_workload_ratio": 0.5,
        }

    monkeypatch.setattr(run_module, "load_hfss_library", lambda: object())
    monkeypatch.setattr(run_module, "run_phase_cell_metrics", fake_phase_cell_metrics)

    phase = run_module._phase_summary(SimConfig(), WorkPoint(), [0, 1, 2], quick=False)

    assert phase["P_H_multiplier"].min() <= 0.25
    assert phase["stress_shortage_severity"].gt(0.30).mean() >= 0.10
    assert phase["stress_shortage_severity"].between(0.03, 0.30).mean() >= 0.20


def test_hfss_loader_uses_only_final_sched_library(monkeypatch):
    from Experiment_results.data import SCHED_LIBRARY_PATH, load_hfss_library

    monkeypatch.delenv("IOTJ_HFSS_SCHED_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("IOTJ_HFSS_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("HFSS_SCHED_LIBRARY_PATH", raising=False)

    library = load_hfss_library()

    assert library.source_path == SCHED_LIBRARY_PATH
    assert library.source_path.name == "torso_scheduler_library.csv"
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


@pytest.mark.parametrize("filename", ["torso_scheduler_library.csv", "torso_proxy_library.csv"])
def test_hfss_loader_accepts_explicit_torso_scheduler_libraries(tmp_path, filename):
    from Experiment_results.data import load_hfss_library

    library_path = tmp_path / filename
    _write_scheduler_library(library_path)

    library = load_hfss_library(library_path)

    assert library.source_path == library_path
    assert set(library.rows) == set(SCHEDULER_LABELS)
    assert library.rows["surface_contact_failure"].sched_include_main is False


def test_hfss_loader_uses_opt_in_environment_library_path(tmp_path, monkeypatch):
    from Experiment_results.data import load_hfss_library

    library_path = tmp_path / "torso_proxy_library.csv"
    _write_scheduler_library(library_path)
    monkeypatch.setenv("IOTJ_HFSS_SCHED_LIBRARY_PATH", str(library_path))

    library = load_hfss_library()

    assert library.source_path == library_path


def test_hfss_loader_requires_canonical_labels_for_explicit_alternate_library(tmp_path):
    from Experiment_results.data import load_hfss_library

    library_path = tmp_path / "torso_scheduler_library.csv"
    _write_scheduler_library(library_path, labels=[label for label in SCHEDULER_LABELS if label != "implant30_stress"])

    with pytest.raises(ValueError, match="missing labels"):
        load_hfss_library(library_path)


def test_revision_library_builders_preserve_labels_and_expected_contrasts():
    from Experiment_results.data import (
        load_g_scaled_library,
        load_coarse_2state_library,
        load_hfss_library,
        load_homogenized_library,
        load_linkbudget_library,
    )

    full = load_hfss_library()
    homogenized = load_homogenized_library()
    linkbudget = load_linkbudget_library()
    coarse = load_coarse_2state_library()

    assert set(homogenized.rows) == set(full.rows)
    assert set(linkbudget.rows) == set(full.rows)
    assert set(coarse.rows) == set(full.rows)

    rest = full.rows["surface_rest"]
    for label, row in homogenized.rows.items():
        assert row.label == label
        assert row.g_norm == rest.g_norm
        assert row.r_norm == rest.r_norm
        assert row.chi_norm == rest.chi_norm
        assert row.p_rx_cap_norm == rest.p_rx_cap_norm

    assert linkbudget.rows["surface_rest"].g_norm == 1.0
    assert linkbudget.rows["implant10_rest"].g_norm < linkbudget.rows["surface_rest"].g_norm
    assert linkbudget.rows["implant30_rest"].g_norm < linkbudget.rows["implant10_rest"].g_norm
    assert all(row.chi_norm == 1.0 for row in linkbudget.rows.values())

    assert coarse.rows["implant10_rest"].g_norm == coarse.rows["implant30_rest"].g_norm
    assert coarse.rows["implant10_stress"].g_norm == coarse.rows["implant30_stress"].g_norm
    assert coarse.rows["surface_sweat"].g_norm == coarse.rows["surface_moderate_loose"].g_norm

    g_scaled = load_g_scaled_library(0.7)
    assert set(g_scaled.rows) == set(full.rows)
    assert g_scaled.source_path.name == "<g-scaled-0.700>"
    assert g_scaled.rows["implant30_stress"].g_norm == pytest.approx(full.rows["implant30_stress"].g_norm * 0.7)
    assert g_scaled.rows["implant30_stress"].r_norm == full.rows["implant30_stress"].r_norm
    assert g_scaled.rows["implant30_stress"].chi_norm == full.rows["implant30_stress"].chi_norm
    assert g_scaled.rows["implant30_stress"].p_rx_cap_norm == full.rows["implant30_stress"].p_rx_cap_norm


def test_run_method_can_use_separate_decision_and_physical_libraries():
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.data import load_hfss_library, load_homogenized_library
    from Experiment_results.sim import run_method

    cfg = SimConfig(T=4, stress_start=1, stress_end=3, post_stress_window=1, recovery_guard_slots=1)
    raw = run_method(
        cfg,
        WorkPoint(),
        "Proposed",
        0,
        "stress",
        load_hfss_library(),
        decision_library=load_homogenized_library(),
    )

    assert "decision_g" in raw.columns
    assert "g" in raw.columns
    stressed = raw[(raw["t"].eq(1)) & raw["node"].eq(cfg.primary_stress_node)].iloc[0]
    assert stressed["label_true"].endswith("_stress")
    assert stressed["decision_g"] != stressed["g"]


def test_revision_experiment_modes_write_expected_csvs(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    for mode, expected in [
        ("necessity_ablation", "necessity_ablation.csv"),
        ("lyap_sweep", "lyap_v_sweep.csv"),
        ("small_network", "small_network_summary.csv"),
        ("estimator_prototype", "estimator_prototype_summary.csv"),
        ("calibrated_estimator", "estimator_closed_loop_summary.csv"),
        ("label_robustness", "label_robustness_summary.csv"),
        ("network_scale", "network_scale_summary.csv"),
        ("necessity_audit", "necessity_audit_summary.csv"),
        ("robustness_boundary", "robustness_boundary_summary.csv"),
        ("rssi_noise_sweep", "rssi_noise_sweep_summary.csv"),
        ("phase2_coarse_library", "output_phase3_coarse_ablation.csv"),
        ("baseline_extra", "baseline_extra_summary.csv"),
        ("heterogeneous_stress", "heterogeneous_implant_summary.csv"),
        ("lp_tol_sweep", "lp_tol_sweep.csv"),
        ("stress_window_sweep", "stress_window_sweep.csv"),
        ("workpoint_cv", "workpoint_cv_summary.csv"),
        ("proxy_perturbation_sweep", "proxy_perturbation_sweep.csv"),
        ("lambda_s_sweep", "lambda_s_sweep.csv"),
        ("phase3_30_seeds", "phase3_30_seed_main_summary.csv"),
    ]:
        run_module.run_pipeline(mode=mode, output_root=tmp_path / mode)
        assert (tmp_path / mode / "output" / expected).exists()


def test_revision_experiment_mode_aliases_write_expected_csvs(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    for mode, expected in [
        ("lyap_v_sweep", "lyap_v_sweep.csv"),
        ("state_error", "estimator_prototype_summary.csv"),
    ]:
        run_module.run_pipeline(mode=mode, output_root=tmp_path / mode)
        assert (tmp_path / mode / "output" / expected).exists()


def test_lyap_v_sweep_marks_nominal_feasibility_split(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="lyap_sweep", output_root=tmp_path)
    frame = pd.read_csv(tmp_path / "output" / "lyap_v_sweep.csv")

    assert "feasible" in frame.columns
    assert frame.loc[frame["V_scale"].le(1.0), "feasible"].all()
    assert not frame.loc[frame["V_scale"].gt(1.0), "feasible"].any()


def test_calibrated_estimator_outputs_closed_loop_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="calibrated_estimator", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "estimator_closed_loop_summary.csv")
    calibration = pd.read_csv(tmp_path / "output" / "estimator_calibration_grid.csv")
    confusion = pd.read_csv(tmp_path / "output" / "estimator_label_confusion.csv")

    for column in ["estimator", "label_accuracy", "stress_false_negative_rate", "stress_shortage_severity", "em_violation_rate", "rx_cap_violation_rate"]:
        assert column in summary.columns
    assert {"estimator", "split", "objective", "label_accuracy", "stress_false_negative_rate"}.issubset(calibration.columns)
    assert {"label_true", "label_decision", "count"}.issubset(confusion.columns)


def test_label_robustness_outputs_four_panel_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="label_robustness", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "label_robustness_summary.csv")

    assert {"panel", "perturbation", "method", "stress_shortage_severity", "surface_served_ratio", "em_violation_rate"}.issubset(summary.columns)
    assert set(summary["panel"]) == {"random_flip", "stress_false_negative", "stuck_mixed_burst", "rssi_calibrated"}


def test_network_scale_outputs_small_mid_full_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="network_scale", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "network_scale_summary.csv")

    assert {"topology", "num_surface", "num_implant", "method", "stress_shortage_severity", "surface_served_ratio"}.issubset(summary.columns)
    assert {"small_3s_1i", "mid_5s_2i", "full_9s_3i"}.issubset(set(summary["topology"]))


def test_necessity_audit_outputs_paired_stats_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="necessity_audit", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "necessity_audit_summary.csv")
    stats = pd.read_csv(tmp_path / "output" / "necessity_audit_wilcoxon.csv")

    assert {"library", "stress_shortage_severity", "em_violation_rate", "rx_cap_violation_rate", "recovery_censored", "surface_served_ratio"}.issubset(summary.columns)
    assert {"comparison", "metric", "p_value", "delta_mean"}.issubset(stats.columns)


def test_robustness_boundary_outputs_extended_panels(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="robustness_boundary", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "robustness_boundary_summary.csv")

    assert {"panel", "perturbation", "stress_shortage_severity", "surface_served_ratio", "em_violation_rate"}.issubset(summary.columns)
    assert set(summary["panel"]) == {"random_flip", "stress_false_negative", "stuck_mixed_burst"}
    assert summary["perturbation"].astype(str).isin({"0.5", "0.50", "0.9", "0.90", "200"}).any()


def test_rssi_noise_sweep_outputs_noise_levels(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="rssi_noise_sweep", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "rssi_noise_sweep_summary.csv")

    assert {"noise_db", "label_accuracy", "stress_false_negative_rate", "stress_shortage_severity", "em_violation_rate"}.issubset(summary.columns)
    assert {0.5, 1.0, 2.0, 5.0}.issubset(set(summary["noise_db"].astype(float)))


def test_phase3_coarse_library_outputs_fine_coarse_and_fallback(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="phase2_coarse_library", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "output_phase3_coarse_ablation.csv")
    by_seed = pd.read_csv(tmp_path / "output" / "output_phase3_coarse_ablation_by_seed.csv")

    assert {"variant", "num_seeds", "surface_served_ratio", "p95_backlog", "recovery_censored", "stress_shortage_severity", "em_violation_rate", "rx_cap_violation_rate"}.issubset(summary.columns)
    assert {"torso_fine_grained", "torso_coarse_2state", "single_state_fallback"}.issubset(set(summary["variant"]))
    assert {"variant", "seed", "stress_shortage_severity"}.issubset(by_seed.columns)


def test_phase3_extra_baseline_outputs_round_robin_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="baseline_extra", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "baseline_extra_summary.csv")

    assert {"method", "stress_shortage_severity", "surface_served_ratio", "em_violation_rate"}.issubset(summary.columns)
    assert "Round-Robin" in set(summary["method"])


def test_phase3_heterogeneous_stress_outputs_fairness_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="heterogeneous_stress", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "heterogeneous_implant_summary.csv")

    assert {"topology", "method", "worst_implant_margin_p05", "implant_margin_std", "implant_fairness_jain"}.issubset(summary.columns)
    assert {"mid_5s_2i", "full_9s_3i"}.issubset(set(summary["topology"]))


def test_phase3_lp_and_stress_window_sweeps_output_contracts(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="lp_tol_sweep", output_root=tmp_path / "tol")
    tol = pd.read_csv(tmp_path / "tol" / "output" / "lp_tol_sweep.csv")
    assert {"lp_tolerance", "stress_shortage_severity", "em_violation_rate", "solve_time_ms_mean", "solve_time_ms_p95"}.issubset(tol.columns)

    run_module.run_pipeline(mode="stress_window_sweep", output_root=tmp_path / "window")
    window = pd.read_csv(tmp_path / "window" / "output" / "stress_window_sweep.csv")
    assert {"stress_window_length", "stress_shortage_severity", "surface_served_ratio"}.issubset(window.columns)


def test_phase3_workpoint_cv_and_wilcoxon_helper_contract(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig
    from Experiment_results.metrics import paired_wilcoxon_pvalue

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    assert 0.0 <= paired_wilcoxon_pvalue(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.0, 1.0, 1.0])) <= 1.0
    run_module.run_pipeline(mode="workpoint_cv", output_root=tmp_path)
    summary = pd.read_csv(tmp_path / "output" / "workpoint_cv_summary.csv")
    assert {"candidate", "distance_from_final", "stress_shortage_severity", "surface_served_ratio"}.issubset(summary.columns)


def test_tier2_proxy_and_lambda_s_sweeps_output_contracts(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))

    run_module.run_pipeline(mode="proxy_perturbation_sweep", output_root=tmp_path / "proxy")
    proxy = pd.read_csv(tmp_path / "proxy" / "output" / "proxy_perturbation_sweep.csv")
    proxy_by_seed = pd.read_csv(tmp_path / "proxy" / "output" / "proxy_perturbation_sweep_by_seed.csv")
    assert {"g_norm_scale", "stress_shortage_severity", "surface_served_ratio", "em_violation_rate", "rx_cap_violation_rate"}.issubset(proxy.columns)
    assert {0.7, 0.8, 0.9, 1.0, 1.1}.issubset(set(proxy["g_norm_scale"].astype(float)))
    assert {"g_norm_scale", "seed", "stress_shortage_severity"}.issubset(proxy_by_seed.columns)

    run_module.run_pipeline(mode="lambda_s_sweep", output_root=tmp_path / "lambda_s")
    lambda_s = pd.read_csv(tmp_path / "lambda_s" / "output" / "lambda_s_sweep.csv")
    lambda_s_by_seed = pd.read_csv(tmp_path / "lambda_s" / "output" / "lambda_s_sweep_by_seed.csv")
    assert {"lambda_s", "stress_shortage_severity", "surface_served_ratio", "implant_energy_margin_p05"}.issubset(lambda_s.columns)
    assert {5.0, 10.0, 20.0, 50.0}.issubset(set(lambda_s["lambda_s"].astype(float)))
    assert {"lambda_s", "seed", "implant_energy_margin_p05"}.issubset(lambda_s_by_seed.columns)


def test_train_holdout_seed_constants_are_disjoint():
    from Experiment_results.run import TEST_SEEDS, TRAIN_SEEDS

    assert TRAIN_SEEDS == list(range(10))
    assert TEST_SEEDS == list(range(10, 20))
    assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)


def test_scenarios_use_hfss_labels_and_no_walk():
    from Experiment_results.scenarios import condition_switch_labels, stress_window_labels

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


def test_two_implant_topology_labels_only_first_two_nodes_as_implants():
    from Experiment_results.scenarios import node_type, stress_window_labels

    assert node_type(0, num_critical=2) == "implant10"
    assert node_type(1, num_critical=2) == "implant30"
    assert node_type(2, num_critical=2) == "surface"

    inside = stress_window_labels(10, 7, affected_nodes=(1,), start=0, end=20, num_critical=2)

    assert inside[0] == "implant10_rest"
    assert inside[1] == "implant30_stress"
    assert inside[2] == "surface_rest"


def test_metrics_use_unified_definitions_and_raw_em():
    from Experiment_results.config import SimConfig
    from Experiment_results.metrics import summarize_raw

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


def _raw_metric_frame(methods: list[str], em_values: list[float]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for t, em_utilization in enumerate(em_values):
            rows.append(
                {
                    "scenario": "condition",
                    "seed": 0,
                    "method": method,
                    "t": t,
                    "node": 0,
                    "node_type": "implant10",
                    "is_critical": True,
                    "is_stressed": False,
                    "regime": "rest" if t == 0 else "surface_sweat",
                    "q": 1.0,
                    "e": 0.50,
                    "e_next": 0.52,
                    "E_min": 0.35,
                    "E_max": 1.0,
                    "uD": 0.1,
                    "uE": 0.2,
                    "mu": 0.5,
                    "xi": 0.0,
                    "served": 0.5,
                    "arrived": 1.0,
                    "em_utilization": em_utilization,
                    "em_violation": False,
                    "rx_cap_ratio": 0.5,
                    "rx_cap_violation": False,
                    "lp_failed": False,
                }
            )
    return pd.DataFrame(rows)


def test_compose_metrics_carries_condition_em_util_p95():
    from Experiment_results.config import SimConfig
    from Experiment_results.run import _compose_metrics

    cfg = SimConfig(T=4, stress_start=1, stress_end=3, post_stress_window=1, recovery_guard_slots=1)
    stress_raw = _raw_metric_frame(["Proposed"], [0.20, 0.20, 0.20, 0.20])
    ablation_raw = _raw_metric_frame(["w/o crit.-urg."], [0.30, 0.30, 0.30, 0.30])
    condition_raw = _raw_metric_frame(["Proposed"], [0.10, 0.20, 0.80, 1.00])

    metrics = _compose_metrics(cfg, condition_raw, stress_raw, ablation_raw)

    assert np.isclose(metrics["Proposed"]["em_util_p95"], pd.Series([0.10, 0.20, 0.80, 1.00]).quantile(0.95))


def test_condition_response_ratio_detects_surface_allocation_changes():
    from Experiment_results.run import _condition_response_ratio

    rows = []
    for regime, surface_uE in [("rest", 0.05), ("surface_sweat", 0.12), ("surface_moderate_loose", 0.16)]:
        rows.append({"method": "Proposed", "regime": regime, "is_critical": True, "uE": 0.20})
        rows.append({"method": "Proposed", "regime": regime, "is_critical": False, "uE": surface_uE})

    assert _condition_response_ratio(pd.DataFrame(rows)) > 0.50


def test_gate_checker_rejects_degenerate_candidate():
    from Experiment_results.gates import evaluate_candidate_gates

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


def _passing_gate_metrics(*, proposed_mean: float = 0.21, proposed_p95: float = 0.99) -> dict[str, dict[str, float]]:
    methods = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware", "Oracle"]
    metrics = {
        method: {
            "served_workload_ratio": 0.80,
            "post_stress_served_ratio": 0.80,
            "surface_served_ratio": 0.75,
            "p95_backlog": 100.0,
            "post_stress_backlog_mean": 10.0,
            "surface_backlog_mean": 10.0,
            "stress_shortage_severity": 0.0,
            "stress_implant_shortage_rate": 0.0,
            "stress_affected_energy_min": 0.40,
            "implant_energy_p05": 0.40,
            "implant_energy_margin_p05": 0.10,
            "recovery_time": 0.0,
            "em_violation_rate": 0.0,
            "rx_cap_violation_rate": 0.0,
            "lp_failure_rate": 0.0,
            "em_util_mean": 0.50,
            "em_util_p95": 0.80,
            "em_util_peak": 1.0,
            "fig4_condition_response_ratio": 0.20,
            "pre_stress_below_min_rate": 0.0,
        }
        for method in methods
    }
    metrics["Proposed"].update({"em_util_mean": proposed_mean, "em_util_p95": proposed_p95})
    metrics["ADT-MAC"].update({"stress_shortage_severity": 0.80, "stress_implant_shortage_rate": 0.80, "recovery_time": 12.0})
    metrics["Lyap.-DPP"].update({"surface_served_ratio": 0.02, "served_workload_ratio": 0.40, "p95_backlog": 120.0})
    metrics["w/o EM-bud."].update({"em_violation_rate": 0.06, "em_util_peak": 1.20})
    metrics["w/o crit.-urg."].update({"stress_shortage_severity": 0.45, "implant_energy_p05": 0.30, "recovery_time": 10.0})
    metrics["w/o implant-aware"].update({"surface_served_ratio": 0.56, "surface_backlog_mean": 10.2})
    metrics["Oracle"].update({"stress_shortage_severity": 0.0})
    return metrics


def _passing_phase_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "B_EM_multiplier": list(range(10)),
            "P_H_multiplier": list(range(10)),
            "stress_shortage_severity": [0.0, 0.0, 0.01, 0.04, 0.12, 0.49, 0.51, 0.70, 0.80, 0.90],
        }
    )


def test_gate_checker_accepts_bursty_torso_em_utilization():
    from Experiment_results.gates import evaluate_candidate_gates

    report = evaluate_candidate_gates(_passing_gate_metrics(proposed_mean=0.21, proposed_p95=0.99), _passing_phase_frame())

    assert "F4-G3" not in report.failed_codes
    assert "F4-G3a" not in report.failed_codes
    assert "F4-G3b" not in report.failed_codes


def test_gate_checker_rejects_proposed_em_headroom_loss():
    from Experiment_results.gates import evaluate_candidate_gates

    report = evaluate_candidate_gates(_passing_gate_metrics(proposed_mean=0.97, proposed_p95=1.0), _passing_phase_frame())

    assert "F4-G3b" in report.failed_codes


def test_phase_counts_treats_torso_transition_severity_as_medium():
    from Experiment_results.gates import _phase_counts

    low, medium, high = _phase_counts(_passing_phase_frame())

    assert np.isclose(low, 0.30)
    assert np.isclose(medium, 0.30)
    assert np.isclose(high, 0.40)


def test_failed_candidate_does_not_release_final_figures(tmp_path):
    from Experiment_results.reports import write_failed_report

    final_dir = tmp_path / "output"
    final_dir.mkdir(parents=True)
    stale = final_dir / "fig3_main_comparison.pdf"
    stale.write_text("stale", encoding="utf-8")

    write_failed_report(final_dir, stage="unit", failed_codes=["F3-G1"], reason="gate failed")

    assert not stale.exists()
    report = final_dir / "failed_report.md"
    assert report.exists()
    assert "F3-G1" in report.read_text(encoding="utf-8")


def test_gate_pass_allows_manifest_payload(tmp_path):
    from Experiment_results.reports import write_json

    payload = {"workpoint_id": "ralph_11", "all_figures_from_same_workpoint": True}
    out = tmp_path / "candidate_metrics.json"
    write_json(out, payload)

    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_smoke_runner_writes_candidate_artifacts(tmp_path):
    from Experiment_results.run import run_pipeline

    run_pipeline(mode="smoke", output_root=tmp_path)

    candidate_dir = tmp_path / "output" / "candidates" / "smoke"
    assert (candidate_dir / "candidate_metrics.json").exists()
    assert (candidate_dir / "candidate_pass_fail.csv").exists()
    assert (candidate_dir / "figure_gate_report.md").exists()
    assert (candidate_dir / "gate_result.json").exists()
    assert (candidate_dir / "quick_fig3.png").exists()
    assert (candidate_dir / "quick_fig5.png").exists()


def test_forced_fail_runner_writes_failed_report_only(tmp_path):
    from Experiment_results.run import run_pipeline

    run_pipeline(mode="force_fail", output_root=tmp_path)

    final_dir = tmp_path / "output"
    assert (final_dir / "failed_report.md").exists()
    assert not (final_dir / "fig3_main_comparison.pdf").exists()


def test_release_final_uses_output_directory(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.gates import GateReport

    written: list[Path] = []

    def fake_plot(_metrics, out_dir):
        out = Path(out_dir) / "figure.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("figure", encoding="utf-8")
        written.append(out)

    monkeypatch.setattr(run_module, "plot_fig3", fake_plot)
    monkeypatch.setattr(run_module, "plot_fig4", lambda _raw, _cfg, out_dir: fake_plot({}, out_dir))
    monkeypatch.setattr(run_module, "plot_fig5", lambda _raw, _cfg, out_dir: fake_plot({}, out_dir))
    monkeypatch.setattr(run_module, "plot_fig6", fake_plot)
    monkeypatch.setattr(run_module, "plot_fig7", lambda _phase, out_dir: fake_plot({}, out_dir))
    monkeypatch.setattr(run_module, "_write_final_reports", lambda *args, **kwargs: None)

    run_module._release_final(
        tmp_path,
        SimConfig(T=2),
        WorkPoint(),
        GateReport(True, [], [], 0.0),
        {},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        [],
    )

    assert written
    assert all(path.parent == tmp_path / "output" for path in written)


def test_phase_only_refreshes_existing_heldout_candidate(tmp_path, monkeypatch):
    from Experiment_results import run as run_module
    from Experiment_results.config import SimConfig

    candidate_dir = tmp_path / "output" / "candidates" / "heldout_final"
    candidate_dir.mkdir(parents=True)
    old_payload = {
        "candidate": "heldout_final",
        "seeds": [10, 11],
        "workpoint": {"lambda_q": 1.31, "lambda_c": 1.53, "lambda_e": 0.21, "lambda_xi": 0.39, "B_EM_ref": 0.84, "P_H": 1.14, "lambda_s": 20.0},
        "metrics": _passing_gate_metrics(),
        "phase": [{"B_EM_multiplier": 0.1, "P_H_multiplier": 0.1, "stress_shortage_severity": 1.0}],
    }
    (candidate_dir / "candidate_metrics.json").write_text(json.dumps(old_payload), encoding="utf-8")

    monkeypatch.setattr(run_module, "_diagnostic", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_module, "SimConfig", lambda *args, **kwargs: SimConfig(T=8, stress_start=2, stress_end=4, post_stress_window=2, recovery_guard_slots=1, *args, **kwargs))
    monkeypatch.setattr(run_module, "_phase_summary", lambda *_args, **_kwargs: _passing_phase_frame())
    monkeypatch.setattr(run_module, "plot_fig7", lambda phase, out_dir: Path(out_dir, "fig7_phase_diagram_implant_shortage.csv").write_text(phase.to_csv(index=False), encoding="utf-8"))

    run_module.run_pipeline(mode="phase_only", output_root=tmp_path)

    payload = json.loads((candidate_dir / "candidate_metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((candidate_dir / "gate_result.json").read_text(encoding="utf-8"))
    assert len(payload["phase"]) == len(_passing_phase_frame())
    assert gate["passed"] is True
    assert (tmp_path / "output" / "phase_only.csv").exists()


def test_fig4_mechanism_frame_uses_proposed_only_for_em_budget_panel():
    from Experiment_results.config import SimConfig
    from Experiment_results.plotting import FIG4_TITLES, build_fig4_mechanism_frames

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
    from Experiment_results.plotting import FIG6_PANELS

    panel_metrics = [metric for metric, _ in FIG6_PANELS]

    assert "served_workload_ratio" in panel_metrics
    assert "recovery_time" not in panel_metrics


def test_fig6_panels_have_mean_and_ci95_error_bars():
    from Experiment_results.config import ABLATION_METHODS
    from Experiment_results.plotting import FIG6_PANELS, bar_values_and_ci95

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
    from Experiment_results.plotting import SCHEME_A_LINESTYLES, SCHEME_A_METHOD_COLORS

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
    from Experiment_results.run import run_pipeline

    final_dir = tmp_path / "output"
    _write_final_redesign_sources(final_dir)
    original_pdf = final_dir / "fig3_main_comparison.pdf"
    original_pdf.write_text("original-pdf", encoding="utf-8")

    run_pipeline(mode="redesign_final", output_root=tmp_path)

    redesign_dirs = sorted((tmp_path / "output").glob("final_ieee_scheme_a_*"))
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
    from Experiment_results import plotting

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


def test_torso_asset_rebuild_marks_fig5_protected_margin_floor():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "rebuild_torso_manuscript_assets.py").read_text(encoding="utf-8")
    fig5_body = source.split("def write_fig5", 1)[1].split("\ndef write_fig6", 1)[0]

    assert "protected_margin_floor" in fig5_body
    assert "Protected margin floor" in fig5_body

