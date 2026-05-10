from __future__ import annotations

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


def test_slot_lp_solver_satisfies_linear_constraints():
    from hfss_paper_results.config import SimConfig, WorkPoint
    from hfss_paper_results.sim import Coefficients, solve_slot_lp

    cfg = SimConfig(num_nodes=3, num_critical=1)
    wp = WorkPoint(lambda_q=1.15, lambda_c=1.58, lambda_e=0.22, lambda_xi=0.36, B_EM_ref=0.55, P_H=1.10)
    decision = Coefficients(
        labels=["implant30_stress", "surface_rest", "surface_sweat"],
        r=np.array([0.8, 1.4, 1.8], dtype=float),
        g=np.array([0.30, 0.40, 0.50], dtype=float),
        eta=np.array([0.30, 0.40, 0.50], dtype=float),
        chi=np.array([0.90, 0.70, 0.80], dtype=float),
        p_rx_cap=np.array([0.40, 0.50, 0.60], dtype=float),
        B_EM=0.55,
    )
    e = np.array([0.40, 0.70, 0.65], dtype=float)
    q = np.array([6.0, 5.0, 3.0], dtype=float)
    rho = np.array([0.10, 0.09, 0.08], dtype=float)
    critical = np.array([True, False, False])

    solution = solve_slot_lp("Proposed", cfg, wp, decision, e, q, rho, critical)

    assert solution.success
    assert solution.backend == "lp"
    assert solution.u_d.shape == (3,)
    assert solution.u_e.shape == (3,)
    assert solution.mu.shape == (3,)
    assert solution.xi.shape == (3,)
    assert float(solution.u_d.sum()) <= 1.0 + 1e-8
    assert float(solution.u_e.sum()) <= 1.0 + 1e-8
    assert np.all(solution.mu <= q + 1e-8)
    assert np.all(solution.mu <= decision.r * solution.u_d + 1e-8)

    e_hat = np.maximum(e - cfg.E_base, 0.0)
    post_energy = e_hat + decision.eta * wp.P_H * solution.u_e - rho * solution.u_d
    assert np.all(post_energy >= -1e-8)
    assert np.all(post_energy <= cfg.E_max + 1e-8)
    assert post_energy[0] + solution.xi[0] >= cfg.E_min_critical - 1e-8
    assert np.all(decision.g * wp.P_H * solution.u_e <= decision.p_rx_cap + 1e-8)
    assert float(np.dot(decision.chi, solution.u_e)) <= decision.B_EM + 1e-8


def test_run_method_records_lp_backend_for_proposed(tmp_path):
    from hfss_paper_results.config import SimConfig, WorkPoint
    from hfss_paper_results.data import load_hfss_library
    from hfss_paper_results.sim import run_method

    cfg = SimConfig(T=4, stress_start=1, stress_end=3, post_stress_window=1, recovery_guard_slots=1)
    sched_path = _write_synthetic_sched_library(tmp_path / "calibrated_library_sched.csv")
    raw = run_method(cfg, WorkPoint(), "Proposed", 0, "stress", load_hfss_library(sched_path))

    assert set(raw["allocation_backend"]) == {"lp"}
    assert not raw["lp_failed"].any()


def test_slot_lp_solver_respects_protected_margin_when_feasible():
    from hfss_paper_results.config import SimConfig, WorkPoint
    from hfss_paper_results.sim import Coefficients, solve_slot_lp

    cfg = SimConfig(num_nodes=1, num_critical=1, protected_margin_frac=0.03)
    wp = WorkPoint(
        lambda_q=0.0,
        lambda_c=0.0,
        lambda_e=0.22,
        lambda_xi=0.36,
        B_EM_ref=0.55,
        P_H=1.0,
        lambda_s=20.0,
    )
    decision = Coefficients(
        labels=["implant30_stress"],
        r=np.array([0.8], dtype=float),
        g=np.array([0.10], dtype=float),
        eta=np.array([1.0], dtype=float),
        chi=np.array([0.10], dtype=float),
        p_rx_cap=np.array([1.0], dtype=float),
        B_EM=1.0,
    )
    e = np.array([cfg.E_min_critical + cfg.E_base], dtype=float)
    q = np.array([0.0], dtype=float)
    rho = np.array([0.0], dtype=float)
    critical = np.array([True])

    solution = solve_slot_lp("Proposed", cfg, wp, decision, e, q, rho, critical)

    safe_floor = cfg.E_min_critical + cfg.protected_margin_frac * (cfg.E_max - cfg.E_min_critical)
    post_energy = np.maximum(e - cfg.E_base, 0.0) + decision.eta * wp.P_H * solution.u_e - rho * solution.u_d
    assert solution.success
    assert post_energy[0] >= safe_floor - 1e-8
    assert solution.safe_slack[0] <= 1e-8


def test_slot_lp_solver_reports_safe_slack_when_margin_infeasible():
    from hfss_paper_results.config import SimConfig, WorkPoint
    from hfss_paper_results.sim import Coefficients, solve_slot_lp

    cfg = SimConfig(num_nodes=1, num_critical=1, protected_margin_frac=0.03)
    wp = WorkPoint(
        lambda_q=0.0,
        lambda_c=0.0,
        lambda_e=0.22,
        lambda_xi=0.36,
        B_EM_ref=0.55,
        P_H=1.0,
        lambda_s=20.0,
    )
    decision = Coefficients(
        labels=["implant30_stress"],
        r=np.array([0.8], dtype=float),
        g=np.array([0.10], dtype=float),
        eta=np.array([1.0], dtype=float),
        chi=np.array([0.10], dtype=float),
        p_rx_cap=np.array([1.0], dtype=float),
        B_EM=0.0,
    )
    e = np.array([cfg.E_min_critical + cfg.E_base], dtype=float)
    q = np.array([0.0], dtype=float)
    rho = np.array([0.0], dtype=float)
    critical = np.array([True])

    solution = solve_slot_lp("Proposed", cfg, wp, decision, e, q, rho, critical)

    safe_floor = cfg.E_min_critical + cfg.protected_margin_frac * (cfg.E_max - cfg.E_min_critical)
    post_energy = np.maximum(e - cfg.E_base, 0.0) + decision.eta * wp.P_H * solution.u_e - rho * solution.u_d
    assert solution.success
    assert post_energy[0] < safe_floor
    assert solution.safe_slack[0] >= safe_floor - post_energy[0] - 1e-8
