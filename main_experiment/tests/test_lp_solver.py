from __future__ import annotations

import numpy as np


def test_slot_lp_solver_satisfies_linear_constraints():
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.sim import Coefficients, solve_slot_lp

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


def test_run_method_records_lp_backend_for_proposed():
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.data import load_hfss_library
    from Experiment_results.sim import run_method

    cfg = SimConfig(T=4, stress_start=1, stress_end=3, post_stress_window=1, recovery_guard_slots=1)
    raw = run_method(cfg, WorkPoint(), "Proposed", 0, "stress", load_hfss_library())

    assert set(raw["allocation_backend"]) == {"lp"}
    assert not raw["lp_failed"].any()


def test_slot_lp_solver_respects_protected_margin_when_feasible():
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.sim import Coefficients, solve_slot_lp

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
    from Experiment_results.config import SimConfig, WorkPoint
    from Experiment_results.sim import Coefficients, solve_slot_lp

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

