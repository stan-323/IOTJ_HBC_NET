from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from Experiment_results.baselines import RoundRobinBaseline
from Experiment_results.config import SimConfig, WorkPoint
from Experiment_results.data import HFSSLibrary, load_hfss_library
from Experiment_results.scenarios import (
    condition_name,
    condition_switch_labels,
    is_critical_node,
    node_type,
    stress_window_labels,
)


@dataclass
class Trace:
    seed: int
    is_critical: np.ndarray
    r_ref: np.ndarray
    eta_ref: np.ndarray
    rho: np.ndarray
    chi_ref: np.ndarray
    arrivals: np.ndarray
    initial_energy: np.ndarray
    initial_queue: np.ndarray


@dataclass
class Coefficients:
    labels: list[str]
    r: np.ndarray
    g: np.ndarray
    eta: np.ndarray
    chi: np.ndarray
    p_rx_cap: np.ndarray
    B_EM: float


@dataclass
class SlotLPSolution:
    u_d: np.ndarray
    u_e: np.ndarray
    mu: np.ndarray
    xi: np.ndarray
    safe_slack: np.ndarray
    success: bool
    status: str
    objective: float
    backend: str = "lp"
    solve_time_ms: float = 0.0


LP_METHODS = {"Proposed", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware"}


def generate_trace(cfg: SimConfig, seed: int, scenario: str) -> Trace:
    rng = np.random.default_rng(seed)
    n = cfg.num_nodes
    critical = np.array([is_critical_node(i, cfg.num_critical) for i in range(n)], dtype=bool)
    surface = ~critical

    r_ref = rng.uniform(0.78, 1.08, size=n)
    r_ref[surface] = rng.uniform(0.95, 1.30, size=surface.sum())
    eta_ref = rng.uniform(0.52, 0.70, size=n)
    eta_ref[critical] = rng.uniform(0.60, 0.82, size=critical.sum())
    rho = rng.uniform(0.090, 0.125, size=n)
    chi_ref = rng.uniform(0.94, 1.06, size=n)

    arrival_rate = rng.uniform(0.010, 0.025, size=n) * cfg.offered_load_scale
    arrival_rate[critical] = rng.uniform(0.015, 0.030, size=critical.sum()) * cfg.offered_load_scale * cfg.critical_load_scale
    arrivals = rng.poisson(arrival_rate, size=(cfg.T, n)).astype(float)
    bursts = rng.random((cfg.T, n)) < cfg.burst_probability
    arrivals += bursts * rng.uniform(cfg.burst_size_min, cfg.burst_size_max, size=(cfg.T, n))
    arrivals += rng.uniform(0.0, 0.12, size=(cfg.T, n)) * arrival_rate
    if scenario == "stress":
        start, end = cfg.stress_start, cfg.stress_end
        for node in cfg.stress_nodes:
            arrivals[start:end, node] *= cfg.stress_arrival_multiplier
            arrivals[start:end, node] += arrival_rate[node] * 0.30

    initial_energy = rng.uniform(0.52, 0.70, size=n)
    initial_energy[critical] = rng.uniform(0.48, 0.58, size=critical.sum())
    initial_queue = rng.uniform(0.0, cfg.initial_queue_scale, size=n)
    return Trace(seed, critical, r_ref, eta_ref, rho, chi_ref, arrivals, initial_energy, initial_queue)


def coefficients_for_labels(
    library: HFSSLibrary,
    cfg: SimConfig,
    wp: WorkPoint,
    trace: Trace,
    labels: list[str],
    *,
    budget_multiplier: float = 1.0,
) -> Coefficients:
    rows = [library.coeff(label) for label in labels]
    g = np.asarray([row.g_norm for row in rows], dtype=float)
    r_norm = np.asarray([row.r_norm for row in rows], dtype=float)
    chi_norm = np.asarray([row.chi_norm for row in rows], dtype=float)
    p_cap_norm = np.asarray([row.p_rx_cap_norm for row in rows], dtype=float)
    return Coefficients(
        labels=labels,
        r=trace.r_ref * r_norm,
        g=g,
        eta=trace.eta_ref * g,
        chi=trace.chi_ref * chi_norm,
        p_rx_cap=np.full(cfg.num_nodes, cfg.P_rx_ref) * p_cap_norm,
        B_EM=wp.B_EM_ref * budget_multiplier,
    )


def _label_func(cfg: SimConfig, scenario: str) -> Callable[[int], list[str]]:
    if scenario == "stress":
        return lambda t: stress_window_labels(
            t,
            cfg.num_nodes,
            affected_nodes=cfg.stress_nodes,
            start=cfg.stress_start,
            end=cfg.stress_end,
            num_critical=cfg.num_critical,
        )
    return lambda t: condition_switch_labels(t, cfg.num_nodes, cfg.num_critical)


def _energy_target_util(method: str, wp: WorkPoint, e: np.ndarray, critical: np.ndarray, in_stress: bool) -> float:
    crit_min = float(e[critical].min()) if critical.any() else float(e.min())
    low = np.clip((0.52 - crit_min) / 0.17, 0.0, 1.0)
    base = {
        "Proposed": 0.58,
        "ADT-MAC": 0.42,
        "Lyap.-DPP": 0.80,
        "w/o crit.-urg.": 0.54,
        "w/o implant-aware": 0.57,
        "Oracle": 0.90,
    }.get(method, 0.58)
    if method == "w/o EM-bud.":
        return 1.35
    stress_bonus = 0.06 if in_stress else 0.0
    return float(min(base + 0.20 * low + stress_bonus + 0.05 * max(wp.lambda_c - 1.55, 0.0), 0.90))


def _allocate_energy(
    method: str,
    cfg: SimConfig,
    wp: WorkPoint,
    decision: Coefficients,
    e: np.ndarray,
    q: np.ndarray,
    critical: np.ndarray,
    in_stress: bool,
    *,
    lyap_v_scale: float = 1.0,
) -> np.ndarray:
    e_hat = np.maximum(e - cfg.E_base, 0.0)
    urgency = np.clip((cfg.E_tar_critical - e_hat) / (cfg.E_tar_critical - cfg.E_min_critical + 1e-9), 0.0, 1.0)
    short = np.clip((cfg.E_min_critical - e_hat) / cfg.E_min_critical, 0.0, 1.0)
    queue = q / (cfg.q_ref + q + 1e-9)

    priority = 0.05 + 0.08 * queue
    if method == "Proposed":
        priority += critical * (2.15 * wp.lambda_c * urgency + 2.05 * wp.lambda_xi * short)
    elif method == "Lyap.-DPP":
        energy_bias = 1.0 / np.sqrt(max(lyap_v_scale, 1e-9))
        priority += critical * ((3.20 * urgency + 2.10 * short + 0.15) * energy_bias)
        priority += (~critical) * 0.02
    elif method == "ADT-MAC":
        priority += critical * (0.42 * urgency + 0.15 * short)
        priority += 0.20 * queue
    elif method == "w/o EM-bud.":
        priority += critical * (2.30 * wp.lambda_c * urgency + 2.20 * wp.lambda_xi * short)
    elif method == "w/o crit.-urg.":
        priority += 0.55 * urgency + 0.20 * short
        priority += 0.10 * queue
    elif method == "w/o implant-aware":
        priority += critical * (1.25 * wp.lambda_c * urgency + 0.50 * wp.lambda_xi * short)
    elif method == "Oracle":
        priority += critical * (5.00 * urgency + 8.00 * short + 0.50)
    priority *= np.maximum(decision.eta, 1e-9)

    enforce_em = method != "w/o EM-bud."
    total_left = 1.0
    em_left = decision.B_EM
    target_em = decision.B_EM * _energy_target_util(method, wp, e, critical, in_stress)
    if not enforce_em:
        target_em = decision.B_EM * _energy_target_util(method, wp, e, critical, in_stress)

    p_rx_denom = np.maximum(decision.g * wp.P_H, 1e-12)
    caps = np.clip(decision.p_rx_cap / p_rx_denom, 0.0, 1.0)
    order = np.argsort(-priority)
    u = np.zeros_like(priority, dtype=float)
    positive = max(float(priority.max()), 1e-9)
    for node in order:
        if priority[node] <= 0.0 or total_left <= 1e-12:
            continue
        cap = min(float(caps[node]), total_left)
        if enforce_em:
            cap = min(cap, em_left / max(decision.chi[node], 1e-12), target_em / max(decision.chi[node], 1e-12))
        else:
            cap = min(cap, target_em / max(decision.chi[node], 1e-12))
        if cap <= 0.0:
            continue
        base = 0.035 if method != "w/o EM-bud." else 0.12
        span = 0.34 if method in {"Proposed", "Lyap.-DPP", "Oracle"} else 0.24
        if method == "Oracle" and critical[node]:
            span = 0.62
        amount = min(cap, base + span * float(priority[node] / positive))
        u[node] = max(amount, 0.0)
        total_left -= u[node]
        spent = decision.chi[node] * u[node]
        if enforce_em:
            em_left -= spent
        target_em -= spent
        if target_em <= 1e-12:
            break
    return np.clip(u, 0.0, 1.0)


def _allocate_data(
    method: str,
    cfg: SimConfig,
    wp: WorkPoint,
    decision: Coefficients,
    e: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    critical: np.ndarray,
    sum_u_e: float,
    *,
    lyap_v_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = min(1.0, max(0.0, (1.0 - cfg.alpha_E * sum_u_e) / cfg.alpha_D))
    if method == "Lyap.-DPP":
        remaining *= float(np.clip(0.58 * np.sqrt(max(lyap_v_scale, 1e-9)), 0.35, 0.95))
    saliency = q / (cfg.q_ref + q + 1e-9)
    weights = wp.lambda_q * saliency * np.maximum(decision.r, 1e-9)
    weights *= np.where(critical, cfg.omega_critical, cfg.omega_surface)
    if method == "ADT-MAC":
        weights += 0.28 * saliency + np.where(critical, 0.04, 0.08)
    elif method == "Lyap.-DPP":
        service_bias = float(np.clip(np.sqrt(max(lyap_v_scale, 1e-9)), 0.50, 1.60))
        weights *= np.where(critical, 0.55, 0.62 * service_bias)
    elif method == "w/o crit.-urg.":
        margin = np.clip((e - cfg.E_min_critical) / max(cfg.E_tar_critical - cfg.E_min_critical, 1e-9), 0.0, 1.0)
        weights *= np.where(critical, 0.35 + 0.65 * margin, 0.98)
    elif method == "Oracle":
        weights *= np.where(critical, 0.08, 0.35)

    e_hat = np.maximum(e - cfg.E_base, 0.0)
    margin = np.clip((e_hat - cfg.E_min_critical) / (cfg.E_tar_critical - cfg.E_min_critical + 1e-9), 0.0, 1.0)
    if method in {"Proposed", "Lyap.-DPP", "Oracle"}:
        guard = np.where(critical, 0.12 + 0.88 * margin, 1.0)
        weights *= guard
    elif method == "w/o implant-aware":
        weights *= np.where(critical, 0.45 + 0.55 * margin, 1.0)

    order = np.argsort(-weights)
    u_d = np.zeros_like(q, dtype=float)
    for node in order:
        if remaining <= 1e-12:
            break
        if weights[node] <= 0.0 or q[node] <= 1e-12:
            continue
        energy_cap = max(0.0, e_hat[node] / max(rho[node], 1e-12))
        need = min(remaining, q[node] / max(decision.r[node], 1e-12), energy_cap, 1.0)
        if method == "ADT-MAC":
            need = min(remaining, max(need, min(0.05, remaining)), energy_cap, 1.0)
        u_d[node] = max(need, 0.0)
        remaining -= u_d[node]
    mu = np.minimum(q, decision.r * u_d)
    return u_d, mu


def solve_slot_lp(
    method: str,
    cfg: SimConfig,
    wp: WorkPoint,
    decision: Coefficients,
    e: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    critical: np.ndarray,
    *,
    lp_tolerance: float | None = None,
) -> SlotLPSolution:
    n = len(q)
    critical_idx = np.flatnonzero(critical)
    c_count = len(critical_idx)
    u_d0 = 0
    u_e0 = n
    mu0 = 2 * n
    xi0 = 3 * n
    safe0 = xi0 + c_count
    total_vars = 3 * n + 2 * c_count

    objective = np.zeros(total_vars, dtype=float)
    phi = q / (cfg.q_ref + q + 1e-9)
    omega = np.where(critical, cfg.omega_critical, cfg.omega_surface)
    objective[mu0 : mu0 + n] = -wp.lambda_q * omega * phi

    e_hat = np.maximum(e - cfg.E_base, 0.0)
    psi = np.zeros(n, dtype=float)
    if method != "w/o crit.-urg." and c_count:
        denom = cfg.E_tar_critical - cfg.E_min_critical + 1e-9
        psi[critical] = np.clip((cfg.E_tar_critical - e_hat[critical]) / denom, 0.0, 1.0)
    objective[u_e0 : u_e0 + n] = wp.lambda_e
    objective[u_e0 : u_e0 + n] -= wp.lambda_c * psi * decision.eta * wp.P_H
    for local_idx, _node in enumerate(critical_idx):
        objective[xi0 + local_idx] = wp.lambda_xi / cfg.E_min_critical
        objective[safe0 + local_idx] = wp.lambda_s / max(cfg.E_max - cfg.E_min_critical, 1e-9)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    row = np.zeros(total_vars, dtype=float)
    row[u_d0 : u_d0 + n] = 1.0
    a_ub.append(row)
    b_ub.append(1.0)

    row = np.zeros(total_vars, dtype=float)
    row[u_e0 : u_e0 + n] = 1.0
    a_ub.append(row)
    b_ub.append(1.0)

    for node in range(n):
        row = np.zeros(total_vars, dtype=float)
        row[mu0 + node] = 1.0
        row[u_d0 + node] = -float(decision.r[node])
        a_ub.append(row)
        b_ub.append(0.0)

        harvest_coeff = float(decision.eta[node] * wp.P_H)

        row = np.zeros(total_vars, dtype=float)
        row[u_e0 + node] = harvest_coeff
        row[u_d0 + node] = -float(rho[node])
        a_ub.append(row)
        b_ub.append(float(cfg.E_max - e_hat[node]))

        row = np.zeros(total_vars, dtype=float)
        row[u_e0 + node] = -harvest_coeff
        row[u_d0 + node] = float(rho[node])
        a_ub.append(row)
        b_ub.append(float(e_hat[node]))

        row = np.zeros(total_vars, dtype=float)
        row[u_e0 + node] = float(decision.g[node] * wp.P_H)
        a_ub.append(row)
        b_ub.append(float(decision.p_rx_cap[node]))

    critical_to_local = {node: local for local, node in enumerate(critical_idx)}
    safe_floor = cfg.E_min_critical + cfg.protected_margin_frac * (cfg.E_max - cfg.E_min_critical)
    enforce_safe_margin = method != "w/o crit.-urg." and cfg.protected_margin_frac > 0.0
    for node in critical_idx:
        local = critical_to_local[int(node)]
        harvest_coeff = float(decision.eta[node] * wp.P_H)
        row = np.zeros(total_vars, dtype=float)
        row[u_e0 + node] = -harvest_coeff
        row[u_d0 + node] = float(rho[node])
        row[xi0 + local] = -1.0
        a_ub.append(row)
        b_ub.append(float(e_hat[node] - cfg.E_min_critical))

        if enforce_safe_margin:
            row = np.zeros(total_vars, dtype=float)
            row[u_e0 + node] = -harvest_coeff
            row[u_d0 + node] = float(rho[node])
            row[safe0 + local] = -1.0
            a_ub.append(row)
            b_ub.append(float(e_hat[node] - safe_floor))

    if method != "w/o EM-bud.":
        row = np.zeros(total_vars, dtype=float)
        row[u_e0 : u_e0 + n] = decision.chi
        a_ub.append(row)
        b_ub.append(float(decision.B_EM))

    bounds: list[tuple[float, float]] = []
    bounds.extend((0.0, 1.0) for _ in range(n))
    bounds.extend((0.0, 1.0) for _ in range(n))
    bounds.extend((0.0, float(max(q[node], 0.0))) for node in range(n))
    bounds.extend((0.0, float(cfg.E_min_critical)) for _ in critical_idx)
    bounds.extend((0.0, float(cfg.E_max - cfg.E_min_critical)) for _ in critical_idx)

    options = None
    if lp_tolerance is not None:
        tol = float(max(lp_tolerance, 1e-10))
        options = {
            "primal_feasibility_tolerance": tol,
            "dual_feasibility_tolerance": tol,
            "ipm_optimality_tolerance": tol,
        }
    started = perf_counter()
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
        options=options,
    )
    solve_time_ms = 1000.0 * (perf_counter() - started)
    if not result.success:
        return SlotLPSolution(
            u_d=np.zeros(n, dtype=float),
            u_e=np.zeros(n, dtype=float),
            mu=np.zeros(n, dtype=float),
            xi=np.zeros(n, dtype=float),
            safe_slack=np.zeros(n, dtype=float),
            success=False,
            status=str(result.message),
            objective=float("nan"),
            solve_time_ms=solve_time_ms,
        )

    x = np.asarray(result.x, dtype=float)
    xi_full = np.zeros(n, dtype=float)
    safe_slack_full = np.zeros(n, dtype=float)
    for local, node in enumerate(critical_idx):
        xi_full[int(node)] = x[xi0 + local]
        safe_slack_full[int(node)] = x[safe0 + local]
    return SlotLPSolution(
        u_d=np.clip(x[u_d0 : u_d0 + n], 0.0, 1.0),
        u_e=np.clip(x[u_e0 : u_e0 + n], 0.0, 1.0),
        mu=np.clip(x[mu0 : mu0 + n], 0.0, q),
        xi=xi_full,
        safe_slack=safe_slack_full,
        success=True,
        status=str(result.message),
        objective=float(-result.fun),
        solve_time_ms=solve_time_ms,
    )


def run_method(
    cfg: SimConfig,
    wp: WorkPoint,
    method: str,
    seed: int,
    scenario: str,
    library: HFSSLibrary | None = None,
    *,
    decision_library: HFSSLibrary | None = None,
    budget_multiplier: float = 1.0,
    power_multiplier: float = 1.0,
    lyap_v_scale: float = 1.0,
    label_estimator: Callable[[int, list[str], Trace, HFSSLibrary], list[str]] | None = None,
    label_func: Callable[..., list[str]] | None = None,
    lp_tolerance: float | None = None,
) -> pd.DataFrame:
    lib = library or load_hfss_library()
    decision_lib = decision_library or lib
    local_wp = WorkPoint(
        wp.lambda_q,
        wp.lambda_c,
        wp.lambda_e,
        wp.lambda_xi,
        wp.B_EM_ref * budget_multiplier,
        wp.P_H * power_multiplier,
        wp.lambda_s,
    )
    trace = generate_trace(cfg, seed, scenario)
    labels_for_t = label_func or _label_func(cfg, scenario)
    q = trace.initial_queue.copy()
    e = trace.initial_energy.copy()
    rows: list[dict[str, object]] = []
    round_robin = RoundRobinBaseline(cfg.num_nodes) if method == "Round-Robin" else None
    for t in range(cfg.T):
        if label_func is None:
            true_labels = labels_for_t(t)
        else:
            try:
                true_labels = label_func(t, trace)
            except TypeError:
                true_labels = label_func(t)
        true_coeff = coefficients_for_labels(lib, cfg, local_wp, trace, true_labels)
        decision_labels = label_estimator(t, true_labels, trace, decision_lib) if label_estimator else true_labels
        decision_coeff = coefficients_for_labels(decision_lib, cfg, local_wp, trace, decision_labels)
        decision_critical = trace.is_critical.copy()
        in_stress = scenario == "stress" and cfg.stress_start <= t < cfg.stress_end

        if method in LP_METHODS:
            lp_solution = solve_slot_lp(method, cfg, local_wp, decision_coeff, e, q, trace.rho, decision_critical, lp_tolerance=lp_tolerance)
            u_e = lp_solution.u_e
            u_d = lp_solution.u_d
            mu_decision = lp_solution.mu
            allocation_backend = lp_solution.backend
            lp_failed = not lp_solution.success
            solver_status = lp_solution.status
            solve_time_ms = lp_solution.solve_time_ms
        elif round_robin is not None:
            u_e, u_d, mu_decision = round_robin.step(t, q, decision_coeff.r)
            allocation_backend = "round_robin"
            lp_failed = False
            solver_status = "not_applicable"
            solve_time_ms = 0.0
        else:
            u_e = _allocate_energy(method, cfg, local_wp, decision_coeff, e, q, decision_critical, in_stress, lyap_v_scale=lyap_v_scale)
            u_d, mu_decision = _allocate_data(method, cfg, local_wp, decision_coeff, e, q, trace.rho, decision_critical, float(u_e.sum()), lyap_v_scale=lyap_v_scale)
            allocation_backend = "heuristic"
            lp_failed = False
            solver_status = "not_applicable"
            solve_time_ms = 0.0

        e_hat = np.maximum(e - cfg.E_base, 0.0)
        p_rx = true_coeff.g * local_wp.P_H * u_e
        harvested = true_coeff.eta * local_wp.P_H * u_e
        tx_cost = trace.rho * u_d
        post_energy = e_hat + harvested - tx_cost
        xi = np.where(trace.is_critical, np.clip(cfg.E_min_critical - post_energy, 0.0, cfg.E_min_critical), 0.0)
        safe_floor = cfg.E_min_critical + cfg.protected_margin_frac * (cfg.E_max - cfg.E_min_critical)
        safe_slack = np.where(
            trace.is_critical,
            np.clip(safe_floor - post_energy, 0.0, cfg.E_max - cfg.E_min_critical),
            0.0,
        )
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
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "t": t,
                    "regime": "stress_window" if in_stress else condition_name(t),
                    "node": node,
                    "node_type": node_type(node, cfg.num_critical),
                    "is_critical": is_critical,
                    "is_stressed": bool(in_stress and node in cfg.stress_nodes),
                    "label_true": true_labels[node],
                    "label_decision": decision_labels[node],
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
                    "safe_floor": float(safe_floor if is_critical else 0.0),
                    "safe_slack": float(safe_slack[node]),
                    "safe_violation": bool(safe_slack[node] > 1e-8),
                    "served": float(mu[node]),
                    "arrived": float(arrived[node]),
                    "h": float(harvested[node]),
                    "c_tx": float(tx_cost[node]),
                    "r": float(true_coeff.r[node]),
                    "eta": float(true_coeff.eta[node]),
                    "g": float(true_coeff.g[node]),
                    "chi": float(true_coeff.chi[node]),
                    "decision_r": float(decision_coeff.r[node]),
                    "decision_eta": float(decision_coeff.eta[node]),
                    "decision_g": float(decision_coeff.g[node]),
                    "decision_chi": float(decision_coeff.chi[node]),
                    "B_EM": float(true_coeff.B_EM),
                    "P_rx": float(p_rx[node]),
                    "P_rx_max": float(true_coeff.p_rx_cap[node]),
                    "rx_cap_ratio": float(rx_ratio),
                    "rx_cap_violation": bool(rx_ratio > 1.0 + cfg.rx_cap_tolerance),
                    "em_load": em_load,
                    "em_utilization": em_util,
                    "em_violation": bool(em_util > 1.0 + cfg.em_violation_tolerance),
                    "shared_frontend": shared,
                    "allocation_backend": allocation_backend,
                    "solver_status": solver_status,
                    "solve_time_ms": solve_time_ms,
                    "lp_failed": lp_failed,
                }
            )
        q = q_next
        e = e_next
    return pd.DataFrame.from_records(rows)


def run_suite(
    cfg: SimConfig,
    wp: WorkPoint,
    methods: list[str],
    seeds: list[int],
    scenario: str,
    library: HFSSLibrary | None = None,
    *,
    decision_library: HFSSLibrary | None = None,
    per_method_libraries: dict[str, HFSSLibrary] | None = None,
    per_method_decision_libraries: dict[str, HFSSLibrary] | None = None,
    budget_multiplier: float = 1.0,
    power_multiplier: float = 1.0,
    lyap_v_scale: float = 1.0,
    label_estimator: Callable[[int, list[str], Trace, HFSSLibrary], list[str]] | None = None,
    label_func: Callable[..., list[str]] | None = None,
    lp_tolerance: float | None = None,
) -> pd.DataFrame:
    lib = library or load_hfss_library()
    frames = [
        run_method(
            cfg,
            wp,
            method,
            seed,
            scenario,
            per_method_libraries.get(method, lib) if per_method_libraries else lib,
            decision_library=(per_method_decision_libraries.get(method) if per_method_decision_libraries else decision_library),
            budget_multiplier=budget_multiplier,
            power_multiplier=power_multiplier,
            lyap_v_scale=lyap_v_scale,
            label_estimator=label_estimator,
            label_func=label_func,
            lp_tolerance=lp_tolerance,
        )
        for seed in seeds
        for method in methods
    ]
    return pd.concat(frames, ignore_index=True)


def run_phase_cell_metrics(
    cfg: SimConfig,
    wp: WorkPoint,
    seeds: list[int],
    library: HFSSLibrary | None = None,
    *,
    budget_multiplier: float = 1.0,
    power_multiplier: float = 1.0,
) -> dict[str, float]:
    lib = library or load_hfss_library()
    local_wp = WorkPoint(
        wp.lambda_q,
        wp.lambda_c,
        wp.lambda_e,
        wp.lambda_xi,
        wp.B_EM_ref * budget_multiplier,
        wp.P_H * power_multiplier,
        wp.lambda_s,
    )
    severity_values: list[float] = []
    shortage_values: list[float] = []
    served_values: list[float] = []

    for seed in seeds:
        trace = generate_trace(cfg, seed, "stress")
        rest_labels = stress_window_labels(
            0,
            cfg.num_nodes,
            affected_nodes=cfg.stress_nodes,
            start=cfg.stress_start,
            end=cfg.stress_end,
            num_critical=cfg.num_critical,
        )
        stress_labels = stress_window_labels(
            cfg.stress_start,
            cfg.num_nodes,
            affected_nodes=cfg.stress_nodes,
            start=cfg.stress_start,
            end=cfg.stress_end,
            num_critical=cfg.num_critical,
        )
        rest_coeff = coefficients_for_labels(lib, cfg, local_wp, trace, rest_labels)
        stress_coeff = coefficients_for_labels(lib, cfg, local_wp, trace, stress_labels)

        q = trace.initial_queue.copy()
        e = trace.initial_energy.copy()
        severity_sum = 0.0
        shortage_count = 0
        stress_count = 0
        served_sum = 0.0
        arrived_sum = 0.0

        for t in range(cfg.T):
            in_stress = cfg.stress_start <= t < cfg.stress_end
            true_coeff = stress_coeff if in_stress else rest_coeff
            lp_solution = solve_slot_lp("Proposed", cfg, local_wp, true_coeff, e, q, trace.rho, trace.is_critical)
            u_e = lp_solution.u_e
            u_d = lp_solution.u_d

            e_hat = np.maximum(e - cfg.E_base, 0.0)
            harvested = true_coeff.eta * local_wp.P_H * u_e
            tx_cost = trace.rho * u_d
            post_energy = e_hat + harvested - tx_cost
            xi = np.where(trace.is_critical, np.clip(cfg.E_min_critical - post_energy, 0.0, cfg.E_min_critical), 0.0)
            e_next = np.clip(post_energy, 0.0, cfg.E_max)
            mu = np.minimum(q, true_coeff.r * u_d)
            arrived = trace.arrivals[t]
            q = np.maximum(q - mu, 0.0) + arrived
            e = e_next
            served_sum += float(mu.sum())
            arrived_sum += float(arrived.sum())

            if in_stress:
                for node in cfg.stress_nodes:
                    if trace.is_critical[node]:
                        stress_count += 1
                        severity_sum += float(xi[node] / cfg.E_min_critical)
                        if e_next[node] < cfg.E_min_critical - 1e-12 or xi[node] > 1e-8:
                            shortage_count += 1

        denom = max(stress_count, 1)
        severity_values.append(severity_sum / denom)
        shortage_values.append(shortage_count / denom)
        served_values.append(served_sum / (arrived_sum + 1e-9))

    return {
        "stress_shortage_severity": float(np.mean(severity_values)) if severity_values else 0.0,
        "stress_implant_shortage_rate": float(np.mean(shortage_values)) if shortage_values else 0.0,
        "served_workload_ratio": float(np.mean(served_values)) if served_values else 0.0,
    }

