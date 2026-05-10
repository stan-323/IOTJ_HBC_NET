from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class WorkPoint:
    lambda_q: float = 1.25
    lambda_c: float = 1.65
    lambda_e: float = 0.22
    lambda_xi: float = 0.40
    B_EM_ref: float = 0.86
    P_H: float = 1.15
    lambda_s: float = 20.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def shifted(
        self,
        *,
        lambda_q: float = 0.0,
        lambda_c: float = 0.0,
        lambda_e: float = 0.0,
        lambda_xi: float = 0.0,
        B_EM_ref: float = 0.0,
        P_H: float = 0.0,
        lambda_s: float = 0.0,
    ) -> "WorkPoint":
        return replace(
            self,
            lambda_q=round(min(max(self.lambda_q + lambda_q, 1.05), 1.45), 4),
            lambda_c=round(min(max(self.lambda_c + lambda_c, 1.30), 1.85), 4),
            lambda_e=round(min(max(self.lambda_e + lambda_e, 0.14), 0.32), 4),
            lambda_xi=round(min(max(self.lambda_xi + lambda_xi, 0.28), 0.50), 4),
            B_EM_ref=round(min(max(self.B_EM_ref + B_EM_ref, 0.78), 0.94), 4),
            P_H=round(min(max(self.P_H + P_H, 1.06), 1.24), 4),
            lambda_s=round(min(max(self.lambda_s + lambda_s, 2.0), 80.0), 4),
        )


@dataclass(frozen=True)
class SimConfig:
    T: int = 1200
    num_nodes: int = 12
    num_critical: int = 3
    alpha_D: float = 0.45
    alpha_E: float = 0.55
    E_max: float = 1.0
    E_min_critical: float = 0.35
    E_tar_critical: float = 0.56
    protected_margin_frac: float = 0.03
    E_base: float = 0.012
    P_rx_ref: float = 0.80
    q_ref: float = 8.0
    omega_critical: float = 1.20
    omega_surface: float = 1.00
    stress_start: int = 500
    stress_end: int = 650
    stress_nodes: tuple[int, ...] = (1,)
    stress_arrival_multiplier: float = 1.45
    offered_load_scale: float = 4.25
    critical_load_scale: float = 1.28
    burst_probability: float = 0.06
    burst_size_min: float = 0.45
    burst_size_max: float = 1.55
    initial_queue_scale: float = 0.30
    em_violation_tolerance: float = 1e-6
    rx_cap_tolerance: float = 1e-8
    recovery_guard_slots: int = 50
    post_stress_window: int = 150

    @property
    def primary_stress_node(self) -> int:
        return int(self.stress_nodes[0])


MAIN_METHODS = ["Proposed", "ADT-MAC", "Lyap.-DPP"]
ABLATION_METHODS = ["Proposed", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware"]
STRESS_METHODS = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o crit.-urg."]
ALL_METHODS = ["Proposed", "ADT-MAC", "Lyap.-DPP", "w/o EM-bud.", "w/o crit.-urg.", "w/o implant-aware"]

METHOD_COLORS = {
    "Proposed": "#1F4E79",
    "ADT-MAC": "#B03A2E",
    "Lyap.-DPP": "#1E7A44",
    "w/o EM-bud.": "#6FA3D0",
    "w/o crit.-urg.": "#A9C2D8",
    "w/o implant-aware": "#7F7F7F",
    "Oracle": "#333333",
}

METHOD_LINESTYLES = {
    "Proposed": "-",
    "ADT-MAC": "--",
    "Lyap.-DPP": "-.",
    "w/o EM-bud.": ":",
    "w/o crit.-urg.": (0, (5, 2)),
    "w/o implant-aware": (0, (3, 1, 1, 1)),
    "Oracle": "-",
}
