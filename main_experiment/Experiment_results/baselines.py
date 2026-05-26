from __future__ import annotations

import numpy as np


class RoundRobinBaseline:
    """Naive lower-bound baseline that ignores state labels and field budgets."""

    def __init__(self, num_nodes: int, *, energy_alloc: float = 1.0) -> None:
        self.num_nodes = int(num_nodes)
        self.energy_alloc = float(energy_alloc)

    def step(self, slot: int, q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node = int(slot % self.num_nodes)
        u_e = np.zeros(self.num_nodes, dtype=float)
        u_d = np.zeros(self.num_nodes, dtype=float)
        u_e[node] = min(max(self.energy_alloc, 0.0), 1.0)
        u_d[node] = min(1.0, float(q[node] / max(r[node], 1e-12))) if q[node] > 0 else 0.0
        mu = np.minimum(q, r * u_d)
        return u_e, u_d, mu
