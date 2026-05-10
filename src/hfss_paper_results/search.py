from __future__ import annotations

from hfss_paper_results.config import WorkPoint


def coarse_workpoints() -> list[tuple[str, WorkPoint]]:
    anchors = [
        ("iter_01_A", WorkPoint(1.15, 1.58, 0.22, 0.36, 0.90, 1.16)),
        ("iter_01_B", WorkPoint(1.28, 1.38, 0.18, 0.32, 0.86, 1.14)),
        ("iter_02_A", WorkPoint(1.20, 1.72, 0.24, 0.42, 0.88, 1.18)),
        ("iter_02_B", WorkPoint(1.34, 1.48, 0.20, 0.36, 0.84, 1.12)),
        ("iter_03_A", WorkPoint(1.22, 1.82, 0.28, 0.48, 0.86, 1.20)),
        ("iter_03_B", WorkPoint(1.40, 1.56, 0.22, 0.42, 0.82, 1.16)),
        ("iter_04_A", WorkPoint(1.18, 1.68, 0.20, 0.40, 0.92, 1.22)),
        ("iter_04_B", WorkPoint(1.42, 1.62, 0.18, 0.44, 0.88, 1.18)),
        ("iter_05_A", WorkPoint(1.26, 1.78, 0.26, 0.46, 0.90, 1.20)),
        ("iter_05_B", WorkPoint(1.36, 1.52, 0.16, 0.38, 0.86, 1.16)),
    ]
    return anchors


def ralph_seed_count(index: int) -> int:
    return 5 if index <= 10 else 20


def ralph_workpoint(best: WorkPoint, index: int) -> WorkPoint:
    if index <= 10:
        offsets = [
            (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
            (0.03, 0.05, 0.00, 0.02, -0.01, 0.01),
            (-0.03, 0.05, 0.01, 0.03, 0.00, 0.02),
            (0.04, -0.04, -0.01, -0.02, 0.01, 0.00),
            (-0.02, -0.05, 0.02, 0.02, 0.02, -0.01),
            (0.00, 0.06, 0.02, 0.03, -0.02, 0.03),
            (0.04, -0.02, -0.02, 0.02, 0.02, 0.00),
            (-0.04, 0.03, 0.02, -0.02, 0.00, 0.02),
            (0.02, 0.06, 0.00, 0.00, -0.01, 0.00),
            (0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        ][index - 1]
    else:
        offsets = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.01, 0.02, 0.0, 0.01, 0.0, 0.0),
            (-0.01, -0.02, 0.0, -0.01, 0.0, 0.0),
            (0.0, 0.0, 0.01, 0.0, 0.0, 0.01),
            (0.0, 0.0, -0.01, 0.0, 0.0, -0.01),
        ][index - 11]
    return best.shifted(
        lambda_q=offsets[0],
        lambda_c=offsets[1],
        lambda_e=offsets[2],
        lambda_xi=offsets[3],
        B_EM_ref=offsets[4],
        P_H=offsets[5],
    )

