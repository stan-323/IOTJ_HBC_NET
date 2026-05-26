from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
HFSS_WORKBENCH = WORKSPACE_ROOT.parent / "_archive" / "2026-05-07_cleanup" / "hfss_workbench" / "hbc_hfss_calibration"
HFSS_OUTPUTS = HFSS_WORKBENCH / "outputs"
HFSS_REVIEW = HFSS_OUTPUTS / "hfss_review"
HFSS_FINAL = HFSS_OUTPUTS / "final"
FIELD_SAMPLES = HFSS_FINAL / "field_samples"

SCHEDULING_LABELS = [
    "surface_rest",
    "surface_moderate_loose",
    "surface_sweat",
    "implant10_rest",
    "implant10_stress",
    "implant30_rest",
    "implant30_stress",
]


def _read_field_sample(path: Path, sigma: float) -> np.ndarray:
    data = np.loadtxt(path, skiprows=1)
    ex_re, ex_im = data[:, 3], data[:, 4]
    ey_re, ey_im = data[:, 5], data[:, 6]
    ez_re, ez_im = data[:, 7], data[:, 8]
    e2 = ex_re**2 + ex_im**2 + ey_re**2 + ey_im**2 + ez_re**2 + ez_im**2
    return sigma * e2


def build_chi_sensitivity(out_dir: Path) -> pd.DataFrame:
    sampled = pd.read_csv(HFSS_FINAL / "sampled_field_proxy_final.csv")
    sampled = sampled[(sampled["cube_mm"].astype(float) == 5.0) & (sampled["label"].isin(SCHEDULING_LABELS))]
    by_label = {str(row["label"]): row for _, row in sampled.iterrows()}
    ref = by_label["surface_rest"]
    ref_q = _read_field_sample(FIELD_SAMPLES / Path(str(ref["sample_points_file"])).name, float(ref["sigma_S_per_m"]))
    ref_stats = {
        "p90": np.percentile(ref_q, 90),
        "p95": np.percentile(ref_q, 95),
        "p99": np.percentile(ref_q, 99),
        "mean": np.mean(ref_q),
    }
    rows: list[dict[str, object]] = []
    for label in SCHEDULING_LABELS:
        row = by_label[label]
        q = _read_field_sample(FIELD_SAMPLES / Path(str(row["sample_points_file"])).name, float(row["sigma_S_per_m"]))
        stats = {
            "p90": np.percentile(q, 90),
            "p95": np.percentile(q, 95),
            "p99": np.percentile(q, 99),
            "mean": np.mean(q),
        }
        values = {f"chi_{key}": float(stats[key] / max(ref_stats[key], 1e-12)) for key in stats}
        rows.append(
            {
                "label": label,
                **values,
                "relative_span_pct": 100.0 * (max(values.values()) - min(values.values())) / max(values["chi_p95"], 1e-12),
            }
        )
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "chi_sensitivity.csv", index=False)
    return frame


def _parse_touchstone_s21(path: Path) -> tuple[float, float]:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!") or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 9:
            freq_ghz = float(parts[0])
            s21_mag = float(parts[3])
            s21_db = 20.0 * math.log10(max(s21_mag, 1e-30))
            return freq_ghz * 1000.0, s21_db
    raise ValueError(f"No S21 row found in {path}")


def build_fd_sensitivity(out_dir: Path) -> pd.DataFrame:
    design_map = {
        "surface_rest": "review_surface_rest",
        "implant10_rest": "review_implant10_rest",
        "implant30_rest": "review_implant30_rest",
        "implant30_stress": "review_implant30_stress",
    }
    freqs = [(13.56, "SP13p56"), (20.0, "SP20p0"), (40.0, "SP40p0")]
    rows: list[dict[str, object]] = []
    for freq, sweep in freqs:
        ref_path = HFSS_REVIEW / f"{design_map['surface_rest']}_{sweep}.s2p"
        _, ref_db = _parse_touchstone_s21(ref_path)
        for label, design in design_map.items():
            _, s21_db = _parse_touchstone_s21(HFSS_REVIEW / f"{design}_{sweep}.s2p")
            rows.append(
                {
                    "label": label,
                    "freq_MHz": freq,
                    "s21_db": s21_db,
                    "r_norm": 10.0 ** ((s21_db - ref_db) / 10.0),
                }
            )
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "fd_sensitivity.csv", index=False)
    return frame


def copy_hfss_model_card_sources(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tissue_materials.csv", "hfss_setup_table.csv", "mesh_convergence.csv"]:
        src = HFSS_OUTPUTS / name
        if src.exists():
            pd.read_csv(src).to_csv(out_dir / name, index=False)


def write_measurement_anchor_readme(out_dir: Path) -> None:
    anchor_dir = out_dir / "measurement_anchor"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    (anchor_dir / "README.md").write_text(
        "\n".join(
            [
                "# Measurement Anchor Placeholder",
                "",
                "In-house HBC measurement data are intentionally not fabricated in this revision run.",
                "Place measured |S21| traces and matching HFSS traces here before enabling the manuscript measurement-validation figure.",
                "Expected files:",
                "- `measurement_s21.csv`: columns `freq_MHz`, `s21_db`.",
                "- `hfss_matched_s21.csv`: columns `freq_MHz`, `s21_db`.",
                "- `measurement_setup.md`: electrode placement, body/contact condition, instrumentation, calibration notes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_revision_assets(out_dir: Path) -> None:
    build_chi_sensitivity(out_dir)
    build_fd_sensitivity(out_dir)
    copy_hfss_model_card_sources(out_dir)
    write_measurement_anchor_readme(out_dir)
    status = HFSS_FINAL / "surface_sweat_recalibration_summary.md"
    if status.exists():
        (out_dir / "hfss_sweat_recalibration_status.md").write_text(status.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (out_dir / "hfss_sweat_recalibration_status.md").write_text(
            "\n".join(
                [
                    "# HFSS Sweat Recalibration Status",
                    "",
                    "Automatic sweat-state re-solving is not completed by this asset builder.",
                    "The current scheduling library still reports the archived HFSS value for `surface_sweat`.",
                    "Do not claim `g_norm <= 4.0` unless a fresh AEDT solve updates `torso_scheduler_library.csv`.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
