from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(os.environ.get("TORSO_HFSS_ROOT", Path(__file__).resolve().parents[1]))
OUT_DIR = ROOT / "outputs"

ENERGY_FREQ_MHZ = 13.56
DATA_FREQ_MHZ = 40.0
IMPLANT_SCHED_ENERGY_SCALE = 2.0
IMPLANT_SCHED_CHI_NORM = 1.15

CANONICAL_CASES = [
    {
        "case": "t_surf_rest",
        "label": "surface_rest",
        "type": "surface",
        "depth_mm": 0.0,
        "condition": "rest",
        "sched_role": "baseline",
    },
    {
        "case": "t_surf_sweat",
        "label": "surface_sweat",
        "type": "surface",
        "depth_mm": 0.0,
        "condition": "sweat",
        "sched_role": "main_surface_wet",
    },
    {
        "case": "t_surf_loose",
        "label": "surface_moderate_loose",
        "type": "surface",
        "depth_mm": 0.0,
        "condition": "moderate_loose",
        "sched_role": "main_surface_degraded",
    },
    {
        "case": "t_surf_fail",
        "label": "surface_contact_failure",
        "type": "surface",
        "depth_mm": 0.0,
        "condition": "contact_failure",
        "sched_role": "legacy_surface_failure_compat",
        "sched_include_main": False,
    },
    {
        "case": "t_imp10_rest",
        "label": "implant10_rest",
        "type": "implant",
        "depth_mm": 10.0,
        "condition": "rest",
        "sched_role": "main_implant",
    },
    {
        "case": "t_imp10_stress",
        "label": "implant10_stress",
        "type": "implant",
        "depth_mm": 10.0,
        "condition": "stress",
        "sched_role": "main_implant_stress",
    },
    {
        "case": "t_imp30_rest",
        "label": "implant30_rest",
        "type": "implant",
        "depth_mm": 30.0,
        "condition": "rest",
        "sched_role": "main_implant",
    },
    {
        "case": "t_imp30_stress",
        "label": "implant30_stress",
        "type": "implant",
        "depth_mm": 30.0,
        "condition": "stress",
        "sched_role": "main_implant_stress",
    },
]

FREQUENCY_SWEEP_CASES = {"t_surf_rest", "t_imp10_rest", "t_imp30_stress"}

ANCHOR_CASES = [
    ("surface_x050", "t_scan_x050"),
    ("surface_x150", "t_scan_x150"),
    ("implant_d10", "t_depth_d010"),
    ("implant_d30", "t_depth_d030"),
    ("implant_d50", "t_depth_d050"),
]

FIELD_PROXY_NOTE = (
    "sampled E-field proxy in 5 mm receiver cube; not SAR compliance "
    "and not a 1g/10g SAR result"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required raw torso output is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def freq_value(row: dict[str, Any]) -> float:
    return round(as_float(row, "freq_MHz"), 10)


def case_value(row: dict[str, Any]) -> str:
    return str(row.get("case") or row.get("label") or "")


def rx_type_value(row: dict[str, Any]) -> str:
    return str(row.get("type") or row.get("rx_type") or "")


def depth_value(row: dict[str, Any]) -> float:
    return as_float(row, "depth_mm", as_float(row, "rx_depth_mm", 0.0))


def s21_mag2(row: dict[str, Any]) -> float:
    value = as_float(row, "S21_mag2")
    if math.isfinite(value):
        return value
    return 10.0 ** (as_float(row, "S21_dB") / 10.0)


def path_loss_db(row: dict[str, Any]) -> float:
    value = as_float(row, "path_loss_dB")
    if math.isfinite(value):
        return value
    return -as_float(row, "S21_dB")


def nearest_freq(rows: list[dict[str, Any]], case: str, freq_mhz: float) -> dict[str, Any]:
    candidates = [row for row in rows if case_value(row) == case]
    if not candidates:
        raise RuntimeError(f"No s-parameter rows for case {case}")
    return min(candidates, key=lambda row: abs(freq_value(row) - freq_mhz))


def field_by_case(field_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case_value(row): row for row in field_rows}


def q_value(field: dict[str, Any] | None, key: str) -> float:
    if not field:
        return float("nan")
    return as_float(field, key)


def normalized(value: float, reference: float) -> float | str:
    if not math.isfinite(value) or not math.isfinite(reference) or reference <= 0.0:
        return ""
    return value / reference


def scheduling_chi_norm(q_p95_norm: float | str, kind: str) -> float | str:
    if not isinstance(q_p95_norm, float) or not math.isfinite(q_p95_norm) or q_p95_norm <= 0.0:
        return ""
    if kind == "implant":
        return IMPLANT_SCHED_CHI_NORM
    return min(math.sqrt(q_p95_norm), 4.0)


def scheduler_energy_scale(kind: str) -> float:
    if kind == "implant":
        return IMPLANT_SCHED_ENERGY_SCALE
    return 1.0


def scheduler_calibration_note(kind: str) -> str:
    if kind == "implant":
        return (
            f"implant energy scaled by {IMPLANT_SCHED_ENERGY_SCALE:g} for LP scheduling units; "
            f"raw S21-normalized gain is retained in g_norm_raw_s21; chi fixed at "
            f"{IMPLANT_SCHED_CHI_NORM:g} because sampled 5 mm Q is a near-electrode load proxy, not SAR"
        )
    return "surface scheduler columns use raw normalized HFSS S21 and sqrt sampled-Q load proxy"


def scheduler_rows(sparam_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_energy = s21_mag2(nearest_freq(sparam_rows, "t_surf_rest", ENERGY_FREQ_MHZ))
    baseline_data = s21_mag2(nearest_freq(sparam_rows, "t_surf_rest", DATA_FREQ_MHZ))
    fields = field_by_case(field_rows)
    baseline_field = fields.get("t_surf_rest")
    baseline_q_p95 = q_value(baseline_field, "Q_p95_W_per_m3")
    baseline_q_mean = q_value(baseline_field, "Q_mean_W_per_m3")
    baseline_q_max = q_value(baseline_field, "Q_max_W_per_m3")

    rows: list[dict[str, Any]] = []
    for meta in CANONICAL_CASES:
        case = str(meta["case"])
        energy = nearest_freq(sparam_rows, case, ENERGY_FREQ_MHZ)
        data = nearest_freq(sparam_rows, case, DATA_FREQ_MHZ)
        field = fields.get(case)
        kind = rx_type_value(energy) or str(meta["type"])
        g_norm_raw = normalized(s21_mag2(energy), baseline_energy)
        r_norm = normalized(s21_mag2(data), baseline_data)
        q_p95_norm = normalized(q_value(field, "Q_p95_W_per_m3"), baseline_q_p95)
        q_mean_norm = normalized(q_value(field, "Q_mean_W_per_m3"), baseline_q_mean)
        q_max_norm = normalized(q_value(field, "Q_max_W_per_m3"), baseline_q_max)
        energy_scale = scheduler_energy_scale(kind)
        g_norm = g_norm_raw * energy_scale if isinstance(g_norm_raw, float) else ""
        chi_norm = scheduling_chi_norm(q_p95_norm, kind)
        cap_raw = g_norm_raw if isinstance(g_norm_raw, float) else ""
        cap = min(g_norm, 1.0) if isinstance(g_norm, float) else ""
        rows.append(
            {
                "case": case,
                "label": meta["label"],
                "type": kind,
                "depth_mm": depth_value(energy),
                "condition": meta["condition"],
                "sched_role": meta["sched_role"],
                "sched_include_main": meta.get("sched_include_main", True),
                "s21_energy_db": as_float(energy, "S21_dB"),
                "path_loss_energy_db": path_loss_db(energy),
                "g_norm_raw_s21": g_norm_raw,
                "g_norm": g_norm,
                "s21_data_db": as_float(data, "S21_dB"),
                "path_loss_data_db": path_loss_db(data),
                "r_norm": r_norm,
                "chi_norm_raw_q_p95": q_p95_norm,
                "chi_norm": chi_norm,
                "chi_definition": "sqrt_sampled_Q_p95_norm_surface_or_fixed_implant_scheduler_chi_not_sar",
                "Q_p95_norm_5mm": q_p95_norm,
                "Q_avg_norm_5mm": q_mean_norm,
                "Q_max_norm_5mm": q_max_norm,
                "p_rx_cap_norm_raw": cap_raw,
                "p_rx_cap_norm": cap,
                "scheduler_energy_scale": energy_scale,
                "scheduler_chi_norm_policy": "fixed_1p15_for_implants; sqrt_Q_p95_clipped_4_for_surface",
                "scheduler_calibration_note": scheduler_calibration_note(kind),
                "cap_clipping_rule": "min(scheduler_g_norm, 1.0)",
                "field_proxy_source": "sampled E-field, 5 mm receiver cube",
                "field_proxy_note": FIELD_PROXY_NOTE,
            }
        )
    return rows


def physical_power_map(s21_db: float, p_tx_mw: float) -> dict[str, float]:
    mag2 = 10.0 ** (s21_db / 10.0)
    p_tx_w = p_tx_mw * 1.0e-3
    return {
        "P_tx_mW": p_tx_mw,
        "P_tx_dBm": 10.0 * math.log10(max(p_tx_mw, 1.0e-300)),
        "S21_mag2": mag2,
        "P_rx_W": p_tx_w * mag2,
        "P_rx_uW": p_tx_w * mag2 * 1.0e6,
    }


def physical_anchor_rows(
    sparam_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    p_tx_mw: float,
    energy_buffer_j: float,
) -> list[dict[str, Any]]:
    fields = field_by_case(field_rows)
    reference_q = q_value(fields.get("t_scan_x050"), "Q_p95_W_per_m3")
    rows: list[dict[str, Any]] = []
    for anchor, case in ANCHOR_CASES:
        try:
            row = nearest_freq(sparam_rows, case, ENERGY_FREQ_MHZ)
        except RuntimeError:
            continue
        s21_db = as_float(row, "S21_dB")
        rows.append(
            {
                "anchor": anchor,
                "case": case,
                "type": rx_type_value(row),
                "distance_mm": as_float(row, "rx_x_mm"),
                "depth_mm": depth_value(row),
                "freq_MHz": ENERGY_FREQ_MHZ,
                "path_loss_dB": path_loss_db(row),
                "S21_dB": s21_db,
                **physical_power_map(s21_db, p_tx_mw),
                "Q_p95_W_per_m3": q_value(fields.get(case), "Q_p95_W_per_m3"),
                "reference_Q_p95_W_per_m3": reference_q if math.isfinite(reference_q) else "",
                "B_EM_0p84_budget_W_per_m3": 0.84 * reference_q if math.isfinite(reference_q) else "",
                "E_min_0p35_J": 0.35 * energy_buffer_j,
                "E_safe_0p3695_J": 0.3695 * energy_buffer_j,
                "energy_buffer_note": "short-window scheduling energy account, not total implant battery capacity",
                "field_proxy_note": FIELD_PROXY_NOTE,
            }
        )
    return rows


def sparam_summary_base(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case_value(row),
        "type": rx_type_value(row),
        "condition": row.get("condition", ""),
        "freq_MHz": freq_value(row),
        "S21_dB": as_float(row, "S21_dB"),
        "path_loss_dB": path_loss_db(row),
        "S21_mag2": s21_mag2(row),
    }


def frequency_summary_rows(sparam_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(
        [item for item in sparam_rows if case_value(item) in FREQUENCY_SWEEP_CASES],
        key=lambda item: (case_value(item), freq_value(item)),
    ):
        case = case_value(row)
        reference = nearest_freq(sparam_rows, case, ENERGY_FREQ_MHZ)
        summary = sparam_summary_base(row)
        summary.update(
            {
                "depth_mm": depth_value(row),
                "S21_mag2_norm_to_13p56": normalized(s21_mag2(row), s21_mag2(reference)),
            }
        )
        out.append(summary)
    return out


def distance_summary_rows(sparam_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = field_by_case(field_rows)
    scan_rows = [row for row in sparam_rows if str(row.get("study", "")) == "C_distance_scan" or case_value(row).startswith("t_scan_x")]
    references = {freq_value(row): row for row in scan_rows if case_value(row) == "t_scan_x050"}
    out: list[dict[str, Any]] = []
    for row in sorted(scan_rows, key=lambda item: (freq_value(item), as_float(item, "rx_x_mm"), case_value(item))):
        reference = references.get(freq_value(row))
        summary = sparam_summary_base(row)
        case = case_value(row)
        summary.update(
            {
                "distance_mm": as_float(row, "rx_x_mm"),
                "depth_mm": depth_value(row),
                "S21_mag2_norm_to_x050": normalized(s21_mag2(row), s21_mag2(reference)) if reference else "",
                "Q_p95_W_per_m3": q_value(fields.get(case), "Q_p95_W_per_m3"),
            }
        )
        out.append(summary)
    return out


def depth_summary_rows(sparam_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = field_by_case(field_rows)
    scan_rows = [row for row in sparam_rows if str(row.get("study", "")) == "D_depth_scan" or case_value(row).startswith("t_depth_d")]
    references = {freq_value(row): row for row in scan_rows if case_value(row) == "t_depth_d010"}
    out: list[dict[str, Any]] = []
    for row in sorted(scan_rows, key=lambda item: (freq_value(item), depth_value(item), case_value(item))):
        reference = references.get(freq_value(row))
        summary = sparam_summary_base(row)
        case = case_value(row)
        summary.update(
            {
                "distance_mm": as_float(row, "rx_x_mm"),
                "depth_mm": depth_value(row),
                "S21_mag2_norm_to_d010": normalized(s21_mag2(row), s21_mag2(reference)) if reference else "",
                "Q_p95_W_per_m3": q_value(fields.get(case), "Q_p95_W_per_m3"),
            }
        )
        out.append(summary)
    return out


def write_postprocessed_outputs(
    input_dir: Path = OUT_DIR,
    output_dir: Path | None = None,
    p_tx_mw: float = 1.0,
    energy_buffer_j: float = 1.0e-5,
) -> dict[str, Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    sparam_rows = read_csv(input_dir / "sparams_raw.csv")
    field_rows = read_csv(input_dir / "field_proxy_samples.csv")

    outputs = {
        "scheduler_library": output_dir / "torso_scheduler_library.csv",
        "physical_anchor_table": output_dir / "physical_anchor_table.csv",
        "frequency_summary": output_dir / "torso_frequency_summary.csv",
        "distance_summary": output_dir / "torso_distance_summary.csv",
        "depth_summary": output_dir / "torso_depth_summary.csv",
    }
    write_csv(
        outputs["scheduler_library"],
        scheduler_rows(sparam_rows, field_rows),
        [
            "case",
            "label",
            "type",
            "depth_mm",
            "condition",
            "sched_role",
            "sched_include_main",
            "s21_energy_db",
            "path_loss_energy_db",
            "g_norm_raw_s21",
            "g_norm",
            "s21_data_db",
            "path_loss_data_db",
            "r_norm",
            "chi_norm_raw_q_p95",
            "chi_norm",
            "chi_definition",
            "Q_p95_norm_5mm",
            "Q_avg_norm_5mm",
            "Q_max_norm_5mm",
            "p_rx_cap_norm_raw",
            "p_rx_cap_norm",
            "scheduler_energy_scale",
            "scheduler_chi_norm_policy",
            "scheduler_calibration_note",
            "cap_clipping_rule",
            "field_proxy_source",
            "field_proxy_note",
        ],
    )
    write_csv(
        outputs["physical_anchor_table"],
        physical_anchor_rows(sparam_rows, field_rows, p_tx_mw, energy_buffer_j),
        [
            "anchor",
            "case",
            "type",
            "distance_mm",
            "depth_mm",
            "freq_MHz",
            "path_loss_dB",
            "S21_dB",
            "P_tx_mW",
            "P_tx_dBm",
            "S21_mag2",
            "P_rx_W",
            "P_rx_uW",
            "Q_p95_W_per_m3",
            "reference_Q_p95_W_per_m3",
            "B_EM_0p84_budget_W_per_m3",
            "E_min_0p35_J",
            "E_safe_0p3695_J",
            "energy_buffer_note",
            "field_proxy_note",
        ],
    )
    write_csv(
        outputs["frequency_summary"],
        frequency_summary_rows(sparam_rows),
        [
            "case",
            "type",
            "condition",
            "depth_mm",
            "freq_MHz",
            "S21_dB",
            "path_loss_dB",
            "S21_mag2",
            "S21_mag2_norm_to_13p56",
        ],
    )
    write_csv(
        outputs["distance_summary"],
        distance_summary_rows(sparam_rows, field_rows),
        [
            "case",
            "type",
            "condition",
            "distance_mm",
            "depth_mm",
            "freq_MHz",
            "S21_dB",
            "path_loss_dB",
            "S21_mag2",
            "S21_mag2_norm_to_x050",
            "Q_p95_W_per_m3",
        ],
    )
    write_csv(
        outputs["depth_summary"],
        depth_summary_rows(sparam_rows, field_rows),
        [
            "case",
            "type",
            "condition",
            "distance_mm",
            "depth_mm",
            "freq_MHz",
            "S21_dB",
            "path_loss_dB",
            "S21_mag2",
            "S21_mag2_norm_to_d010",
            "Q_p95_W_per_m3",
        ],
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess solved torso HFSS raw outputs into scheduler-facing CSVs.")
    parser.add_argument("--input-dir", type=Path, default=OUT_DIR, help="Directory containing sparams_raw.csv and field_proxy_samples.csv.")
    parser.add_argument("--out-dir", type=Path, help="Directory for postprocessed CSVs. Defaults to --input-dir.")
    parser.add_argument("--p-tx-mw", type=float, default=1.0, help="Transmit power used in physical_anchor_table.csv.")
    parser.add_argument(
        "--energy-buffer-j",
        type=float,
        default=1.0e-5,
        help="Short-window scheduler energy account used for physical anchor notes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_postprocessed_outputs(args.input_dir, args.out_dir, args.p_tx_mw, args.energy_buffer_j)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
