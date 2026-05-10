from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ansys.aedt.core import Hfss

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hbc_hfss_calibration import (  # noqa: E402
    AEDT_VERSION,
    CASES,
    ENERGY_FREQ_MHZ,
    PROJECT_PATH as BASE_PROJECT_PATH,
    ROOT,
    SETUP_NAME,
    SWEEP_NAME,
    Case,
    local_material_for_case,
    receiver_z_for_case,
)


REVIEW_PROJECT_PATH = ROOT / "projects" / "hbc_hfss_review_experiments.aedt"
REVIEW_DIR = ROOT / "outputs" / "hfss_review"
REVIEW_SAMPLE_DIR = REVIEW_DIR / "field_samples"
FINAL_DIR = ROOT / "outputs" / "final"
FINAL_SAMPLE_DIR = FINAL_DIR / "field_samples"
COPY_PACKAGE = Path(os.environ.get("HBC_HFSS_COPY_PACKAGE", ROOT.parent / "hfss_final_copy_package"))


def ensure_dirs() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main_case(label: str) -> Case:
    for case in CASES:
        if case.label == label:
            return case
    raise RuntimeError(f"Unknown case label: {label}")


def final_cases() -> list[dict[str, Any]]:
    surface_rest = main_case("surface_rest")
    return [
        {
            "label": "surface_rest",
            "raw_label": "surface_rest",
            "case": surface_rest,
            "source_project": "review",
            "source_design": "review_surface_rest",
            "sample_prefix": "surface_rest",
            "sched_role": "baseline",
            "sched_include_main": True,
        },
        {
            "label": "surface_moderate_loose",
            "raw_label": "surface_moderate_loose",
            "case": replace(surface_rest, name="review_surface_loose_005um", label="surface_loose_005um", condition="loose", loose_gap_mm=0.005),
            "source_project": "review",
            "source_design": "review_surface_loose_005um",
            "sample_prefix": "surface_loose_005um",
            "sched_role": "main_surface_degraded",
            "sched_include_main": True,
        },
        {
            "label": "surface_contact_failure",
            "raw_label": "surface_contact_failure",
            "case": main_case("surface_loose"),
            "source_project": "base",
            "source_design": "case02_surface_loose",
            "sample_prefix": "surface_contact_failure",
            "sched_role": "stress_test_only",
            "sched_include_main": False,
        },
        {
            "label": "surface_sweat",
            "raw_label": "surface_sweat",
            "case": main_case("surface_sweat"),
            "source_project": "base",
            "source_design": "case03_surface_sweat",
            "sample_prefix": "surface_sweat",
            "sched_role": "main_surface_wet",
            "sched_include_main": True,
        },
        {
            "label": "implant10_rest",
            "raw_label": "implant10_rest",
            "case": main_case("implant10_rest"),
            "source_project": "review",
            "source_design": "review_implant10_rest",
            "sample_prefix": "implant10_rest",
            "sched_role": "main_implant",
            "sched_include_main": True,
        },
        {
            "label": "implant10_stress",
            "raw_label": "implant10_stress",
            "case": main_case("implant10_stress"),
            "source_project": "base",
            "source_design": "case05_implant10_stress",
            "sample_prefix": "implant10_stress",
            "sched_role": "main_implant_stress",
            "sched_include_main": True,
        },
        {
            "label": "implant30_rest",
            "raw_label": "implant30_rest",
            "case": main_case("implant30_rest"),
            "source_project": "review",
            "source_design": "review_implant30_rest",
            "sample_prefix": "implant30_rest",
            "sched_role": "main_implant",
            "sched_include_main": True,
        },
        {
            "label": "implant30_stress",
            "raw_label": "implant30_stress",
            "case": main_case("implant30_stress"),
            "source_project": "review",
            "source_design": "review_implant30_stress",
            "sample_prefix": "implant30_stress",
            "sched_role": "main_implant_stress",
            "sched_include_main": True,
        },
    ]


def sample_points_mm(case: Case, rx_z_mm: float, side_mm: float, n: int = 5) -> list[list[float]]:
    half = side_mm / 2.0
    if case.rx_type == "surface":
        start = [case.rx_x_mm - half, -half, -side_mm]
        stop = [case.rx_x_mm + half, half, 0.0]
    else:
        start = [case.rx_x_mm - half, -half, rx_z_mm - half]
        stop = [case.rx_x_mm + half, half, rx_z_mm + half]
    pts: list[list[float]] = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                frac = [ix / (n - 1), iy / (n - 1), iz / (n - 1)]
                pts.append([start[axis] + frac[axis] * (stop[axis] - start[axis]) for axis in range(3)])
    return pts


def parse_field_magnitudes(path: Path) -> list[float]:
    mags: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("Complex") or line.startswith("#"):
                continue
            vals = [float(token) for token in line.split()]
            if len(vals) < 9:
                continue
            ex = complex(vals[3], vals[4])
            ey = complex(vals[5], vals[6])
            ez = complex(vals[7], vals[8])
            mags.append(math.sqrt(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2))
    return mags


def export_sampled_field(hfss: Hfss, case: Case, side_mm: float, out: Path, solution: str) -> None:
    rx_z = receiver_z_for_case(case)
    points = [[coord / 1000.0 for coord in point] for point in sample_points_mm(case, rx_z, side_mm)]
    hfss.post.export_field_file(
        quantity="E",
        solution=solution,
        output_file=str(out),
        sample_points=points,
        intrinsics={"Freq": f"{ENERGY_FREQ_MHZ}MHz", "Phase": "0deg"},
        export_in_si_system=True,
    )


def copy_existing_sample(case_info: dict[str, Any], side_mm: float, out: Path) -> bool:
    src = REVIEW_SAMPLE_DIR / f"{case_info['sample_prefix']}_{int(side_mm)}mm.fld"
    if src.exists():
        shutil.copy2(src, out)
        return True
    return False


def ensure_missing_base_samples(case_infos: list[dict[str, Any]]) -> None:
    base_needed = [info for info in case_infos if info["source_project"] == "base"]
    missing: list[tuple[dict[str, Any], float, Path]] = []
    for info in base_needed:
        for side in [5.0, 10.0]:
            out = FINAL_SAMPLE_DIR / f"{info['sample_prefix']}_{int(side)}mm.fld"
            if not out.exists():
                missing.append((info, side, out))
    if not missing:
        return
    hfss = Hfss(
        project=str(BASE_PROJECT_PATH),
        design=missing[0][0]["source_design"],
        solution_type="Modal",
        version=AEDT_VERSION,
        non_graphical=True,
        new_desktop=True,
        close_on_exit=False,
        remove_lock=True,
    )
    try:
        for info, side, out in missing:
            hfss.set_active_design(info["source_design"])
            export_sampled_field(hfss, info["case"], side, out, f"{SETUP_NAME} : {SWEEP_NAME}")
    finally:
        hfss.release_desktop(close_projects=True, close_desktop=True)


def prepare_sample_files(case_infos: list[dict[str, Any]]) -> None:
    for info in case_infos:
        for side in [5.0, 10.0]:
            out = FINAL_SAMPLE_DIR / f"{info['sample_prefix']}_{int(side)}mm.fld"
            if out.exists():
                continue
            if info["source_project"] == "review" and copy_existing_sample(info, side, out):
                continue
    ensure_missing_base_samples(case_infos)


def sampled_stats_for_case(info: dict[str, Any], side: float) -> dict[str, Any]:
    path = FINAL_SAMPLE_DIR / f"{info['sample_prefix']}_{int(side)}mm.fld"
    mags = parse_field_magnitudes(path)
    if not mags:
        raise RuntimeError(f"No sampled field rows parsed from {path}")
    sorted_mags = sorted(mags)
    p95_index = min(len(sorted_mags) - 1, math.ceil(0.95 * len(sorted_mags)) - 1)
    emax = max(mags)
    ep95 = sorted_mags[p95_index]
    eavg = sum(mags) / len(mags)
    sigma = local_material_for_case(info["case"]).conductivity
    return {
        "label": info["label"],
        "source_design": info["source_design"],
        "source_project": info["source_project"],
        "cube_mm": side,
        "sample_count": len(mags),
        "Emax_sampled_V_per_m": emax,
        "Ep95_sampled_V_per_m": ep95,
        "Eavg_sampled_V_per_m": eavg,
        "sigma_S_per_m": sigma,
        "Q_max": sigma * emax * emax,
        "Q_p95": sigma * ep95 * ep95,
        "Q_avg": sigma * eavg * eavg,
        "sample_points_file": str(path),
    }


def write_sampled_proxy(case_infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in case_infos:
        for side in [5.0, 10.0]:
            rows.append(sampled_stats_for_case(info, side))
    baseline = next(row for row in rows if row["label"] == "surface_rest" and float(row["cube_mm"]) == 5.0)
    for metric in ["Q_max", "Q_p95", "Q_avg"]:
        base = float(baseline[metric])
        for row in rows:
            row[f"{metric}_norm"] = float(row[metric]) / base
    write_csv(
        FINAL_DIR / "sampled_field_proxy_final.csv",
        rows,
        [
            "label",
            "source_project",
            "source_design",
            "cube_mm",
            "sample_count",
            "Emax_sampled_V_per_m",
            "Ep95_sampled_V_per_m",
            "Eavg_sampled_V_per_m",
            "sigma_S_per_m",
            "Q_max",
            "Q_p95",
            "Q_avg",
            "Q_max_norm",
            "Q_p95_norm",
            "Q_avg_norm",
            "sample_points_file",
        ],
    )
    return rows


def source_row_by_label(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    for row in rows:
        if row["label"] == label:
            return row
    raise RuntimeError(f"Missing source library row: {label}")


def float_or_blank(value: Any) -> float | str:
    try:
        if value == "":
            return ""
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return out if math.isfinite(out) else ""


def write_libraries(case_infos: list[dict[str, Any]], sampled_rows: list[dict[str, Any]]) -> None:
    review_rows = read_csv(REVIEW_DIR / "calibrated_library_review.csv")
    sampled_5 = {row["label"]: row for row in sampled_rows if float(row["cube_mm"]) == 5.0}
    raw_rows: list[dict[str, Any]] = []
    sched_rows: list[dict[str, Any]] = []
    for info in case_infos:
        src = source_row_by_label(review_rows, info["raw_label"])
        sample = sampled_5[info["label"]]
        g_norm = float(src["g_norm"])
        chi_p95 = float(sample["Q_p95_norm"])
        chi_avg = float(sample["Q_avg_norm"])
        chi_max = float(sample["Q_max_norm"])
        chi_sched = chi_p95
        cap_raw = g_norm / chi_sched if chi_sched > 0 else ""
        cap_clipped = min(cap_raw, 1.0) if cap_raw != "" else ""
        raw = {
            "case": src["case"],
            "label": info["label"],
            "type": src["type"],
            "depth_mm": src["depth_mm"],
            "condition": src["condition"],
            "s21_energy_db": src["s21_energy_db"],
            "path_loss_energy_db": src["path_loss_energy_db"],
            "g_norm": src["g_norm"],
            "s21_data_db": src["s21_data_db"],
            "path_loss_data_db": src["path_loss_data_db"],
            "r_norm": src["r_norm"],
            "field_calculator_Q_rx_max": src["Q_rx_max"],
            "field_calculator_chi_norm": src["chi_norm"],
            "field_calculator_p_rx_cap_norm": src["p_rx_cap_norm"],
            "sampled_Q_max_norm_5mm": chi_max,
            "sampled_Q_p95_norm_5mm": chi_p95,
            "sampled_Q_avg_norm_5mm": chi_avg,
            "sampled_Q_source": "sampled E-field, 5 mm receiver cube, 125 points",
            "sched_role": info["sched_role"],
            "sched_include_main": info["sched_include_main"],
            "source_note": src.get("source_note", ""),
            "review_note": src.get("review_note", ""),
        }
        raw_rows.append(raw)
        sched_rows.append(
            {
                "case": src["case"],
                "label": info["label"],
                "type": src["type"],
                "depth_mm": src["depth_mm"],
                "condition": src["condition"],
                "sched_role": info["sched_role"],
                "sched_include_main": info["sched_include_main"],
                "s21_energy_db": src["s21_energy_db"],
                "path_loss_energy_db": src["path_loss_energy_db"],
                "g_norm": g_norm,
                "s21_data_db": src["s21_data_db"],
                "path_loss_data_db": src["path_loss_data_db"],
                "r_norm": src["r_norm"],
                "chi_norm": chi_sched,
                "chi_definition": "sampled_Q_p95_norm_5mm",
                "Q_p95_norm_5mm": chi_p95,
                "Q_avg_norm_5mm": chi_avg,
                "Q_max_norm_5mm": chi_max,
                "p_rx_cap_norm_raw": cap_raw,
                "p_rx_cap_norm": cap_clipped,
                "cap_clipping_rule": "min(raw_cap, 1.0)",
                "field_proxy_source": "sampled E-field, 5 mm receiver cube, 125 points",
                "field_proxy_note": "Scheduling library uses sampled Q_p95_norm instead of field-calculator point max. surface_contact_failure is stress-test only.",
            }
        )
    raw_fields = [
        "case",
        "label",
        "type",
        "depth_mm",
        "condition",
        "s21_energy_db",
        "path_loss_energy_db",
        "g_norm",
        "s21_data_db",
        "path_loss_data_db",
        "r_norm",
        "field_calculator_Q_rx_max",
        "field_calculator_chi_norm",
        "field_calculator_p_rx_cap_norm",
        "sampled_Q_max_norm_5mm",
        "sampled_Q_p95_norm_5mm",
        "sampled_Q_avg_norm_5mm",
        "sampled_Q_source",
        "sched_role",
        "sched_include_main",
        "source_note",
        "review_note",
    ]
    sched_fields = [
        "case",
        "label",
        "type",
        "depth_mm",
        "condition",
        "sched_role",
        "sched_include_main",
        "s21_energy_db",
        "path_loss_energy_db",
        "g_norm",
        "s21_data_db",
        "path_loss_data_db",
        "r_norm",
        "chi_norm",
        "chi_definition",
        "Q_p95_norm_5mm",
        "Q_avg_norm_5mm",
        "Q_max_norm_5mm",
        "p_rx_cap_norm_raw",
        "p_rx_cap_norm",
        "cap_clipping_rule",
        "field_proxy_source",
        "field_proxy_note",
    ]
    write_csv(FINAL_DIR / "calibrated_library_raw.csv", raw_rows, raw_fields)
    write_csv(FINAL_DIR / "calibrated_library_sched.csv", sched_rows, sched_fields)


def write_summary() -> None:
    sched = read_csv(FINAL_DIR / "calibrated_library_sched.csv")
    lines = [
        "# Final HFSS Library Freeze",
        "",
        "No new physical HFSS cases are added in this step.",
        "",
        "Final scheduling field proxy:",
        "",
        "`chi_norm = sampled_Q_p95_norm_5mm`",
        "",
        "`p_rx_cap_norm_raw = g_norm / chi_norm`",
        "",
        "`p_rx_cap_norm = min(p_rx_cap_norm_raw, 1.0)`",
        "",
        "Main scheduling excludes `surface_contact_failure`; it is retained as `stress_test_only`.",
        "",
        "| label | g_norm | r_norm | chi_norm | p_rx_cap_norm | role | main |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sched:
        lines.append(
            f"| {row['label']} | {float(row['g_norm']):.4g} | {float(row['r_norm']):.4g} | "
            f"{float(row['chi_norm']):.4g} | {float(row['p_rx_cap_norm']):.4g} | "
            f"{row['sched_role']} | {row['sched_include_main']} |"
        )
    (FINAL_DIR / "final_library_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_package() -> None:
    if not COPY_PACKAGE.exists():
        return
    target = COPY_PACKAGE / "final_libraries"
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "calibrated_library_raw.csv",
        "calibrated_library_sched.csv",
        "sampled_field_proxy_final.csv",
        "final_library_summary.md",
    ]:
        shutil.copy2(FINAL_DIR / name, target / name)


def main() -> int:
    try:
        ensure_dirs()
        infos = final_cases()
        prepare_sample_files(infos)
        sampled = write_sampled_proxy(infos)
        write_libraries(infos, sampled)
        write_summary()
        copy_to_package()
        (FINAL_DIR / "finalize_run_summary.txt").write_text("status=completed\n", encoding="utf-8")
        return 0
    except Exception:
        ensure_dirs()
        (FINAL_DIR / "finalize_run_summary.txt").write_text("status=failed\n" + traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
